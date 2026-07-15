from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:
    import torch.distributed.nn
    from torch import distributed as dist

    try:
        from torch.distributed import _functional_collectives as dist_fcol
    except ImportError:
        dist_fcol = None

    has_distributed = True
except ImportError:
    has_distributed = False
    dist_fcol = None


def _all_gather_with_grad(tensor):
    if dist_fcol is not None and hasattr(dist_fcol, "all_gather_tensor_autograd"):
        return dist_fcol.all_gather_tensor_autograd(tensor.contiguous(), gather_dim=0, group=dist.group.WORLD)
    return torch.cat(torch.distributed.nn.all_gather(tensor), dim=0)


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = _all_gather_with_grad(image_features)
        all_text_features = _all_gather_with_grad(text_features)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        if logit_bias is not None:
            logits_per_image += logit_bias
            logits_per_text += logit_bias

        return logits_per_image, logits_per_text

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            logit_bias=None,
            output_dict=False,
    ):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features,
            text_features,
            logit_scale,
            logit_bias=logit_bias,
        )

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # deprecated legacy convenience, see below; pass None with -100 masked labels
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            z_loss_weight=0.0,
            compute_dtype=torch.float32,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        # Preferred contract (pad_id=None): labels arrive with invalid positions already masked to -100
        # (built task-side from the batch text_valid mask; see CoCaTask). pad_id is the legacy value-based
        # convenience: when set, label positions equal to pad_id are additionally ignored -- note this
        # also drops genuine tokens sharing the pad value (e.g. SimpleTokenizer id 0). Default kept at 0
        # for backward compat with external callers passing raw (unmasked) labels.
        self.pad_id = pad_id
        if z_loss_weight < 0:
            raise ValueError(f"z_loss_weight must be non-negative, got {z_loss_weight}")
        self.z_loss_weight = float(z_loss_weight)
        self.compute_dtype = resolve_caption_loss_dtype(compute_dtype)
        # Public module used by the logits path when z-loss is disabled, so replacing it
        # (e.g. label smoothing) keeps working in either loss-compute mode.
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
            self,
            image_features,
            text_features,
            logits=None,
            labels=None,
            logit_scale=None,
            output_dict=False,
            caption_loss_ce=None,
            caption_loss_z=None,
    ):
        """Two ways to supply the caption term:

        * legacy: ``logits`` [B, L, V] + ``labels`` [B, L] -- CE computed here (materialized logits);
        * fused: ``caption_loss_ce`` scalar precomputed by the model via ``fused_linear_cross_entropy``
          (see CoCa/MaMMUT ``forward(labels=...)``), with ``caption_loss_z`` alongside when the model
          also computed the z term -- only the loss weighting is applied here.

        """
        assert logit_scale is not None, 'logit_scale is required'
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=image_features.device)

        if caption_loss_ce is None:
            assert logits is not None and labels is not None, \
                'CoCaLoss needs (logits, labels) when the model does not supply caption_loss_ce'
            if self.pad_id is not None:
                labels = labels.masked_fill(labels == self.pad_id, -100)
            if self.z_loss_weight == 0:
                # Honor the public module while applying the same optional loss-logit upcast as
                # caption_cross_entropy. Reduced diagnostics/objectives remain fp32.
                loss_logits = logits.to(self.compute_dtype) if self.compute_dtype is not None else logits
                caption_loss_ce = self.caption_loss(loss_logits.permute(0, 2, 1), labels).float()
            else:
                caption_loss_ce, caption_loss_z = caption_cross_entropy(
                    logits,
                    labels,
                    ignore_index=-100,
                    z_loss=True,
                    compute_dtype=self.compute_dtype,
                )
        elif self.z_loss_weight and caption_loss_z is None:
            raise ValueError(
                'CoCaLoss has z_loss_weight != 0 but the model-supplied caption term has no '
                'caption_loss_z; pass caption_z_loss=True into the model forward so the fused '
                'loss computes one, or set z_loss_weight=0.')

        # caption_loss_weight scales only the CE term; z applies at exactly z_loss_weight,
        # matching GenLIP/GenLAP semantics for the same flag.
        caption_loss = self.caption_loss_weight * caption_loss_ce
        if self.z_loss_weight:
            caption_loss = caption_loss + self.z_loss_weight * caption_loss_z

        if output_dict:
            output = {"contrastive_loss": clip_loss, "caption_loss": caption_loss}
            output["caption_loss_ce"] = caption_loss_ce.detach()
            if caption_loss_z is not None:
                output["caption_loss_z"] = caption_loss_z.detach()
            return output

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
            chunk_size: int = 0,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        self.chunk_size = chunk_size  # 0 = no chunking (original behavior)
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        if self.chunk_size > 0:
            return self._chunked_loss(image_features, text_features, logit_scale, logit_bias, negative_only)
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def _chunked_loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        """Memory-efficient loss that chunks the logit computation.

        Peak memory: O(chunk_size * N) instead of O(B * N).
        Useful when per-device batch is large (e.g. B > 4096).

        Uses the identities -logsigmoid(-x) = softplus(x) and
        softplus(-x) - softplus(x) = -x to avoid materializing a labels
        tensor: the all-negative loss is softplus(logits), and each diagonal
        positive needs only a -logits[k, i+k] correction.
        """
        B = image_features.shape[0]
        N = text_features.shape[0]
        chunk_size = min(self.chunk_size, B)
        total_loss = torch.zeros((), device=image_features.device, dtype=torch.float32)

        for i in range(0, B, chunk_size):
            end_i = min(i + chunk_size, B)
            img_chunk = image_features[i:end_i]
            logits = self.get_logits(img_chunk, text_features, logit_scale, logit_bias)

            # Treat every pair as negative: -logsigmoid(-logits) == softplus(logits)
            chunk_loss = F.softplus(logits).sum()

            if not negative_only:
                # Replace local positives with positive-pair loss:
                # softplus(-x) - softplus(x) == -x, so subtract the positive logits.
                num_pos = max(0, min(end_i, N) - i)
                if num_pos > 0:
                    rows = torch.arange(num_pos, device=logits.device)
                    pos_logits = logits[rows, i + rows]
                    chunk_loss = chunk_loss - pos_logits.sum()

            total_loss = total_loss + chunk_loss

        return total_loss / B

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            if self.dist_impl == 'bidir':
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right
                    )
                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "shift":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right,
                    )
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left
            elif self.dist_impl == "reduce":
                for i in range(self.world_size):
                    text_from_other = torch.distributed.nn.all_reduce(
                        text_features * (self.rank == i),
                        torch.distributed.ReduceOp.SUM,
                    )
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        text_from_other,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "gather":
                all_text = _all_gather_with_grad(text_features).chunk(self.world_size, dim=0)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        all_text[i],
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                assert False

        return {"contrastive_loss": loss} if output_dict else loss


_CAPTION_LOSS_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "model": None,
}


def resolve_caption_loss_dtype(dtype) -> Optional[torch.dtype]:
    """Resolve the caption-loss mode: fp32 upcast, or None for model/ambient behavior."""
    if dtype is torch.float32:
        return dtype
    if dtype is None:
        return None
    try:
        return _CAPTION_LOSS_DTYPES[str(dtype).lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(_CAPTION_LOSS_DTYPES))
        raise ValueError(f"caption loss compute dtype must be one of {{{choices}}}, got {dtype!r}") from exc


def _caption_ce_z_from_valid_logits(logits, target, z_loss: bool, compute_dtype: Optional[torch.dtype]):
    """Return fp32 CE and squared-log-normalizer sums for already-filtered targets.

    ``compute_dtype`` is the resolved optional fp32 upcast (callers resolve once via
    :func:`resolve_caption_loss_dtype`; this runs per checkpointed chunk). ``z_sum`` is a zero
    scalar when ``z_loss`` is off (kept a tensor for checkpoint-friendly output structure).
    """
    if compute_dtype is not None:
        logits = logits.to(compute_dtype)
    if z_loss:
        # CE = logsumexp(logits) - target_logit. Reusing log_z avoids a second vocabulary reduction
        # when the auxiliary z-loss is enabled.
        log_z = torch.logsumexp(logits, dim=-1)
        target_logits = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        ce_sum = (log_z - target_logits).float().sum()
        z_sum = log_z.float().square().sum()
    else:
        ce_sum = F.cross_entropy(logits, target, reduction="none").float().sum()
        z_sum = ce_sum.new_zeros(())
    return ce_sum, z_sum


def _chunk_linear_ce(hidden, weight, bias, target, z_loss, compute_dtype):
    # F.linear remains in the model's dtype / surrounding AMP context. The optional fp32 upcast applies
    # only to CE/logsumexp after this dominant vocabulary GEMM.
    logits = F.linear(hidden, weight, bias)
    return _caption_ce_z_from_valid_logits(logits, target, z_loss, compute_dtype)


def _graph_tied_zero(hidden, weight, bias):
    """Return an fp32 zero with graph edges to all fused-loss inputs.

    Callers use this with an empty hidden tensor or on meta, so ``hidden.sum()`` reads no storage.
    Scalar parameter selects avoid reducing the full LM head while still producing full-shaped zero
    gradients in backward.
    """
    zero = hidden.sum() + weight[0, 0]
    if bias is not None:
        zero = zero + bias[0]
    return zero.float() * 0.0


@torch.compiler.disable
def fused_linear_cross_entropy(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        chunk_size: int = 4096,
        reduction: str = "mean",
        z_loss: bool = False,
        compute_dtype=torch.float32,
):
    """Memory-efficient linear projection + cross-entropy without materializing full logits.

    Computes ``cross_entropy(linear(hidden, weight, bias), target)`` in chunks over the token dimension,
    materializing only one ``[chunk_size, vocab]`` block at a time. The loss-compute block defaults to fp32
    or can follow the model dtype and ambient autocast policy; returned scalar components are fp32.
    Under autograd each chunk is gradient-checkpointed so the logits are recomputed in backward, bounding
    peak memory to one chunk regardless of batch/sequence length. This mirrors the fused linear
    cross-entropy used by the GenLIP reference (Liger kernel) and is essential for large vocabularies
    (~100k).

    Returns UNWEIGHTED ``(ce, z)`` components -- objective weighting/composition belongs to the loss
    module or task, never here or in a model forward. ``z`` is None when ``z_loss`` is off.

    TODO(torch>=2.13): compare/dispatch to ``F.linear_cross_entropy`` once the env has it — verify it
    bounds memory (not just kernel fusion), matches ignore_index/mean semantics (run
    tests/test_fused_caption_loss.py against it), and beats this path at ~50k vocab before switching.

    Args:
        hidden: ``[N, D]`` features (already flattened over batch/sequence).
        weight: ``[vocab, D]`` projection (e.g. an untied LM head weight).
        target: ``[N]`` token ids; positions equal to ``ignore_index`` are skipped.
        bias: Optional ``[vocab]`` bias.
        chunk_size: Number of tokens per chunk.
        reduction: ``"mean"`` (over non-ignored tokens) or ``"sum"``.
        z_loss: Also compute the mean ``square(logsumexp(logits))`` z component.
        compute_dtype: ``float32`` explicitly upcasts CE/logsumexp inputs; ``model`` (or ``None``)
            leaves their dtype and ambient autocast behavior unchanged.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"unsupported reduction={reduction!r}; expected 'mean' or 'sum'")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    compute_dtype = resolve_caption_loss_dtype(compute_dtype)
    if hidden.is_meta:
        # The valid-mask filtering below is data-dependent, which meta cannot resolve; shape/memory
        # dry runs only need a graph-attached fp32 scalar.
        zero = _graph_tied_zero(hidden, weight, bias)
        return zero, (zero if z_loss else None)
    # Drop ignored (padding) positions before the head GEMM -- they carry zero loss and zero grad,
    # and padded rows are the norm for every caller.
    valid = target != ignore_index
    n_valid = valid.sum()
    hidden = hidden[valid]
    target = target[valid]

    n_tokens = hidden.shape[0]
    use_ckpt = torch.is_grad_enabled() and hidden.requires_grad
    ce_total = hidden.new_zeros((), dtype=torch.float32)
    z_total = hidden.new_zeros((), dtype=torch.float32)
    needs_graph_tie = torch.is_grad_enabled() and (
        hidden.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    )
    if n_tokens == 0 and needs_graph_tie:
        # The chunk loop never runs; keep backward valid and DDP parameters marked as used.
        zero = _graph_tied_zero(hidden, weight, bias)
        ce_total = ce_total + zero
        z_total = z_total + zero
    for start in range(0, n_tokens, chunk_size):
        h_chunk = hidden[start:start + chunk_size]
        t_chunk = target[start:start + chunk_size]
        if use_ckpt:
            ce_chunk, z_chunk = checkpoint(
                _chunk_linear_ce, h_chunk, weight, bias, t_chunk, z_loss, compute_dtype,
                use_reentrant=False,
            )
        else:
            ce_chunk, z_chunk = _chunk_linear_ce(
                h_chunk, weight, bias, t_chunk, z_loss, compute_dtype)
        ce_total = ce_total + ce_chunk
        z_total = z_total + z_chunk
    if reduction == "mean":
        denominator = n_valid.clamp(min=1)
        ce_total = ce_total / denominator
        z_total = z_total / denominator
    return ce_total, (z_total if z_loss else None)


def caption_cross_entropy(
        logits: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int = -100,
        z_loss: bool = False,
        compute_dtype=torch.float32,
):
    """Materialized-logits counterpart of :func:`fused_linear_cross_entropy`.

    Returns unweighted ``(ce, z)`` components like the fused variant; ``z`` is None when
    ``z_loss`` is off. Reductions are masked rather than the logits boolean-indexed: a
    ``logits[valid]`` gather would materialize (and save for backward) a second
    ``[n_valid, vocab]`` tensor, roughly doubling the activation memory of this already
    logits-bound path.
    """
    compute_dtype = resolve_caption_loss_dtype(compute_dtype)
    logits = logits.reshape(-1, logits.shape[-1])
    target = target.reshape(-1)
    valid = target != ignore_index
    n_valid = valid.sum().clamp(min=1)
    if compute_dtype is not None:
        logits = logits.to(compute_dtype)
    if z_loss:
        zero = logits.new_zeros((), dtype=torch.float32)
        log_z = torch.logsumexp(logits, dim=-1)
        target_logits = logits.gather(-1, target.masked_fill(~valid, 0).unsqueeze(-1)).squeeze(-1)
        # torch.where (not multiply-by-mask) keeps padded rows exactly out of the reductions.
        # Like the historical full-tensor cross_entropy(ignore_index) path, padded logits are
        # still reduced over, so they must be finite (extreme magnitudes are fine).
        ce = torch.where(valid, (log_z - target_logits).float(), zero).sum() / n_valid
        z = torch.where(valid, log_z.float().square(), zero).sum() / n_valid
    else:
        ce = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="none")
        ce = ce.float().sum() / n_valid
        z = None
    return ce, z


class GenLipLoss(nn.Module):
    """Pure autoregressive language-modeling loss for GenLIP.

    Next-token cross-entropy over the (already shifted) caption logits/labels. Image patch positions and
    padding tokens are expected to be masked with ``ignore_index`` in the labels by the task. For training,
    prefer the model's built-in fused loss path (see :func:`fused_linear_cross_entropy`) which avoids
    materializing full-vocabulary logits; this module is the simple logits-based variant for standalone use.
    """

    def __init__(
            self,
            ignore_index: int = -100,
            z_loss_weight: float = 0.0,
            compute_dtype=torch.float32,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        if z_loss_weight < 0:
            raise ValueError(f"z_loss_weight must be non-negative, got {z_loss_weight}")
        self.z_loss_weight = float(z_loss_weight)
        self.compute_dtype = resolve_caption_loss_dtype(compute_dtype)

    def forward(self, logits, labels, output_dict: bool = False):
        ce, z = caption_cross_entropy(
            logits, labels,
            ignore_index=self.ignore_index,
            z_loss=bool(self.z_loss_weight),
            compute_dtype=self.compute_dtype,
        )
        loss = ce if z is None else ce + self.z_loss_weight * z
        if not output_dict:
            return loss
        output = {"caption_loss": loss}
        output["caption_loss_ce"] = ce.detach()
        if z is not None:
            output["caption_loss_z"] = z.detach()
        return output
