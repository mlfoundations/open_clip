from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
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
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

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
                use_horovod=self.use_horovod,
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
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=logits.device)

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

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
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
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
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

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
                all_text = torch.distributed.nn.all_gather(text_features)
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
