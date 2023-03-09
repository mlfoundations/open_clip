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

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss


class MLMLoss(nn.Module):

    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        assert logits.shape[:2] == labels.shape[:2]
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1)

        # only compute loss on masked logits
        mask = (labels != self.ignore_index)
        masked_logits = logits[mask]
        masked_labels = labels[mask]
        return F.cross_entropy(masked_logits, masked_labels, ignore_index=self.ignore_index)


class ITMLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, itm_logits, itm_labels):
        itm_logits = itm_logits.view(-1)
        itm_labels = itm_labels.view(-1)
        return F.binary_cross_entropy_with_logits(itm_logits, itm_labels)


class MAELoss(nn.Module):

    def __init__(self, norm_pix_loss):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss

    def patchify(self, imgs, p):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        assert imgs.shape[2] == imgs.shape[3], 'image must be square'
        assert imgs.shape[2] % p == 0, 'image size must be divisible by patch size'

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        patch_area = pred.shape[-1] / 3
        patch_size = int(patch_area**0.5)
        target = self.patchify(imgs, patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        one_mask = (mask == 1)
        loss = (pred[one_mask] - target[one_mask]) ** 2
        loss = loss.mean()  # mean loss on removed patches
        return loss


class FlavaLoss(ClipLoss):

    def __init__(
        self,
        contrastive_loss_weight,
        itm_loss_weight,
        mlm_loss_weight,
        mae_loss_weight,
        mae_norm_pix_loss,
        *args,
        **kwargs):
        super().__init__(*args, **kwargs)

        self.mlm_loss = MLMLoss()
        self.mae_loss = MAELoss(mae_norm_pix_loss)
        self.itm_loss = ITMLoss()

        self.contrastive_loss_weight = contrastive_loss_weight
        self.itm_loss_weight = itm_loss_weight
        self.mlm_loss_weight = mlm_loss_weight
        self.mae_loss_weight = mae_loss_weight

    def forward_mlm(self, mlm_logits, mlm_labels):
        return {
            "mlm_loss": self.mlm_loss_weight * self.mlm_loss(mlm_logits, mlm_labels)
        }

    def forward_mae(self, image, mae_mask, mae_logits):
        return {
            "mae_loss": self.mae_loss_weight * self.mae_loss(image, mae_logits, mae_mask)
        }

    def forward(
        self,
        *,
        # contrastive
        image_features,
        text_features,
        logit_scale,

        # mae
        image,
        mae_mask,
        mm_mae_logits,

        # mlm
        mlm_labels,
        mm_mlm_logits,

        # itm
        itm_logits,
        itm_labels,
    ):
        clip_loss = super().forward(image_features, text_features, logit_scale)
        itm_loss = self.itm_loss(itm_logits, itm_labels)
        mm_mlm_loss = self.mlm_loss(mm_mlm_logits, mlm_labels)
        mm_mae_loss = self.mae_loss(image, mm_mae_logits, mae_mask)

        clip_loss = self.contrastive_loss_weight * clip_loss
        itm_loss = self.itm_loss_weight * itm_loss
        mm_mlm_loss = self.mlm_loss_weight * mm_mlm_loss
        mm_mae_loss = self.mae_loss_weight * mm_mae_loss

        return {
            "contrastive_loss": clip_loss,
            "itm_loss": itm_loss,
            "mlm_loss": mm_mlm_loss,
            "mae_loss": mm_mae_loss,
        }
