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
        masked_idx = torch.where(labels != self.ignore_index)
        masked_logits = logits[masked_idx]
        masked_labels = labels[masked_idx]

        return F.cross_entropy(masked_logits, masked_labels, ignore_index=self.ignore_index)


class ITMLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, itm_logits, itm_labels):
        itm_logits = itm_logits.view(-1)
        itm_labels = itm_labels.view(-1)
        return F.binary_cross_entropy_with_logits(itm_logits, itm_labels)


class FlavaLoss(ClipLoss):

    def __init__(
        self,
        contrastive_loss_weight,
        itm_loss_weight,
        mlm_loss_weight,
        mae_loss_weight,
        *args,
        **kwargs):
        super().__init__(*args, **kwargs)

        self.mlm_loss = MLMLoss()
        self.itm_loss = ITMLoss()

        self.contrastive_loss_weight = contrastive_loss_weight
        self.itm_loss_weight = itm_loss_weight
        self.mlm_loss_weight = mlm_loss_weight
        self.mae_loss_weight = mae_loss_weight

    def forward(
        self,
        *,
        image_features,
        text_features,
        logit_scale,
        # text_masked_logits,
        text_masked_labels,
        itm_logits,
        itm_labels,
        mm_masked_logits,
    ):
        clip_loss = super().forward(image_features, text_features, logit_scale)
        itm_loss = self.itm_loss(itm_logits, itm_labels)
        mm_mlm_loss = self.mlm_loss(mm_masked_logits, text_masked_labels)
        # TODO: add MAE loss

        clip_loss = self.contrastive_loss_weight * clip_loss
        itm_loss = self.itm_loss_weight * itm_loss
        mm_mlm_loss = self.mlm_loss_weight * mm_mlm_loss

        return {
            "contrastive_loss": clip_loss,
            "itm_loss": itm_loss,
            "mlm_loss": mm_mlm_loss,
        }
