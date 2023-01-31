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
        features_a,
        features_b,
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
            all_features_a = hvd.allgather(features_a)
            all_features_b = hvd.allgather(features_b)
        else:
            with torch.no_grad():
                all_features_a = hvd.allgather(features_a)
                all_features_b = hvd.allgather(features_b)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features_a = list(all_features_a.chunk(world_size, dim=0))
                gathered_features_b = list(all_features_b.chunk(world_size, dim=0))
                gathered_features_a[rank] = features_a
                gathered_features_b[rank] = features_b
                all_features_a = torch.cat(gathered_features_a, dim=0)
                all_features_b = torch.cat(gathered_features_b, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_features_a = torch.cat(torch.distributed.nn.all_gather(features_a), dim=0)
            all_features_b = torch.cat(torch.distributed.nn.all_gather(features_b), dim=0)
        else:
            gathered_features_a = [torch.zeros_like(features_a) for _ in range(world_size)]
            gathered_features_b = [torch.zeros_like(features_b) for _ in range(world_size)]
            dist.all_gather(gathered_features_a, features_a)
            dist.all_gather(gathered_features_b, features_b)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features_a[rank] = features_a
                gathered_features_b[rank] = features_b
            all_features_a = torch.cat(gathered_features_a, dim=0)
            all_features_b = torch.cat(gathered_features_b, dim=0)

    return all_features_a, all_features_b


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


    def forward(self, features_a, features_b, logit_scale,output_dict=False):
        device = features_a.device

        if self.world_size > 1:
            all_features_a, all_features_b = gather_features(
                features_a, features_b,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_feature_a = logit_scale * features_a @ all_features_b.T
                logits_per_feature_b = logit_scale * features_b @ all_features_a.T
            else:
                logits_per_feature_a = logit_scale * all_features_a @ all_features_b.T
                logits_per_feature_b = logits_per_feature_a.T
        else:
            logits_per_feature_a = logit_scale * features_a @ features_b.T
            logits_per_feature_b = logit_scale * features_b @ features_a.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_feature_a.shape[0]
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
        clip_loss = super().forward(image_features, text_features, logit_scale)
        clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss

