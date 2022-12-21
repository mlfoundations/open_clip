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
        features_1,
        features_2,
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
            all_features_1 = hvd.allgather(features_1)
            all_features_2 = hvd.allgather(features_2)
        else:
            with torch.no_grad():
                all_features_1 = hvd.allgather(features_1)
                all_features_2 = hvd.allgather(features_2)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features_1 = list(all_features_1.chunk(world_size, dim=0))
                gathered_features_2 = list(all_features_2.chunk(world_size, dim=0))
                gathered_features_1[rank] = features_1
                gathered_features_2[rank] = features_2
                all_features_1 = torch.cat(gathered_features_1, dim=0)
                all_features_2 = torch.cat(gathered_features_2, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_features_1 = torch.cat(torch.distributed.nn.all_gather(features_1), dim=0)
            all_features_2 = torch.cat(torch.distributed.nn.all_gather(features_2), dim=0)
        else:
            gathered_features_1 = [torch.zeros_like(features_1) for _ in range(world_size)]
            gathered_features_2 = [torch.zeros_like(features_2) for _ in range(world_size)]
            dist.all_gather(gathered_features_1, features_1)
            dist.all_gather(gathered_features_2, features_2)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features_1[rank] = features_1
                gathered_features_2[rank] = features_2
            all_features_1 = torch.cat(gathered_features_1, dim=0)
            all_features_2 = torch.cat(gathered_features_2, dim=0)

    return all_features_1, all_features_2


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

    def forward(self, features_1, features_2, logit_scale):
        device = features_1.device
        if self.world_size > 1:
            all_features_1, all_features_2 = gather_features(
                features_1, features_2,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_feature_1 = logit_scale * features_1 @ all_features_2.T
                logits_per_feature_2 = logit_scale * features_2 @ all_features_1.T
            else:
                logits_per_feature_1 = logit_scale * all_features_1 @ all_features_2.T
                logits_per_feature_2 = logits_per_feature_1.T
        else:
            logits_per_feature_1 = logit_scale * features_1 @ features_2.T
            logits_per_feature_2 = logit_scale * features_2 @ features_1.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_feature_1.shape[0]
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
            F.cross_entropy(logits_per_feature_1, labels) +
            F.cross_entropy(logits_per_feature_2, labels)
            ) / 2
        return total_loss
