import time
import random
import numpy as np

import torch
import torch.distributed.nn
from torch import distributed as dist, nn as nn
from torch.nn import functional as F

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


class DGAClipLoss:

    def __init__(self, args, device, rank, epoch, num_batches_per_epoch, autocast):
        self.device = device
        self.rank = rank
        self.autocast = autocast
        self.epoch = epoch
        self.num_batches_per_epoch = num_batches_per_epoch

        self.batch_size = args.batch_size
        self.batch_size_train = args.batch_size_train

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True 

    def __call__(self, model, scaler, images, texts):
        # generate and setup random seed, thus forwarding a sample twice will produce the same embedding
        stable_random_seed = int(time.time() * 1000 % 1000000)
        self.setup_seed(stable_random_seed + self.rank)

        # first time forward without grad 
        with torch.no_grad():
            model.requires_grad_(False)  # for avoiding DDP bug
            image_embeddings_local, text_embeddings_local = [], []

            for _idx_l in range(0, self.batch_size, self.batch_size_train):
                _images = images[_idx_l: _idx_l + self.batch_size_train]
                _texts = texts[_idx_l: _idx_l + self.batch_size_train]

                with self.autocast():
                    _image_embeddings, _text_embeddings, logit_scale = model(_images, _texts)

                image_embeddings_local.append(_image_embeddings)
                text_embeddings_local.append(_text_embeddings)
            
            # (i, d), (t, d)
            image_embeddings_local = torch.cat(image_embeddings_local, dim = 0)
            text_embeddings_local = torch.cat(text_embeddings_local, dim = 0)
            
            logit_scale_sqrt = torch.sqrt(logit_scale)

            # (i, d)
            image_embeddings_global = torch.cat(torch.distributed.nn.all_gather(image_embeddings_local), dim=0)
            # (t, d)
            text_embeddings_global = torch.cat(torch.distributed.nn.all_gather(text_embeddings_local), dim=0)
    
            s_i2t_nm = image_embeddings_global @ text_embeddings_local.T
            s_i2t_mn = image_embeddings_local @ text_embeddings_global.T

            # (i, t'), (i', t)
            s_i2t_nm *= logit_scale
            s_i2t_mn *= logit_scale
            
            # (i), (t)
            targets_i2t = torch.arange(self.batch_size * self.rank, self.batch_size * (self.rank + 1), device=self.device)
            targets_t2i = torch.arange(self.batch_size * self.rank, self.batch_size * (self.rank + 1), device=self.device)

            total_loss = 0.5 * (F.cross_entropy(s_i2t_mn, targets_i2t) + F.cross_entropy(s_i2t_nm.T, targets_t2i)).cpu()
            
            # (i'), (t')
            s_i2t_esum_local = torch.sum(torch.exp(s_i2t_mn), dim = 1)
            s_t2i_esum_local = torch.sum(torch.exp(s_i2t_nm.T), dim = 1)
            
            # (i), (t)
            s_i2t_esum = torch.cat(torch.distributed.nn.all_gather(s_i2t_esum_local), dim=0).unsqueeze(dim = 1)
            s_t2i_esum = torch.cat(torch.distributed.nn.all_gather(s_t2i_esum_local), dim=0).unsqueeze(dim = 1)

            p_i2t_mn = torch.exp(s_i2t_mn) / s_i2t_esum[self.batch_size * self.rank: self.batch_size * (self.rank + 1), :]
            p_t2i_nm = torch.exp(s_i2t_mn.T) / s_t2i_esum
            left_I = (p_i2t_mn + p_t2i_nm.T) @ text_embeddings_global - text_embeddings_local * 2
            
            p_i2t_nm = torch.exp(s_i2t_nm) / s_i2t_esum
            p_t2i_mn = torch.exp(s_i2t_nm.T) / s_t2i_esum[self.batch_size * self.rank: self.batch_size * (self.rank + 1), :]
            left_T = (p_i2t_nm.T + p_t2i_mn) @ image_embeddings_global - image_embeddings_local * 2
            
            # (i, d) = (1) * ((i, t) @ (t, d))
            left_I *= logit_scale_sqrt
            left_T *= logit_scale_sqrt

            model.requires_grad_(True)  # # for avoiding DDP bug
        
        self.setup_seed(stable_random_seed + self.rank)

        # second time forward with grad
        for _idx_l in range(0, self.batch_size, self.batch_size_train):

            _images = images[_idx_l: _idx_l + self.batch_size_train]
            _texts = texts[_idx_l: _idx_l + self.batch_size_train]

            with self.autocast():
                # (i', d), (t', d)
                _image_embeddings, _text_embeddings, logit_scale = model(_images, _texts)

            # (i', d), (t', d)
            _left_I = left_I[_idx_l: _idx_l + self.batch_size_train]
            _left_T = left_T[_idx_l: _idx_l + self.batch_size_train]
            
            logit_scale_sqrt = torch.sqrt(logit_scale)

            # (i')
            loss_temp_i = _left_I * _image_embeddings
            loss_temp_t = _left_T * _text_embeddings

            loss_temp = (loss_temp_i + loss_temp_t).sum() / 2 / self.batch_size
            loss_temp = loss_temp * logit_scale_sqrt
            
            # backward each sub-iteration
            if scaler is not None:
                scaler.scale(loss_temp).backward()
            else:
                loss_temp.backward()

        self.total_loss = total_loss

        return self, logit_scale
    
    def backward(self, grad):
        pass

    def item(self):
        return self.total_loss.item()