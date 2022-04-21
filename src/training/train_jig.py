import logging
import math
import os
from contextlib import suppress

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from grad_cache import GradCache
from open_clip import ClipLoss

from .device import DeviceEnv
from .loss import LossCfg
from .optim import OptimCfg
from .utils import unwrap_model


class CacheWrapper(torch.nn.Module):

    def __init__(self, type: str, base_model: torch.nn.Module):
        super().__init__()
        if type == 'text':
            self.tower = base_model.text
            self.logit_scale = None
        else:
            self.tower = base_model.visual
            self.logit_scale = base_model.logit_scale

    def forward(self, x):
        rep = self.tower(x)
        rep = F.normalize(rep, dim=-1)
        if self.logit_scale is not None:
            # logit scale applied via image features when using GC
            rep = rep.mul_(self.logit_scale.exp())
        return rep


# class CacheWrapper(torch.nn.Module):
#
#     def __init__(self, model: torch.nn.Module):
#         super().__init__()
#         self.model = model
#
#     def forward(self, image, text):
#         image, text, logit_scale = self.model(image, text)
#         image.mul_(logit_scale)
#         return image, text


class TrainJig:

    def __init__(
            self,
            model: nn.Module,
            dev_env: DeviceEnv,
            loss_cfg: LossCfg,
            optim_cfg: OptimCfg,
            grad_cache_chunk_size: int = 0,
    ):
        super().__init__()

        self.epoch = 0
        self.model = model
        self.dev_env = dev_env
        self.loss = ClipLoss(
            local_loss=loss_cfg.local_loss,
            gather_with_grad=loss_cfg.gather_with_grad,
            cache_labels=loss_cfg.cache_labels,
            rank=dev_env.rank,
            world_size=dev_env.world_size,
            use_horovod=dev_env.horovod)

        self.wrapped_models = []
        self.gc = None
        self.optimizer = None
        self.scaler = torch.cuda.amp.GradScaler() if dev_env.amp else None
        self.autocast = torch.cuda.amp.autocast if dev_env.amp else suppress

        ddp_args = {}
        if dev_env.static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True

        if grad_cache_chunk_size:
            self.wrapped_models = [
                CacheWrapper('image', self.model),
                CacheWrapper('text', self.model),
            ]
            if dev_env.ddp:
                assert not dev_env.sync_bn, 'Synchronzied BN + GradCache not tested/supported.'
                self.wrapped_models = [
                    torch.nn.parallel.DistributedDataParallel(m, device_ids=[dev_env.device], **ddp_args)
                    for m in self.wrapped_models]
            self.gc = GradCache(
                models=self.wrapped_models,
                chunk_sizes=[grad_cache_chunk_size, grad_cache_chunk_size],
                scaler=self.scaler,
                fp16=dev_env.amp,
                loss_fn=self.loss,
            )
        else:
            if dev_env.ddp:
                if dev_env.sync_bn:
                    self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

                self.model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[dev_env.device], **ddp_args)

        self._setup_optimizer(optim_cfg)

    def _setup_optimizer(self, optim_cfg: OptimCfg):
        assert optim_cfg.type == 'adamw'
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)
        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": optim_cfg.wd},
            ],
            lr=optim_cfg.lr,
            betas=optim_cfg.betas,
            eps=optim_cfg.eps,
        )
        if self.dev_env.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=self.model.named_parameters())
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        self.optimizer = optimizer

    def _train_step(self, images, texts):

        with self.autocast():
            image_features, text_features, logit_scale = self.model(images, texts)
            total_loss = self.loss(image_features, text_features, logit_scale)

        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            if self.dev_env.horovod:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
            else:
                self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        return total_loss

    def _train_step_gc(self, images, texts):
        total_loss = self.gc(images, texts)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        return total_loss

    def start_train_epoch(self):
        self.model.train()
        return self.epoch

    def end_train_epoch(self):
        self.epoch += 1
        return self.epoch

    def train_step(self, images, texts):
        self.optimizer.zero_grad()

        if self.gc is None:
            loss = self._train_step(images, texts)
        else:
            loss = self._train_step_gc(images, texts)

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            logit_scale = unwrap_model(self.model).logit_scale
            logit_scale.clamp_(0, math.log(100))

        return dict(total_loss=loss.detach(), logit_scale=logit_scale.detach().exp())

    def eval_step(self, batch):
        pass

    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def resume(self, path):
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location=self.dev_env.device)
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                self.epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not self.dev_env.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                self.model.load_state_dict(sd)
                if self.optimizer is not None:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                if self.scaler is not None and 'scaler' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler'])
                logging.info(f"=> resuming checkpoint '{path}' (epoch {self.epoch})")
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                self.model.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{path}' (epoch {self.epoch})")
        else:
            logging.info("=> no checkpoint found at '{}'".format(path))

    def state_dict(self, name=''):
        state_dict = {
            "epoch": self.epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if name:
            state_dict['name'] = name
        if self.scaler is not None:
            state_dict["scaler"] = self.scaler.state_dict()
        return state_dict
