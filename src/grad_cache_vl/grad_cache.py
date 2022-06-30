from typing import List, Union, Callable, Any
from contextlib import nullcontext
from itertools import repeat
from collections import UserDict
import logging
from xmlrpc.client import Boolean

import torch
from torch import autograd
from torch import nn, Tensor
from torch.cuda.amp import GradScaler, autocast

from .context_managers import RandContext

logger = logging.getLogger(__name__)


class GradCache:
    """
    Gradient Cache class. Implements input chunking, first graph-less forward pass, Gradient Cache creation, second
    forward & backward gradient computation. Optimizer step is not included. Native torch automatic mixed precision is
    supported. User needs to handle gradient unscaling and scaler update after a gradeitn cache step.
    """
    def __init__(
            self,
            models: List[nn.Module],
            chunk_sizes: Union[int, List[int]],
            loss_fn: Callable[..., Tensor],
            split_input_fn: Callable[[Any, int], Any] = None,
            get_rep_fn: Callable[..., Tensor] = None,
            fp16: bool = False,
            scaler: GradScaler = None,
            vl_model: bool = False
    ):
        """
        Initialize the Gradient Cache class instance.
        :param models: A list of all encoder models to be updated by the current cache.
        :param chunk_sizes: An integer indicating chunk size. Or a list of integers of chunk size for each model.
        :param loss_fn: A loss function that takes arbitrary numbers of representation tensors and
        arbitrary numbers of keyword arguments as input. It should not in any case modify the input tensors' relations
        in the autograd graph, which are later relied upon to create the gradient cache.
        :param split_input_fn: An optional function that split generic model input into chunks. If not provided, this
        class will try its best to split the inputs of supported types. See `split_inputs` function.
        :param get_rep_fn: An optional function that takes generic model output and return representation tensors. If
        not provided, the generic output is assumed to be the representation tensor.
        :param fp16: If True, run mixed precision training, which requires scaler to also be set.
        :param scaler: A GradScaler object for automatic mixed precision training.
        """
        self.models = models
        if isinstance(chunk_sizes, int):
            self.chunk_sizes = [chunk_sizes for _ in range(len(models))]
        else:
            self.chunk_sizes = chunk_sizes

        self.split_input_fn = split_input_fn
        self.get_rep_fn = get_rep_fn
        self.loss_fn = loss_fn

        if fp16:
            assert scaler is not None, "mixed precision training requires a gradient scaler passed in"

        self.fp16 = fp16
        self.scaler = scaler

        self._get_input_tensors_strict = False

    def __call__(self, *args, **kwargs):
        """
        Call the cache_step function.
        :return: Current step loss.
        """
        return self.cache_step(*args, **kwargs)

    def split_inputs(self, model_input, chunk_size: int) -> List:
        """
        Split input into chunks. Will call user provided `split_input_fn` if specified. Otherwise,
        it can handle input types of tensor, list of tensors and dictionary of tensors.
        :param model_input: Generic model input.
        :param chunk_size:  Size of each chunk.
        :return: A list of chunked model input.
        """
        # delegate splitting to user provided function
        if self.split_input_fn is not None:
            return self.split_input_fn(model_input, chunk_size)

        if isinstance(model_input, (dict, UserDict)) and all(isinstance(x, Tensor) for x in model_input.values()):
            keys = list(model_input.keys())
            chunked_tensors = [model_input[k].split(chunk_size, dim=0) for k in keys]
            return [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

        elif isinstance(model_input, list) and all(isinstance(x, Tensor) for x in model_input):
            chunked_x = [t.split(chunk_size, dim=0) for t in model_input]
            return [list(s) for s in zip(*chunked_x)]

        elif isinstance(model_input, Tensor):
            return list(model_input.split(chunk_size, dim=0))

        elif isinstance(model_input, tuple) and list(map(type, model_input)) == [list, dict]:
            args_chunks = self.split_inputs(model_input[0], chunk_size)
            kwargs_chunks = self.split_inputs(model_input[1], chunk_size)
            return list(zip(args_chunks, kwargs_chunks))

        else:
            raise NotImplementedError(f'Model input split not implemented for type {type(model_input)}')

    def get_input_tensors(self, model_input) -> List[Tensor]:
        """
        Recursively go through model input and grab all tensors, which are then used to record current device random
        states. This method will do its best to parse types of Tensor, tuple, list, dict and UserDict. Other types will
        be ignored unless self._get_input_tensors_strict is set to True, in which case an exception will be raised.
        :param model_input: input to model
        :return: all torch tensors in model_input
        """
        if isinstance(model_input, Tensor):
            return [model_input]

        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])

        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])

        elif self._get_input_tensors_strict:
            raise NotImplementedError(f'get_input_tensors not implemented for type {type(model_input)}')

        else:
            return []

    def model_call(self, model: nn.Module, model_input):
        """
        Literally call the model's __call__ method.
        :param model: model to be called
        :param model_input: input to the model call
        :return: model output
        """
        with autocast() if self.fp16 else nullcontext():
            if isinstance(model_input, Tensor):
                return model(model_input)
            elif isinstance(model_input, list):
                return model(*model_input)
            elif isinstance(model_input, (dict, UserDict)):
                return model(**model_input)
            elif isinstance(model_input, tuple) and list(map(type, model_input)) == [list, dict]:
                model_args, model_kwargs = model_input
                return model(*model_args, **model_kwargs)
            else:
                raise NotImplementedError

    def get_reps(self, model_out) -> Tensor:
        """
        Return representation tensor from generic model output
        :param model_out: generic model output
        :return: a single tensor corresponding to the model representation output
        """
        if self.get_rep_fn is not None:
            return self.get_rep_fn(model_out)
        else:
            return model_out

    def compute_loss(self, *reps: Tensor, **loss_kwargs) -> Tensor:
        """
        Compute the loss based on the representation tensors. The tensors should be ordered same as the list of models
        registered in this GradCache class instance.
        :param reps: Representations for computing the loss.
        :param loss_kwargs: Keyword arguments input to the loss function.
        :return: the loss tensor.
        """
        loss = self.loss_fn(*reps, **loss_kwargs)
        return loss

    def forward_no_grad(
            self,
            model: nn.Module,
            model_inputs,
            vl=False
    ):
        """
        The first forward pass without gradient computation.
        :param model: Encoder model.
        :param model_inputs: Model input already broken into chunks.
        :return: A tuple of a) representations and b) recorded random states.
        """
        rnd_states_v = []
        rnd_states_l = []
        v_reps = []
        l_reps = []
        s_rep = None
        with torch.no_grad():
            for x in model_inputs:
                rnd_states_v.append(RandContext(*self.get_input_tensors(x[0])))
                rnd_states_l.append(RandContext(*self.get_input_tensors(x[1])))
                y = self.model_call(model, x)
                (v_rep, l_rep, s_rep) = self.get_reps(y)
                v_reps.append(v_rep)
                l_reps.append(l_rep)
        v_reps = torch.cat(v_reps, dim=0)
        l_reps = torch.cat(l_reps, dim=0)
        return v_reps, l_reps, s_rep, [rnd_states_v, rnd_states_l]

    def build_vl_cache(self, vision: Tensor, language: Tensor, logit_scale, lock_img: bool) -> [List[Tensor], Tensor]:
        """
        Compute the gradient cache
        :param reps: Computed representations from all encoder models
        :param loss_kwargs: Extra keyword arguments to the loss function
        :return: A tuple of a) gradient cache for each encoder model, and b) loss tensor

        image_features, text_features, logit_scale = model(images, texts)
        total_loss = loss(image_features, text_features, logit_scale)
        """
        vision_d = vision.requires_grad_()
        logit_scale = logit_scale.requires_grad_()
        language_d = language.requires_grad_()
        autocast = torch.cuda.amp.autocast if self.fp16 else nullcontext
        with autocast():
            loss = self.compute_loss(vision_d, language_d, logit_scale)
            if self.scaler is not None:
                old_loss = loss
                #logging.debug("scaling, unscaled loss is {}".format(loss))
                loss = self.scaler.scale(loss)
                #logging.debug("scaled loss is {}".format(loss))
        #TODO: horovod, amp+distributed
        (v_cache, l_cache, s_cache) = autograd.grad(loss, [vision_d, language_d, logit_scale])
        if self.scaler is not None:
            return v_cache, l_cache, logit_scale.clone().detach(), old_loss.clone().detach()
        else:
            return v_cache, l_cache, s_cache, loss.detach()

    def forward_backward_vl(
        self,
        model: nn.Module,
        model_inputs,
        v_cache: List[Tensor],
        l_cache: List[Tensor],
        s_cache: List[Tensor],
        v_rnd_st: List[RandContext],
        l_rnd_st: List[RandContext],
        no_sync_except_last: bool = False,
        lock_img: bool = False
        ):
        """
        Run the second forward and the backward pass to compute gradient for a model.
        :param model: Encoder model.
        :param model_inputs: Chunked input to the encoder model.
        :param cached_gradients: Chunked gradient cache tensor for each input.
        :param random_states: Each input's device random state during the first forward.
        :param no_sync_except_last: If True, under distributed setup, only trigger gradient reduction across processes
        for the last sub-batch's forward-backward pass.
        """
        if no_sync_except_last:
            sync_contexts = [model.no_sync for _ in range(len(v_cache) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(v_cache))]
        for x, v_state, l_state, v_cache, l_cache, sync_context in zip(model_inputs, v_rnd_st, l_rnd_st, v_cache, l_cache, sync_contexts):
            with sync_context():
                with v_state:
                    with l_state:
                        y = self.model_call(model, x)
                (v_reps, l_reps, s_reps) = self.get_reps(y)
                if v_reps.requires_grad:
                    autograd.backward(tensors=[v_reps, l_reps, s_reps], grad_tensors=[v_cache, l_cache, s_cache])
                else:
                    autograd.backward(tensors=[l_reps, s_reps], grad_tensors=[l_cache, s_cache])

        return s_reps

    def cache_step(
            self,
            *model_inputs,
            vl_model: bool = False,
            no_sync_except_last: bool = False,
            lock_img: bool = False,
            **loss_kwargs
    ) -> Tensor:
        """
        Run a cached step to compute gradient over the inputs.
        :param model_inputs: Input to each encoder model. Should be in similar order as the class's model.
        :param no_sync_except_last: If True, under distributed setup, for each model, only trigger gradient reduction
        across processes for the last sub-batch's forward-backward pass.
        :param loss_kwargs: Additional keyword arguments to the loss function.
        :return: The current's loss.
        """
        if not vl_model:
            raise NotImplementedError
        if no_sync_except_last:
            assert all(map(lambda m: isinstance(m, nn.parallel.DistributedDataParallel), self.models)), \
                'Some of models are not wrapped in DistributedDataParallel. Make sure you are running DDP with ' \
                'proper initializations.'

        model_inputs = [self.split_inputs(x, chunk_size) for x, chunk_size in zip(model_inputs, self.chunk_sizes)]
        #limit chunk sizes to a single model
        self.chunk_sizes = self.chunk_sizes[:1]
        v_reps = []
        l_reps = []
        all_rnd_states_v = []
        all_rnd_states_l = []
        for x in model_inputs:
            model_v, model_l, model_s, rnd_states = self.forward_no_grad(self.models[0], x, vl=True)
            v_reps.append(model_v)
            l_reps.append(model_l)
            all_rnd_states_v.append(rnd_states[0])
            all_rnd_states_l.append(rnd_states[1])
        v_cache, l_cache, s_cache, loss = self.build_vl_cache(*v_reps, *l_reps, model_s, lock_img=lock_img)
        v_cache = v_cache.split(self.chunk_sizes[0], dim=0)
        l_cache = l_cache.split(self.chunk_sizes[0], dim=0)
        for model, x, v_rnd_st, l_rnd_st in zip(
                self.models, model_inputs, all_rnd_states_v, all_rnd_states_l):
            self.forward_backward_vl(model, x, v_cache, l_cache, s_cache, v_rnd_st, l_rnd_st, no_sync_except_last=no_sync_except_last, lock_img=lock_img)
        # logging.debug("types: vcache, {} lcache {} scache {} loss {}".format(type(v_cache), type(l_cache), type(s_cache), type(loss)))
        # logging.debug("requires grad: vcache, {} lcache {} scache {} loss {}".format(v_cache[0].requires_grad, l_cache[0].requires_grad, s_cache.requires_grad, loss.requires_grad))
        return loss, s_cache
