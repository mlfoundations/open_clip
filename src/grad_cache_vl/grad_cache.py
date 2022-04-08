from typing import List, Union, Callable, Any
from contextlib import nullcontext
from itertools import repeat
from collections import UserDict
import logging

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
        
        # print("random state is {}".format(rnd_states))

        if vl:
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
            # print("after forward no grad, v_reps is size {}, contents {}".format(v_reps.size(), v_reps))
            # print("after forward no grad, l_reps is size {}, contents {}".format(l_reps.size(), l_reps))
            return v_reps, l_reps, s_rep, [rnd_states_v, rnd_states_l]
        else:
            rnd_states = []
            model_reps = []

            with torch.no_grad():
                for x in model_inputs:
                    rnd_states.append(RandContext(*self.get_input_tensors(x)))
                    y = self.model_call(model, x)
                    model_reps.append(self.get_reps(y))
            model_reps = torch.cat(model_reps, dim=0)
            return None, model_reps, None, rnd_states

    def build_cache(self, vl = False, *reps: Tensor, **loss_kwargs) -> [List[Tensor], Tensor]:
        """
        Compute the gradient cache
        :param reps: Computed representations from all encoder models
        :param loss_kwargs: Extra keyword arguments to the loss function
        :return: A tuple of a) gradient cache for each encoder model, and b) loss tensor

        image_features, text_features, logit_scale = model(images, texts)
        total_loss = loss(image_features, text_features, logit_scale)
        """
        reps = [r.detach().requires_grad_() for r in reps]
        with autocast() if self.fp16 else nullcontext():
            loss = self.compute_loss(*reps, **loss_kwargs)

        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        cache = [r.grad for r in reps]

        return cache, loss.detach()

    def build_vl_cache(self, vision: Tensor, language: Tensor, logit_scale) -> [List[Tensor], Tensor]:
        """
        Compute the gradient cache
        :param reps: Computed representations from all encoder models
        :param loss_kwargs: Extra keyword arguments to the loss function
        :return: A tuple of a) gradient cache for each encoder model, and b) loss tensor

        image_features, text_features, logit_scale = model(images, texts)
        total_loss = loss(image_features, text_features, logit_scale)
        """
        # print(vision.size())
        # print(vision[0].grad_fn)
        # for v in vision:
        #     v = v.detach().requires_grad_()
        # for l in language:
        #     l = l.detach().requires_grad_()
        # vl = [v.detach().requires_grad_() for v in vision]
        # print(vision[0].grad_fn)
        # ll = [l.detach().requires_grad_() for l in language]
        # vision = torch.vstack(vl)
        # print(vision[0].grad_fn)
        # language = torch.vstack(ll)
        # print(*img_ft)
        # print(torch.vstack(img_ft))
        #vision_d = vision.detach()
        vision_d = vision
        vision_d.requires_grad_()

        logit_scale = logit_scale.requires_grad_()
        # vision_d.retain_grad()
        # print("vision_d requires grad?")
        # print(vision_d.requires_grad)
        language_d = language
        # language_d = language.detach()
        language_d.requires_grad_()
        # language_d.retain_grad()
        with autocast() if self.fp16 else nullcontext():
            loss = self.loss_fn(vision_d, language_d, logit_scale)
            #print("in cache build, loss shape is {}, vision_d shape is {}, language_d shape is {}".format(loss.shape, vision_d.shape, language_d.shape))
            #print("earlier loss is {}".format(loss))
            #autograd.grad will calculate gradient of one tensor w.r.t. the other
            #detach removed the gradient function itself
            # print("loss is {}, loss grad_fn is {}".format(loss, loss.grad_fn))

        #if self.fp16:
            #self.scaler.scale(loss).backward()
        #else:
            #loss.backward()
        # print("vision_d grad is now {}".format(vision_d.grad))
        # v_prev = torch.ones_like(vision_d)
        # l_prev = torch.ones_like(language_d)
        #caches = autograd.grad(loss, [vision_d, language_d, logit_scale])
        caches = autograd.grad(loss, [vision_d, language_d, logit_scale])
        # print(len(caches), caches)
        v_cache = caches[0]
        l_cache = caches[1]
        s_cache = caches[2]
        # print(s_cache)
        #print('in cache_build, cache shapes are {} {} {}'.format(v_cache.shape, l_cache.shape, s_cache.shape))
        # l_cache = autograd.grad(loss, language_d)
        # v_cache = torch.tensor([v.grad for v in vision_d])
        # print("After build, v_cache is size {}, contents {}".format(len(v_cache), v_cache))
        # print("After build, l_cache is size {}, contents {}".format(len(l_cache), l_cache))
        # l_cache = torch.tensor([l.grad for l in language_d])
        return v_cache, l_cache, s_cache, loss.detach()


    def forward_backward(
            self,
            model: nn.Module,
            model_inputs,
            cached_gradients: List[Tensor],
            random_states: List[RandContext],
            no_sync_except_last: bool = False
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
            sync_contexts = [model.no_sync for _ in range(len(model_inputs) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_inputs))]

        for x, state, gradient, sync_context in zip(model_inputs, random_states, cached_gradients, sync_contexts):
            with sync_context():
                with state:
                    y = self.model_call(model, x)
                reps = self.get_reps(y)
                surrogate = torch.dot(reps.flatten(), gradient.flatten())
                surrogate.backward()

    def forward_backward_vl(
        self,
        model: nn.Module,
        model_inputs,
        v_cache: List[Tensor],
        l_cache: List[Tensor],
        s_cache: List[Tensor],
        v_rnd_st: List[RandContext],
        l_rnd_st: List[RandContext],
        no_sync_except_last: bool = False
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
            #for item in x:
                #print("x_item is {}".format(item))
            with sync_context():
                # print("random state is {}".format(state))
                model_reps = []
                with v_state:
                    with l_state:
                        y = self.model_call(model, x)
                    # print("y is {} length {}".format(type(y), len(y)))
                (v_reps, l_reps, s_reps) = self.get_reps(y)
                # loss = self.loss_fn(v_reps, l_reps, s_reps)
                # print("loss is {}".format(loss))
                # loss.backward()
                # v_list = [tup[0] for tup in model_reps]
                # v_list = v_reps.flatten()
                # l_list = [tup[1] for tup in model_reps]
                # l_list = l_reps.flatten()
                #print("Before backprop, sizes: v_reps {}, l_reps {}, s_reps {}, v_cache {}, l_cache {}, s_cache {}".format(v_reps.shape, l_reps.shape, s_reps.shape, v_cache.unsqueeze(dim=0).shape, l_cache.shape, s_cache.shape))
                #print("v_reps.T contents are {}".format(v_reps.T))
                #print("v_cache.T contents are {}".format(v_cache.T))
                #print("l_reps.T contents are {}".format(l_reps.T))
                #print("l_cache.T contents are {}".format(l_cache.T))
                # surrogate_v = v_reps.T @ v_cache.unsqueeze(dim=0).T
                # surrogate_l = l_reps.T @ l_cache.unsqueeze(dim=0).T
                # ones_vec = torch.ones_like(surrogate_v)
                #v_reps.T.backward(gradient=torch.matmul(v_cache, v_reps))
                #l_reps.T.backward(gradient=torch.matmul(l_cache, l_reps))
                #v_reps.T.backward(gradient=v_reps.T * v_cache)
                #l_reps.T.backward(gradient=l_reps.T * l_cache)
                s_reps.backward(gradient=s_cache, retain_graph=True)
                v_reps.backward(gradient=v_cache, retain_graph=True)
                l_reps.backward(gradient=l_cache)
                # surrogate_v = torch.matmul(v_cache, v_reps)
                #surrogate_l = torch.matmul(l_cache, l_reps)
                # ones_vec = torch.ones_like(surrogate_v)
                #surrogate = (surrogate_v + surrogate_l)/2
                # surrogate = torch.mean(surrogate_v + surrogate_l)
                #model.backward(gradient=[surrogate_v, surrogate_l])
                #autograd.backward(tensors=[v_reps.T, l_reps.T], grad_tensors=[v_cache.flatten(), l_cache.flatten()])
                #surrogate_v = v_reps.T * v_cache
                #surrogate_l = l_reps.T * l_cache
                #surrogate = (surrogate_l + surrogate_v).sum()
                #surrogate.backward()
                # print("contents: surrogate {}, size: surrogate {}".format(surrogate, surrogate.size()))
                #surrogate.backward(gradient=ones_vec)
                #, retain_graph=True
                #surrogate_v.backward(gradient=ones_vec, retain_graph=True)
                #surrogate_l.backward(gradient=ones_vec)
        # print("Backprop complete, returning")

    def cache_step(
            self,
            *model_inputs,
            vl_model: bool = False,
            no_sync_except_last: bool = False,
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
        all_reps = []
        all_rnd_states = []

        if no_sync_except_last:
            assert all(map(lambda m: isinstance(m, nn.parallel.DistributedDataParallel), self.models)), \
                'Some of models are not wrapped in DistributedDataParallel. Make sure you are running DDP with ' \
                'proper initializations.'

        model_inputs = [self.split_inputs(x, chunk_size) for x, chunk_size in zip(model_inputs, self.chunk_sizes)]
        #print('input chunks,',len(model_inputs))
        if vl_model:
            #limit chunk sizes to a single model
            self.chunk_sizes = self.chunk_sizes[:1]
            v_reps = []
            l_reps = []
            #s_reps = []
            all_rnd_states_v = []
            all_rnd_states_l = []
            for x in model_inputs:
                model_v, model_l, model_s, rnd_states = self.forward_no_grad(self.models[0], x, vl=True)
                v_reps.append(model_v)
                l_reps.append(model_l)
                #s_reps.append(model_s)
                all_rnd_states_v.append(rnd_states[0])
                all_rnd_states_l.append(rnd_states[1])
            
            # print("v_reps 0 is {}".format(v_reps[0]))
            # print()
            # print("length of v_reps is {}, size is {}, is leaf is {}, requires grad is {}".format(len(v_reps), v_reps[0].size(), v_reps[0].is_leaf, v_reps[0].requires_grad))
            # print()
            # print(*all_reps[2])
            #logscl = torch.mean(*s_reps)
            # print("logscl is {}".format(logscl))
            v_cache, l_cache, s_cache, loss = self.build_vl_cache(*v_reps, *l_reps, model_s)
            v_cache = v_cache.split(self.chunk_sizes[0], dim=0)
            l_cache = l_cache.split(self.chunk_sizes[0], dim=0)
            # print("chunk size is {}".format(self.chunk_sizes))
            # for c, chunk_size in zip(v_cache.T, self.chunk_sizes):
            #     print("c is now shape {}".format(c.shape))
            #     print("c split is now shape {}".format(c.split(chunk_size)))
            # v_cache = [c.split(chunk_size) for c, chunk_size in zip(v_cache, self.chunk_sizes)]
            # l_cache = [c.split(chunk_size) for c, chunk_size in zip(l_cache, self.chunk_sizes)]
            # print("v_cache is now length {}, v_cache 0 is shape {} v_cache 0 0 is shape {}".format(len(v_cache), v_cache[0].shape, v_cache[0][0].shape))
            for model, x, v_rnd_st, l_rnd_st in zip(
                    self.models, model_inputs, all_rnd_states_v, all_rnd_states_l):
                # print("random state is {}".format(rnd_states))
                # print("model, x, v_cache, l_cache, rnd_states")
                # print(model, x, v_cache, l_cache, rnd_states)
                self.forward_backward_vl(model, x, v_cache, l_cache, s_cache, v_rnd_st, l_rnd_st, no_sync_except_last=no_sync_except_last)
        
        else:
            for model, x in zip(self.models, model_inputs):
                _, model_reps, _, rnd_states = self.forward_no_grad(model, x, vl=False)
                all_reps.append(model_reps)
                all_rnd_states.append(rnd_states)

            cache, loss = self.build_cache(*all_reps, **loss_kwargs)
            cache = [c.split(chunk_size) for c, chunk_size in zip(cache, self.chunk_sizes)]

            for model, x, model_cache, rnd_states in zip(
                    self.models, model_inputs, cache, all_rnd_states):
                self.forward_backward(model, x, model_cache, rnd_states, no_sync_except_last=no_sync_except_last)

        return loss
