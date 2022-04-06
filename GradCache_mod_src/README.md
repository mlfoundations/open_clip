



# Gradient Cache
Gradient Cache is a simple technique for unlimitedly scaling contrastive learning batch far beyond GPU/TPU memory constraint. This means training that used to take heavy hardware, e.g. 8 V100 GPU, can be done on a single GPU. In addition, Gradient Cache allow users to replace big RAM GPU/TPU with much more cost efficient high FLOP low RAM systems.

This repo holds a generic implementation of Gradient Cache described in our paper [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup
](https://arxiv.org/abs/2101.06983). Both Pytorch and JAX frameworks are supported.
```
@inproceedings{gao2021scaling,
     title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
     author={Luyu Gao, Yunyi Zhang, Jiawei Han, Jamie Callan},
     booktitle ={Proceedings of the 6th Workshop on Representation Learning for NLP},
     year={2021},
}
```

**NEW: We now support JAX and TPU!**

Gradient Cache has also been integrated into dense passage retrieval (DPR). Checkout our [GC-DPR toolkit](https://github.com/luyug/GC-DPR).
## Installation
First install your desired deep learning backend, either Pytorch or JAX.  To install GradCache, clone this repo and run pip.
```
git clone https://github.com/luyug/GradCache
cd GradCache
pip install .
```
For development,
```
pip install --editable .
```

## Usage
Gradient caching functionalities are implemented in `GradCache` class.  If you are developing a **new project** instead of patching an old one, also checkout our [functional approach](#functional-approach) for a effort reduced approach. 

For JAX/Flax user, take a look at a simple train function [here](https://github.com/luyug/GradCache/blob/8463340a15a2395fc33b9a1f40f5f4946b7cbad8/src/grad_cache/cachex/training.py#L9).

### Initialization
The class's `__init__` method defines the cache and has several functional parameters `*_fn` for easy adjust of model behaviors. Alternatively you can also sub-class GradCache.
```
grad_cache.GradCache(  
  models: List[nn.Module],  
  chunk_sizes: Union[int, List[int]],  
  loss_fn: Callable[..., Tensor],  
  split_input_fn: Callable[[Any, int], Any] = None,  
  get_rep_fn: Callable[..., Tensor] = None,  
  fp16: bool = False,  
  scaler: GradScaler = None,  
)
``` 
**models** - A list of encoder models to be updated with with the Gradient Cache.

**chunk_sizes** - An integer indicating chunk size. Or a list of integers of chunk size for each model. This controls for each model the sub-batch size to run forward-backward pass and should be set based on available GPU memory. A value too small will leave the GPU under utilized.

**loss_fn** -  A loss function that takes representation tensors of number equal to number of models in `models` and arbitrary numbers of keyword arguments. It should compute loss based on the input tensors, and in no case modify the input tensors' relations in the autograd graph, which are later relied upon to create the gradient cache.

**split_input_fn** - An optional function that split generic model input into chunks based on defined chunk_sizes. If not provided, this  class will try its best to split the inputs of supported types. See `split_inputs` function.

**get_rep_fn** - An optional function that takes generic model output and return representation tensors. If  not provided, the generic output is assumed to be the representation tensor.

**fp16** - If True, run mixed precision training, which requires scaler to also be set.

**scaler** - A GradScaler object for automatic mixed precision training.

### Cache Gradient Step
To run a cached gradient computatoin step, call `cache_step` function,

```
cache_step(  
  *model_inputs,  
  no_sync_except_last: bool = False,  
  **loss_kwargs  
)
```
Run a single gradient cache step. Upon function return, updates are computed for each model in `self.models` with gradient populated on the weights, as if the `model_inputs` are run as a huge single batch on sufficiently large hardware.  Calling an GradCache object with `__call__` will also invoke this function.

**model_inputs** - List of inputs to each encoder model. Should be in similar order as `self.models`.

**no_sync_except_last** - If True, under distributed setup, for each model, only trigger gradient reduction across processes for the last sub-batch's forward-backward pass. This could come in handy when dealing with a) large model, and/or b) non trivial number of sub-batches.

**loss_kwargs** - Additional keyword arguments to the loss function `loss_fn`. This is intended to enable flexible loss computation (thanks to dynamic graph in Pytorch) such as reduction, weighting, etc. Potentially, using `loss_kwargs` you can incorporate outputs from those encoder models not tracked by the cache. 

**Return** - loss, the current steps loss scaler tensor (detached from the graph).

### Natively Supported Input Types
- x: Tensor - will be passed in as `model(x)`
- x: List[Tensor] - will be passed in as `model(*x)`
- x: Dict[str, Tensor] (or UserDict[str, Tensor]) - will be passed in as `model(**x)`
- x: Tuple[List[Tensor], Dict[str, Tensor]] - will be passed in as `model(*x[0], **x[1])`

Other generic input are not fully supported, we perform model call using the following heuristics,

- x: List[Any] - will be passed in as `model(*x)`
- x: Dict[str, Any] - will be passed in as `model(**x)`
- x: Tuple[List[Any], Dict[str, Any]] - will be passed in as `model(*x[0], **x[1])`

To run with them, `split_input_fn` should be specified during cache initialization to break these inputs  into smaller batches.  In some rare cases, you may also need to override  `get_input_tensors` when its heuristic can not grab enough tensors that covers all cuda devices that hold some tensors in the input.


## Example Usage with Huggingface Transformers
### Learning a Bi-encoder
Say we want to learn a embedding space of labels and text. Consider the following four pairs. (In practice, you will have many more and much longer text entries.)
```
labels = ['fruit', 'meat', 'school', 'company']
texts = [
  'this is an apple', 
  'steak should be cooked medium rare', 
  'cmu is pittsburgh', 
  'apple sells laptop'
]
```

Initialize our encoder models,
```
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoder1 = AutoModel.from_pretrained("bert-base-uncased").cuda()
encoder2 = AutoModel.from_pretrained("bert-base-uncased").cuda()
```
Initialize the GradCache object,
```
from grad_cache import GradCache
from grad_cache.loss import SimpleContrastiveLoss

loss_fn = SimpleContrastiveLoss()
gc = GradCache(
  models=[encoder1, encoder2], 
  chunk_sizes=2, 
  loss_fn=loss_fn, 
  get_rep_fn=lambda v: v.pooler_output
)
```
Here we use the **get_rep_fn** argument to specify a function that takes generic Huggingface model output and return the actual representation tensor. 

Create model input,
```
xx = tokenizer(tt, return_tensors='pt', padding=True)
yy = tokenizer(tt2, return_tensors='pt', padding=True)
```
Run a cache step,
```
gc(xx, yy, reduction='mean')
```
Here we use `reduction='mean'` as a **loss_kwargs** to control loss behavior. With a defined `optimizer`, the full gradient update can be done as,
```
optimizer.zero_grad()
gc(xx, yy, reduction='mean')
optimizer.step()
``` 

### Use Tied Encoder?
This is naturally handled by the (magic of) dynamic graph. You pass shallow copies of the same encoder model to the GradCache init method.
```
tied_encoder = AutoModel.from_pretrained("bert-base-uncased").cuda()
gc = GradCache(
  models=[tied_encoder , tied_encoder], 
  chunk_sizes=2, 
  loss_fn=loss_fn, 
  get_rep_fn=lambda v: v.pooler_output
)
```
Under the hood, distinct hooks will be registered to make correct gradient computation.
### Distributed Training with Multiple GPUs?
We expect cross process communication of representations to be handled by the `loss_fn`. 
```
from grad_cache.loss import DistributedContrastiveLoss
loss_fn_dist = DistributedContrastiveLoss()
```
Properly wrap the the encoder models for gradient reduction,
```
encoder1_ddp = DistributedDataParallel(
	encoder1, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
encoder2_ddp = DistributedDataParallel(
	encoder2, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
```
You can initialize the cache use the distributed loss and the DDP models,
```
gc = GradCache(
  models=[encoder1_ddp, encoder2_ddp], 
  chunk_sizes=2, 
  loss_fn=loss_fn_dist, 
  get_rep_fn=lambda v: v.pooler_output
)
```
Run a cache step,
```
gc(xx, yy, no_sync_except_last=True, reduction='mean')
```
Set `no_sync_except_last=True` to avoid unnecessary gradient reduction.

## Functional Approach
### Decorators
If you are developing a new project, we recommend also checking out the decorators we have provided to create higher order functions for cache.
```
grad_cache.functional.cached(func: Callable[..., Tensor])
```
A decorator that takes a model call function into a cached compatible version.  

**func** - A function that calls the model and return representation tensor.

**Return** - A function that returns 1) representation leaf tensors for cache construction, 2) a closure function for  the 2nd forward and the cached backward. Call 2) with 1) as argument after calling backward on the loss Tensor.
```
grad_cache.functional.cat_input_tensor(func: Callable[..., Tensor])
```
A decorator that concatenates positional and keyword arguments of type List[Tensor] into a single Tensor  on the 0th dimension. This can come in handy dealing with results of representation tensors from multiple  cached forward.  

**func** - A loss function 

**Return** -  Decorated loss function for cached results.

```
grad_cache.functional.gather_input_tensor(func: Callable[..., Tensor], axis=0)
```
A decorator that all-gather positional and keyword arguments of type Tensor and concatenate them on axis. Intended to be used to create distributed contrastive learning loss.

**func** - A loss function 

**Return** -  Decorated loss function for distributed training.
### Usage
The functional decorators are particular useful if your data loader is emitting small batches, from which you can construct the big batch. Say you also want to do automatic mixed precision, we first define the model call function and loss function,
```
from grad_cache.functional import cached, cat_input_tensor

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

@cached
@autocast()
def  call_model(model, input):
	return model(**input).pooler_output

@cat_input_tensor
@autocast()
def  contrastive_loss(x, y):
	target = torch.arange(0, y.size(0), int(y.size(0) / x.size(0)), device=x.device)
	scores = torch.matmul(x, y.transpose(0, 1))
	return F.cross_entropy(scores, target=target)
```
Say you have a DataLoader `loader` emitting small batches of tuple `(xx, yy)`  of size (M * N) and  that you want to train by aggregating 16 small batches to get a batch of (16M * 16N),

```
cache_x = []
cache_y = []
closures_x = []
closures_y = []

for step, sub_batch in enumerate(loader):  
    xx, yy = sub_batch
    rx, cx = call_model(bert, xx)
    ry, cy = call_model(bert, yy)
    
    cache_x.append(rx)
    cache_y.append(ry)
    closuresx.append(cx)
    closuresy.append(cy)
    
    if (step + 1) % 16 == 0:
        loss = contrastive_loss(cache_x, cache_y)
        scaler.scale(loss).backward()
        
	for f, r in zip(closuresx, cache_x):
            f(r)
        for f, r in zip(closuresy, cache_y):
            f(r)

        cache_x = []
        cache_y = []
        closures_x = []
        closures_y = []
	
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
``` 
### Distributed Training
Running distributed multi-process training requires: 1) (all-)gather representations across devices and 2) (all-reduce) gradients across devices. Both steps will happen **outside** the cached decorated funtions. 

The latter is easy to achieve by wrapping encoders, e.g. a `bert`, in `DistributedDataParallel`.
```
bert = DistributedDataParallel(
	bert, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
```

The former requires extra distributed ops in the loss function, which should be done according the original loss definition. For example,
```
from torch import distributed as dist
from grad_cache.functional import cat_input_tensor, gather_input_tensor

@cat_input_tensor
@gather_input_tensor
@autocast()
def contrastive_loss(x, y):
    target = torch.arange(0, y.size(0), int(y.size(0) / x.size(0)), device=x.device)
    scores = torch.matmul(x, y.transpose(0, 1))
    # scale the loss as DistributedDataParallel will do mean reduce
    return F.cross_entropy(scores, target=target) * dist.get_world_size()  
```
## Code Structure
[grad_cache/grad_cache.py](src/grad_cache/grad_cache.py) - Define the GradCache class. The code is under 300 lines including comments. For development, we encourage you to read through it.

[grad_cache/functional.py](src/grad_cache/functional.py) - Define decorators to create higher order function for gradient caching from ordinary model call functions and loss functions.
