## Pretrained model details


### Pretrained model interface

We offer a simple model interface to instantiate both pre-trained and untrained models.

NOTE: Many existing checkpoints use the QuickGELU activation from the original OpenAI models. This activation is actually less efficient than native torch.nn.GELU in recent versions of PyTorch. The model defaults are now nn.GELU, so one should use model definitions with `-quickgelu` postfix for the OpenCLIP pretrained weights. All OpenAI pretrained weights will always default to QuickGELU. One can also use the non `-quickgelu` model definitions with pretrained weights using QuickGELU but there will be an accuracy drop, for fine-tune that will likely vanish for longer runs.

Future trained models will use nn.GELU.

```python
>>> import open_clip
>>> open_clip.list_pretrained()
[('RN50', 'openai'), 
('RN50', 'yfcc15m'), 
('RN50', 'cc12m'), 
('RN50-quickgelu', 'openai'), 
('RN50-quickgelu', 'yfcc15m'), 
('RN50-quickgelu', 'cc12m'), 
('RN101', 'openai'), 
('RN101', 'yfcc15m'), 
('RN101-quickgelu', 'openai'), 
('RN101-quickgelu', 'yfcc15m'), 
('RN50x4', 'openai'), 
('RN50x16', 'openai'), 
('RN50x64', 'openai'), 
('ViT-B-32', 'openai'), 
('ViT-B-32', 'laion400m_e31'), 
('ViT-B-32', 'laion400m_e32'), 
('ViT-B-32', 'laion2b_e16'), 
('ViT-B-32', 'laion2b_s34b_b79k'), 
('ViT-B-32', 'datacomp_m_s128m_b4k'), 
('ViT-B-32', 'commonpool_m_clip_s128m_b4k'), 
('ViT-B-32', 'commonpool_m_laion_s128m_b4k'), 
('ViT-B-32', 'commonpool_m_image_s128m_b4k'), 
('ViT-B-32', 'commonpool_m_text_s128m_b4k'), 
('ViT-B-32', 'commonpool_m_basic_s128m_b4k'), 
('ViT-B-32', 'commonpool_m_s128m_b4k'), 
('ViT-B-32', 'datacomp_s_s13m_b4k'), 
('ViT-B-32', 'commonpool_s_clip_s13m_b4k'), 
('ViT-B-32', 'commonpool_s_laion_s13m_b4k'), 
('ViT-B-32', 'commonpool_s_image_s13m_b4k'), 
('ViT-B-32', 'commonpool_s_text_s13m_b4k'), 
('ViT-B-32', 'commonpool_s_basic_s13m_b4k'), 
('ViT-B-32', 'commonpool_s_s13m_b4k'), 
('ViT-B-32-quickgelu', 'openai'), 
('ViT-B-32-quickgelu', 'laion400m_e31'), 
('ViT-B-32-quickgelu', 'laion400m_e32'), 
('ViT-B-16', 'openai'), 
('ViT-B-16', 'laion400m_e31'), 
('ViT-B-16', 'laion400m_e32'), 
('ViT-B-16', 'laion2b_s34b_b88k'), 
('ViT-B-16', 'datacomp_l_s1b_b8k'), 
('ViT-B-16', 'commonpool_l_clip_s1b_b8k'), 
('ViT-B-16', 'commonpool_l_laion_s1b_b8k'), 
('ViT-B-16', 'commonpool_l_image_s1b_b8k'), 
('ViT-B-16', 'commonpool_l_text_s1b_b8k'), 
('ViT-B-16', 'commonpool_l_basic_s1b_b8k'), 
('ViT-B-16', 'commonpool_l_s1b_b8k'), 
('ViT-B-16-plus-240', 'laion400m_e31'), 
('ViT-B-16-plus-240', 'laion400m_e32'), 
('ViT-L-14', 'openai'), 
('ViT-L-14', 'laion400m_e31'), 
('ViT-L-14', 'laion400m_e32'), 
('ViT-L-14', 'laion2b_s32b_b82k'), 
('ViT-L-14', 'datacomp_xl_s13b_b90k'), 
('ViT-L-14', 'commonpool_xl_clip_s13b_b90k'), 
('ViT-L-14', 'commonpool_xl_laion_s13b_b90k'), 
('ViT-L-14', 'commonpool_xl_s13b_b90k'), 
('ViT-L-14-336', 'openai'), 
('ViT-H-14', 'laion2b_s32b_b79k'), 
('ViT-g-14', 'laion2b_s12b_b42k'), 
('ViT-g-14', 'laion2b_s34b_b88k'), 
('ViT-bigG-14', 'laion2b_s39b_b160k'), 
('roberta-ViT-B-32', 'laion2b_s12b_b32k'), 
('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'), 
('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k'), 
('convnext_base', 'laion400m_s13b_b51k'), 
('convnext_base_w', 'laion2b_s13b_b82k'), 
('convnext_base_w', 'laion2b_s13b_b82k_augreg'), 
('convnext_base_w', 'laion_aesthetic_s13b_b82k'), 
('convnext_base_w_320', 'laion_aesthetic_s13b_b82k'), 
('convnext_base_w_320', 'laion_aesthetic_s13b_b82k_augreg'), 
('convnext_large_d', 'laion2b_s26b_b102k_augreg'), 
('convnext_large_d_320', 'laion2b_s29b_b131k_ft'), 
('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup'), 
('convnext_xxlarge', 'laion2b_s34b_b82k_augreg'), 
('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_rewind'), 
('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_soup'), 
('coca_ViT-B-32', 'laion2b_s13b_b90k'), 
('coca_ViT-B-32', 'mscoco_finetuned_laion2b_s13b_b90k'), 
('coca_ViT-L-14', 'laion2b_s13b_b90k'), 
('coca_ViT-L-14', 'mscoco_finetuned_laion2b_s13b_b90k'), 
('EVA01-g-14', 'laion400m_s11b_b41k'), 
('EVA01-g-14-plus', 'merged2b_s11b_b114k'), 
('EVA02-B-16', 'merged2b_s8b_b131k'), 
('EVA02-L-14', 'merged2b_s4b_b131k'), 
('EVA02-L-14-336', 'merged2b_s6b_b61k'), 
('EVA02-E-14', 'laion2b_s4b_b115k'), 
('EVA02-E-14-plus', 'laion2b_s9b_b144k')
]

>>> model, train_transform, eval_transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
```

## Pretrained models

Below are details for several of our pretrained models.

### LAION-400M - https://laion.ai/laion-400-open-dataset

We ran experiments in an attempt to reproduce OpenAI's ViT results with the comparably sized (and open) LAION-400M dataset. Trained
weights can be found in release [v0.2](https://github.com/mlfoundations/open_clip/releases/tag/v0.2-weights).

The LAION400M weights have been trained on the JUWELS supercomputer (see acknowledgements section below).

#### ViT-B/32 224x224

We replicate OpenAI's results on ViT-B/32, reaching a top-1 ImageNet-1k zero-shot accuracy of 62.96%.

<img src="https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/laion_clip_zeroshot.png" width="700">

__Zero-shot comparison (courtesy of Andreas Fürst)__
<img src="https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/laion_openai_compare_b32.jpg" width="700">

ViT-B/32 was trained with 128 A100 (40 GB) GPUs for ~36 hours, 4600 GPU-hours. The per-GPU batch size was 256 for a global batch size of 32768. 256 is much lower than it could have been (~320-384) due to being sized initially before moving to 'local' contrastive loss.

#### ViT-B/16 224x224

The B/16 LAION400M training reached a top-1 ImageNet-1k zero-shot validation score of 67.07.

<img src="https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/laion_clip_zeroshot_b16.png" width="700">

This was the first major train session using the updated webdataset 0.2.x code. A bug was found that prevented shards from being shuffled properly between nodes/workers each epoch. This was fixed part way through training (epoch 26) but likely had an impact.

ViT-B/16 was trained with 176 A100 (40 GB) GPUS for ~61 hours, 10700 GPU-hours. Batch size per GPU was 192 for a global batch size of 33792.

#### ViT-B/16+ 240x240

The B/16+ 240x240 LAION400M training reached a top-1 ImageNet-1k zero-shot validation score of 69.21.

This model is the same depth as the B/16, but increases the
  * vision width from 768 -> 896
  * text width from 512 -> 640
  * the resolution 224x224 -> 240x240 (196 -> 225 tokens)

<img src="https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/laion_clip_zeroshot_b16_plus_240.png" width="700">

Unlike the B/16 run above, this model was a clean run with no dataset shuffling issues.

ViT-B/16+ was trained with 224 A100 (40 GB) GPUS for ~61 hours, 13620 GPU-hours. Batch size per GPU was 160 for a global batch size of 35840.

#### ViT-L/14 224x224

The L/14 LAION-400M training reached a top-1 ImageNet-1k zero-shot validation score of 72.77.

<img src="https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/laion_clip_zeroshot_l14.png" width="700">

ViT-L/14 was trained with 400 A100 (40 GB) GPUS for ~127 hours, 50800 GPU-hours. Batch size per GPU was 96 for a global batch size of 38400. Grad checkpointing was enabled.

### LAION-2B (en) - https://laion.ai/laion-5b-a-new-era-of-open-large-scale-multi-modal-datasets/

A ~2B sample subset of LAION-5B with english captions (https://huggingface.co/datasets/laion/laion2B-en)

#### ViT-B/32 224x224
A ViT-B/32 trained on LAION-2B, reaching a top-1 ImageNet-1k zero-shot accuracy of 65.62%.

<img src="https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/laion2b_clip_zeroshot_b32.png" width="700">

ViT-B/32 was trained with 112 A100 (40 GB) GPUs. The per-GPU batch size was 416 for a global batch size of 46592. Compute generously provided by [stability.ai](https://stability.ai/).

A second iteration of B/32 was trained on stability.ai cluster with a larger global batch size and learning rate, hitting 66.6% top-1. See https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K

#### ViT-L/14 224x224

A ViT-L/14 with a 75.3% top-1 ImageNet-1k zero-shot was trained on JUWELS Booster. See model details here https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K

These weights use a different dataset mean and std than others. Instead of using the OpenAI mean & std, inception style normalization `[-1, 1]` is used via a mean and std of `[0.5, 0.5, 0.5]`. This is handled automatically if using `open_clip.create_model_and_transforms` from pretrained weights.

#### ViT-H/14 224x224

A ViT-H/14 with a 78.0% top-1 ImageNet-1k zero-shot was trained on JUWELS Booster. See model details here https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K

#### ViT-g/14 224x224

A ViT-g/14 with a 76.6% top-1 ImageNet-1k zero-shot was trained on JUWELS Booster. See model details here https://huggingface.co/laion/CLIP-ViT-g-14-laion2B-s12B-b42K

This model was trained with a shorted schedule than other LAION-2B models with 12B samples seen instead of 32+B. It matches LAION-400M training in samples seen. Many zero-shot results are lower as a result, but despite this it performs very well in some OOD zero-shot and retrieval tasks.


#### ViT-B/32 roberta base

A ViT-B/32 with roberta base encoder with a 61.7% top-1 ImageNet-1k zero-shot was trained on stability. See model details here https://huggingface.co/laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k
This is the first openclip model using a HF text tower. It has better performance on a range of tasks compared to the standard text encoder, see [metrics](https://huggingface.co/laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k/blob/main/unknown.png)

#### ViT-B/32 xlm roberta base

A ViT-B/32 with xlm roberta base encoder with a 62.33% top-1 ImageNet-1k zero-shot was trained on stability. See model details here https://huggingface.co/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k
This is the first openclip model trained on the full laion5B dataset; hence the first multilingual clip trained with openclip. It has better performance on a range of tasks compared to the standard text encoder, see [metrics](https://huggingface.co/laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k/blob/main/metrics.png)
A preliminary multilingual evaluation was run: 43% on imagenet1k italian (vs 21% for english B/32), 37% for imagenet1k japanese (vs 1% for english B/32 and 50% for B/16 clip japanese). It shows the multilingual property is indeed there as expected. Larger models will get even better performance.

#### ViT-H/14 xlm roberta large

A ViT-H/14 with xlm roberta large encoder with a 77.0% (vs 78% for the english equivalent) top-1 ImageNet-1k zero-shot was trained on stability. See model details here https://huggingface.co/laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k

This model was trained following the [LiT](https://arxiv.org/abs/2111.07991) methodology: the image tower was frozen (initialized from english openclip ViT-H/14), the text tower was initialized from [xlm roberta large](https://huggingface.co/xlm-roberta-large) and unfrozen. This reduced training cost by a 3x factor.

See full english [metrics](https://huggingface.co/laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k/resolve/main/results_xlm_roberta_large.png)

On zero shot classification on imagenet with translated prompts this model reaches:

* 56% in italian (vs 21% for https://github.com/clip-italian/clip-italian)
* 53% in japanese (vs 54.6% for https://github.com/rinnakk/japanese-clip)
* 55.7% in chinese (to be compared with https://github.com/OFA-Sys/Chinese-CLIP)


#### YFCC-15M

Below are checkpoints of models trained on YFCC-15M, along with their zero-shot top-1 accuracies on ImageNet and ImageNetV2. These models were trained using 8 GPUs and the same hyperparameters described in the "Sample running code" section, with the exception of `lr=5e-4` and `epochs=32`.

* [ResNet-50](https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt) (32.7% / 27.9%)
* [ResNet-101](https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pt) (34.8% / 30.0%)

#### CC12M - https://github.com/google-research-datasets/conceptual-12m

* [ResNet-50](https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt) (36.45%)


### CommonPool and DataComp models

As part of [DataComp](https://github.com/mlfoundations/datacomp), we trained models on CommonPool using various data filtering strategies.

The best performing models are specified below for the xlarge scale, see our paper [DataComp: In seearch of the next generation of multimodal datasets](https://arxiv.org/abs/2304.14108) for more details.

Additional models and more information can be found at [/docs/datacomp_models.md](/docs/datacomp_models.md).


* `datacomp_xl_s13b_b90k`: A ViT-L/14 trained on DataComp-1B for 12.8B steps and batch size 90k. Achieves 79.2% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K. 

* `commonpool_xl_clip_s13b_b90k`: A ViT-L/14 trained on CommonPool-XL filtered using CLIP scores, for 12.8B steps and batch size 90k. Achieves 76.4% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K.

* `commonpool_xl_laion_s13b_b90k`: A ViT-L/14 trained on CommonPool-XL filtered using the LAION-2B filtering scheme, for 12.8B steps and batch size 90k. Achieves 75.5% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-L-14-CommonPool.XL.laion-s13B-b90K.

* `commonpool_xl_s13b_b90k`: A ViT-L/14 trained on CommonPool-XL without any filtering, for 12.8B steps and batch size 90k. Achieves 72.3% zero-shot accuracy on ImageNet. Available at https://huggingface.co/laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K.


