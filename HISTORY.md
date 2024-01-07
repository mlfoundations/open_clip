## 2.24.0

* Fix missing space in error message
* use model flag for normalizing embeddings
* init logit_bias for non siglip pretrained models
* Fix logit_bias load_checkpoint addition 
* Make CoCa model match CLIP models for logit scale/bias init
* Fix missing return of "logit_bias" in CoCa.forward
* Add NLLB-CLIP with SigLIP models
* Add get_logits method and NLLB tokenizer
* Remove the empty file src/open_clip/generation_utils.py
* Update params.py: "BatchNorm" -> "LayerNorm" in the description string for "--lock-text-freeze-layer-norm"

## 2.23.0

* Add CLIPA-v2 models
* Add SigLIP models
* Add MetaCLIP models
* Add NLLB-CLIP models
* CLIPA train code
* Minor changes/fixes
    * Remove protobuf version limit
    * Stop checking model name when loading CoCa models
    * Log native wandb step
    * Use bool instead of long masks

## 2.21.0

* Add SigLIP loss + training support
* Add more DataComp models (B/16, B/32 and B/32@256)
* Update default num workers
* Update CoCa generation for `transformers>=4.31`
* PyTorch 2.0 `state_dict()` compatibility fix for compiled models
* Fix padding in `ResizeMaxSize`
* Convert JIT model on state dict load for `pretrained='filename…'`
* Other minor changes and fixes (typos, README, dependencies, CI)

## 2.20.0

* Add EVA models
* Support serial worker training
* Fix Python 3.7 compatibility 

## 2.19.0

* Add DataComp models

## 2.18.0

* Enable int8 inference without `.weight` attribute

## 2.17.2

* Update push_to_hf_hub

## 2.17.0

* Add int8 support
* Update notebook demo
* Refactor zero-shot classification code

## 2.16.2

* Fixes for context_length and vocab_size attributes 

## 2.16.1

* Fixes for context_length and vocab_size attributes 
* Fix --train-num-samples logic
* Add HF BERT configs for PubMed CLIP model

## 2.16.0

* Add improved g-14 weights
* Update protobuf version

## 2.15.0

* Add convnext_xxlarge weights
* Fixed import in readme
* Add samples per second per gpu logging
* Fix slurm example

## 2.14.0

* Move dataset mixtures logic to shard level
* Fix CoCa accum-grad training
* Safer transformers import guard
* get_labels refactoring

## 2.13.0

* Add support for dataset mixtures with different sampling weights
* Make transformers optional again 

## 2.12.0

* Updated convnext configs for consistency
* Added input_patchnorm option
* Clean and improve CoCa generation
* Support model distillation
* Add ConvNeXt-Large 320x320 fine-tune weights

## 2.11.1

* Make transformers optional
* Add MSCOCO CoCa finetunes to pretrained models

## 2.11.0

* coca support and weights
* ConvNeXt-Large weights

## 2.10.1

* `hf-hub:org/model_id` support for loading models w/ config and weights in Hugging Face Hub

## 2.10.0

* Added a ViT-bigG-14 model.
* Added an up-to-date example slurm script for large training jobs.
* Added a option to sync logs and checkpoints to S3 during training.
* New options for LR schedulers, constant and constant with cooldown
* Fix wandb autoresuming when resume is not set
* ConvNeXt `base` & `base_w` pretrained models added
* `timm-` model prefix removed from configs
* `timm` augmentation + regularization (dropout / drop-path) supported

## 2.9.3

* Fix wandb collapsing multiple parallel runs into a single one

## 2.9.2

* Fix braceexpand memory explosion for complex webdataset urls

## 2.9.1

* Fix release

## 2.9.0

* Add training feature to auto-resume from the latest checkpoint on restart via `--resume latest`
* Allow webp in webdataset
* Fix logging for number of samples when using gradient accumulation
* Add model configs for convnext xxlarge

## 2.8.2

* wrapped patchdropout in a torch.nn.Module

## 2.8.1

* relax protobuf dependency
* override the default patch dropout value in 'vision_cfg'

## 2.8.0

* better support for HF models
* add support for gradient accumulation
* CI fixes
* add support for patch dropout
* add convnext configs


## 2.7.0

* add multilingual H/14 xlm roberta large

## 2.6.1

* fix setup.py _read_reqs

## 2.6.0

* Make openclip training usable from pypi.
* Add xlm roberta large vit h 14 config.

## 2.5.0

* pretrained B/32 xlm roberta base: first multilingual clip trained on laion5B
* pretrained B/32 roberta base: first clip trained using an HF text encoder

## 2.4.1

* Add missing hf_tokenizer_name in CLIPTextCfg.

## 2.4.0

* Fix #211, missing RN50x64 config. Fix type of dropout param for ResNet models
* Bring back LayerNorm impl that casts to input for non bf16/fp16 
* zero_shot.py: set correct tokenizer based on args
* training/params.py: remove hf params and get them from model config

## 2.3.1

* Implement grad checkpointing for hf model.
* custom_text: True if hf_model_name is set
* Disable hf tokenizer parallelism 

## 2.3.0

* Generalizable Text Transformer with HuggingFace Models (@iejMac)

## 2.2.0

* Support for custom text tower
* Add checksum verification for pretrained model weights 

## 2.1.0

* lot including sota models, bfloat16 option, better loading, better metrics

## 1.2.0

* ViT-B/32 trained on Laion2B-en
* add missing openai RN50x64 model

## 1.1.1

* ViT-B/16+
* Add grad checkpointing support
* more robust data loader
