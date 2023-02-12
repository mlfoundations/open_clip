## 2.13.0

* Add support for dataset mixtures with different sampling weights
* make transformers optional again 

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
