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
