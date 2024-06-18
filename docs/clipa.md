## CLIPA

In this work, we present a surprising finding that there exists an _inverse_ scaling law for CLIP training, 
whereby the larger the image/text encoders used, the shorter the sequence length of image/text tokens that can be applied in training. 
Moreover, we showcase that the strategy for reducing image/text token length plays a crucial role in determining the quality of this scaling law.

![](/docs/inverse_scaling_law.png)

As a result of this finding, we are able to successfully train CLIP even by using academic resources. 
For example, on an A100 eight-GPU server, our CLIP models achieve zero-shot top-1 ImageNet accuracies of **63.2%** in about **2 days**, 
**67.8%** in about **3 days**, and **69.3%** in about **4 days**.

Moreover, We find that CLIPA at scale leads to state-of-the-art performance. For example, our CLIPA-v2 H/14 achieves a zero-shot top-1 ImageNet accuracy of **81.8%**,
with a budget less than **$15000**.

![](/docs/clipa_acc_compute.png)

For more details, please see our paper [An Inverse Scaling Law for CLIP Training](https://arxiv.org/abs/2305.07017) and 
[CLIPA-v2: Scaling CLIP Training with 81.1% Zero-shot ImageNet Accuracy within a $10,000 Budget; An Extra $4,000 Unlocks 81.8% Accuracy](https://arxiv.org/abs/2306.15658).


Eight token length reduction strategies are investigated in this work, detailed as follows.


## Image token length reduction

![](/docs/clipa_reduce_image_token.png)

* `resize`: use `--force-image-size` to specify the image size you want to adopt. We find this strategy generally works the best as it retains full image information.

* `random mask`: Randomly mask out image patches. use `--force-patch-dropout` to specify the mask ratio you want to adopt. 

* `grid mask`: Preserve one patch in each 2 × 2 grid window. We do not provide implementation for grid masking, as it is only experimental and we generally find resizing works better.

* `block mask`: Keep a single block and remove other patches. We do not provide implementation for block masking, as it is only experimental and we generally find resizing works better.


## Text token length reduction

* `syntax mask`: Assign different masking priorities to parts of speech. Specify `"text_mask": syntax` in `"tokenizer_kwargs"` in `"text_cfg"` of model config `json` file to use. 
Specifically, we prioritize retaining nouns, followed by adjectives, and then other words. 
We find this strategy generally works the best as it retains critical information for contrastive learning.

* `truncate`: Truncation selects the first N text tokens and discards the rest. This is the default setting of `open_clip`. 

* `random mask`: Randomly drops a portion of the text tokens. Specify `"text_mask": random` in `"tokenizer_kwargs"` in `"text_cfg"` of model config `json` file to use. 

* `block mask`: Randomly preserves consecutive text sequences. Specify `"text_mask": block` in `"tokenizer_kwargs"` in `"text_cfg"` of model config `json` file to use. 


## Installation

The installation is really the same as `open_clip`, except for the usage of Natural Language Toolkit (NLTK) in `syntax mask` of text token length reduction.
Please follow the [official doc](https://www.nltk.org/) to install NLTK.

Note that the the usage of NLTK brings two constraints:
* Because certain functions like `nltk.pos_tag` from NLTK only support English and Russian for now, the `syntax mask` only works for English. 
we have not tested it on Russian or any other language. Theoretically, it should work the same, given a proper language processing toolkit for other languages.
If you still want to apply `syntax mask` on other languages, try finding the right toolkit. Otherwise, use other text token length reduction strategies
* some modules of NLTK like `punkt` or `averaged_perceptron_tagger` need to be downloaded first before using NLTK.
We have included the downloading code in `tokenizer.py`, but this might cause trouble in certain cases.
You may want to manually download those modules first, by `nltk.download('punkt')` and `nltk.download('averaged_perceptron_tagger')`,
and then setup the environmental variable before running the script `export NLTK_DATA=cache`. 
Note that this is a one-time effort. Remember to comment out those `nltk.download` lines in `tokenizer.py` afterwards.

## Training
We provide example scripts to reproduce our CLIPA results on an A100 eight-GPU machine under path `docs/script_examples/clipa`.

For instance, to reproduce the CLIPA-L16(I37,T8) results, first run the pre-training script
```
bash docs/script_examples/clipa/vit_l16/i37_t8_pretrain.sh
```
and fine-tune the pre-trained checkpoint with
```
bash docs/script_examples/clipa/vit_l16/i37_t8_finetune.sh
```
- Remember to change the path to dataset to your own path.
- This is a two-stage training pipeline. Remember to change the path to pre-trained checkpoint to your own when fine-tuning.
- The training time is ~3 days for pre-training and ~1 day for fine-tuning on an A100 eight-GPU machine.

## Model Weights
Below are CLIPA trained weights on LAION-400M with an A100 eight-GPU machine. 
All models are pre-trained for 6 epochs with reduced input token lengths and subsequently fine-tuned for 0.36 epoch with full input token lengths.


|                     |                                      Pre-trained Weights                                       | zero-shot IN-1K |
|---------------------|:----------------------------------------------------------------------------------------------:|:---------------:|
| CLIPA-B/16(I50,T16) | [download](https://drive.google.com/file/d/1MDpz8gV2Vjaazk16rBhLxU8811U7_cGL/view?usp=sharing) |      59.7       |
| CLIPA-L/16(I17,T16) | [download](https://drive.google.com/file/d/1Tr2GYiKAaMH6EGIn5l7eX_1K20eaA3WA/view?usp=sharing) |      60.3       |
| CLIPA_L/16(I37,T8)  | [download](https://drive.google.com/file/d/1EM1ChRNARpLckkJjf6m7njCY3xyvpGBu/view?usp=sharing) |      57.9       |

|                     |                                       Fine-tuned Weights                                       | zero-shot IN-1K |
|---------------------|:----------------------------------------------------------------------------------------------:|:-----:|
| CLIPA-B/16(I50,T16) | [download](https://drive.google.com/file/d/1fURK0K_a3-83jVEI4PVEbnEJb_V6UbGv/view?usp=sharing) | 63.2  |
| CLIPA-L/16(I17,T16) | [download](https://drive.google.com/file/d/18qqZGOTGOgb3I3JWONuat6qObsgLq7sR/view?usp=sharing) | 67.8  |
| CLIPA_L/16(I37,T8)  | [download](https://drive.google.com/file/d/1lV7pLORUK04T9QKKx9TpYtMws-AZrib0/view?usp=sharing) | 69.3  |


## CLIPA-v2
We also provide example scripts to reproduce our CLIPA-v2 H/14 results under path `docs/script_examples/clipav2`.
Note that the original results are obtained with [our JAX implementation](https://github.com/UCSC-VLAA/CLIPA/tree/master/clipa_jax).
These scripts are written after manually scanning the JAX config files.
As it is infeasible for us to retrain those models again with pytorch, its correctness cannot be verified with 100% confidence. Use them at your own discretion.
