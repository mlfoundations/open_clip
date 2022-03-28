# OpenCLIP

Welcome to an open source implementation of OpenAI's [CLIP](https://arxiv.org/abs/2103.00020) (Contrastive Language-Image Pre-training). 

The goal of this repository is to enable training models with contrastive image-text supervision, and to investigate their properties such as robustness to distribution shift. Our starting point is an implementation of CLIP that matches the accuracy of the original CLIP models when trained on the same dataset.
Specifically, a ResNet-50 model trained with our codebase on OpenAI's [15 million image subset of YFCC](https://github.com/openai/CLIP/blob/main/data/yfcc100m.md) achieves **32.7%** top-1 accuracy on ImageNet. OpenAI's CLIP model reaches **31.3%** when trained on the same subset of YFCC. For ease of experimentation, we also provide code for training on the 3 million images in the [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/download) dataset, where a ResNet-50x4 trained with our codebase reaches 22.2% top-1 ImageNet accuracy.

As we describe in more detail [below](#why-are-low-accuracy-clip-models-interesting), CLIP models in a medium accuracy regime already allow us to draw conclusions about the robustness of larger CLIP models since the models follow [reliable scaling laws](https://arxiv.org/abs/2107.04649).

This codebase is work in progress, and we invite all to contribute in making it more acessible and useful. In the future, we plan to add support for TPU training and release larger models. We hope this codebase facilitates and promotes further research in contrastive image-text learning. Please submit an issue or send an email if you have any other requests or suggestions.

Note that `src/clip` is a copy of OpenAI's official [repository](https://github.com/openai/CLIP) with minimal changes.

#### Fine-tuning

This repository is focused on training CLIP models. To fine-tune a *trained* zero-shot model on a downstream classification task such as ImageNet, please see [our other repository: WiSE-FT](https://github.com/mlfoundations/wise-ft). The [WiSE-FT repository](https://github.com/mlfoundations/wise-ft) contains code for our paper on [Robust Fine-tuning of Zero-shot Models](https://arxiv.org/abs/2109.01903), in which we introduce a technique for fine-tuning zero-shot models while preserving robustness under distribution shift.

## Data


### Conceptual Captions

OpenCLIP reads a CSV file with two columns: a path to an image, and a text caption. The names of the columns are passed as an argument to `main.py`.

The script `src/data/gather_cc.py` will collect the Conceptual Captions images. First, download the [Conceptual Captions URLs](https://ai.google.com/research/ConceptualCaptions/download) and then run the script from our repository:

```bash
python3 src/data/gather_cc.py path/to/Train_GCC-training.tsv path/to/Validation_GCC-1.1.0-Validation.tsv
```

Our training set contains 2.89M images, and our validation set contains 13K images.


### YFCC and other datasets

In addition to specifying the training data via CSV files as mentioned above, our codebase also supports [webdataset](https://github.com/webdataset/webdataset), which is recommended for larger scale datasets. The expected format is a series of `.tar` files. Each of these `.tar` files should contain two files for each training example, one for the image and one for the corresponding text. Both files should have the same name but different extensions. For instance, `shard_001.tar` could contain files such as `abc.jpg` and `abc.txt`. You can learn more about `webdataset` at [https://github.com/webdataset/webdataset](https://github.com/webdataset/webdataset). We use `.tar` files with 1,000 data points each, which we create using [tarp](https://github.com/webdataset/tarp).

You can download the YFCC dataset from [Multimedia Commons](http://mmcommons.org/).
Similar to OpenAI, we used a subset of YFCC to reach the aforementioned accuracy numbers.
The indices of images in this subset are in [OpenAI's CLIP repository](https://github.com/openai/CLIP/blob/main/data/yfcc100m.md).


## Training CLIP

### Install dependencies

```bash
conda env create
conda activate open_clip
```

### Add directory to pythonpath:

```bash
cd open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"
```


### Sample running code:

```bash
nohup python -u src/training/main.py \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/path/to/train_data.csv"  \
    --val-data="/path/to/validation_data.csv"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=/path/to/imagenet/root/val/ \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=30 \
    --workers=8 \
    --model RN50
```

Note: `imagenet-val` is the path to the *validation* set of ImageNet for zero-shot evaluation, not the training set!
You can remove this argument if you do not want to perform zero-shot evaluation on ImageNet throughout training. Note that the `val` folder should contain subfolders. If it doest not, please use [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

When run on a machine with 8 GPUs the command should produce the following training curve for Conceptual Captions:

![CLIP zero shot training curve](/docs/clip_zeroshot.png)

More detailed curves for Conceptual Captions are given at [/docs/clip_conceptual_captions.md](/docs/clip_conceptual_captions.md).

When training a RN50 on YFCC the same hyperparameters as above are used, with the exception of `lr=5e-4` and `epochs=32`.

Note that to use another model, like `ViT-B/32` or `RN50x4` or `RN50x16` or `ViT-B/16`, specify with `--model RN50x4`.

### Launch tensorboard:
```bash
tensorboard --logdir=logs/tensorboard/ --port=7777
```

### Sample resuming from a checkpoint:

```bash
python src/training/main.py \
    --train-data="/path/to/train_data.csv" \
    --val-data="/path/to/validation_data.csv"  \
    --resume /path/to/checkpoints/epoch_K.pt
```

### Sample evaluation only:

```bash
python src/training/main.py \
    --val-data="/path/to/validation_data.csv"  \
    --resume /path/to/checkpoints/epoch_K.pt
```

## Trained models

Below are checkpoints of models trained on YFCC-15M, along with their zero-shot top-1 accuracies on ImageNet and ImageNetV2. These models were trained using 8 GPUs and the same hyperparameters described in the "Sample running code" section, with the exception of `lr=5e-4` and `epochs=32`.

* [ResNet-50](https://drive.google.com/file/d/1bbNMrWAq3NxCAteHmbrYgBKpGAA9j9pn/view?usp=sharing) (32.7% / 27.9%)
* [ResNet-101](https://drive.google.com/file/d/1vOorR3pKkuuA_Gv7OgqMtaETNjTmy_3m/view?usp=sharing) (34.8% / 30.0%)

## Interface

Instead of providing an interface for `open_clip`, we re-use [OpenAI's](https://github.com/openai/CLIP).

If you have an open_clip checkpoint saved at the path `open_clip_checkpoint_pth` you can run the following code to store an OpenAI CLIP compatible checkpoint at `clip_checkpoint_pth`
```python
import torch
open_clip_cp = torch.load(open_clip_checkpoint_pth)
open_clip_sd = {k[7:]:v for k, v in open_clip_cp['state_dict'].items()}
torch.save(open_clip_sd, clip_checkpoint_pth)
```
now you can use OpenAI's interface
```python
import clip # import OpenAI's CLIP
model = clip.load(clip_checkpoint_pth)
```

## Scaling trends

The plot below shows how zero-shot performance of CLIP models varies as we scale the number of samples used for training. Zero-shot performance increases steadily for both ImageNet and [ImageNetV2](https://arxiv.org/abs/1902.10811), and is far from saturated at ~15M samples.

<img src="docs/scaling.png" width="700">

## Why are low-accuracy CLIP models interesting?

**TL;DR:** CLIP models have high effective robustness, even at small scales.

CLIP models are particularly intriguing because they are more robust to natural distribution shifts (see Section 3.3 in the [CLIP paper](https://arxiv.org/abs/2103.00020)).
This phenomena is illustrated by the figure below, with ImageNet accuracy on the x-axis
and [ImageNetV2](https://arxiv.org/abs/1902.10811) (a reproduction of the ImageNet validation set with distribution shift) accuracy on the y-axis.
Standard training denotes training on the ImageNet train set and the CLIP zero-shot models
are shown as stars.

![CLIP scatter plot](/docs/effective_robustness.png)

As observed by [Taori et al., 2020](https://arxiv.org/abs/2007.00644) and [Miller et al., 2021](https://arxiv.org/abs/2107.04649), the in-distribution
and out-of-distribution accuracies of models trained on ImageNet follow a predictable linear trend (the red line in the above plot). *Effective robustness*
quantifies robustness as accuracy beyond this baseline, i.e., how far a model lies above the red line. Ideally a model would not suffer from distribution shift and fall on the y = x line ([trained human labelers are within a percentage point of the y = x line](http://proceedings.mlr.press/v119/shankar20c.html)).

Even though the CLIP models trained with
this codebase achieve much lower accuracy than those trained by OpenAI, our models still lie on the same
trend of improved effective robustness (the purple line). Therefore, we can study what makes
CLIP robust without requiring industrial-scale compute.

For more more information on effective robustness, please see:

- [Recht et al., 2019](https://arxiv.org/abs/1902.10811).
- [Taori et al., 2020](https://arxiv.org/abs/2007.00644).
- [Miller et al., 2021](https://arxiv.org/abs/2107.04649).

## The Team

Current development of this repository is led by [Ross Wightman](https://rwightman.com/), [Cade Gordon](http://cadegordon.io/), and [Vaishaal Shankar](http://vaishaal.com/).

The original version of this repository is from a group of researchers at UW, Google, Stanford, Amazon, Columbia, and Berkeley.

[Gabriel Ilharco*](http://gabrielilharco.com/), [Mitchell Wortsman*](https://mitchellnw.github.io/), [Nicholas Carlini](https://nicholas.carlini.com/), [Rohan Taori](https://www.rohantaori.com/), [Achal Dave](http://www.achaldave.com/), [Vaishaal Shankar](http://vaishaal.com/), [John Miller](https://people.eecs.berkeley.edu/~miller_john/), [Hongseok Namkoong](https://hsnamkoong.github.io/), [Hannaneh Hajishirzi](https://homes.cs.washington.edu/~hannaneh/), [Ali Farhadi](https://homes.cs.washington.edu/~ali/), [Ludwig Schmidt](https://people.csail.mit.edu/ludwigs/)

Special thanks to [Jong Wook Kim](https://jongwook.kim/) and [Alec Radford](https://github.com/Newmu) for help with reproducing CLIP!

## Citing

If you found this repository useful, please consider citing:
```bibtex
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}
```

```bibtex
@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
```

[![DOI](https://zenodo.org/badge/390536799.svg)](https://zenodo.org/badge/latestdoi/390536799)
