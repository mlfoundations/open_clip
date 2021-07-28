# OpenCLIP

Welcome to an open source implementation of OpenAI's [CLIP](https://arxiv.org/abs/2103.00020) (Contrastive Language-Image Pre-training). 

The goal of this repository is to match the accuracy of the original CLIP models when trained on the same dataset. For example, our implementation reaches 22.2% top-1 ImageNet when training a ResNet 50x4 on the 3 million images in the Conceptual Captions dataset, and 32.7% top-1 ImageNet accuracy when training a RN50 on OpenAI's [15 million image subset of YFCC](https://github.com/openai/CLIP/blob/main/data/yfcc100m.md). OpenAI's CLIP model reaches 31.3% on the same subset of YFCC.

Note that `src/clip` is a copy of OpenAI's official [repo](https://github.com/openai/CLIP) with minimal changes.

## Data


### Conceptual Captions

OpenCLIP reads a CSV file with two columns: a path to an image, and a text caption. The names of the columns are passed as an argument to `main.py`.

The script `src/data/gather_cc.py` will collect the Conceptual Captions images. First, download the [Conceptual Captions URLs](https://ai.google.com/research/ConceptualCaptions/download) and then run the following script from our repository:

```
python3 src/data/gather_cc.py path/to/Train_GCC-training.tsv path/to/Validation_GCC-1.1.0-Validation.tsv
```

Our training set contains 2.89M images, and our validation set contains 13K images.


### YFCC and other datasets

In addition to specifying the training data via CSV files as mentioned above, our codebase also supports [webdataset](https://github.com/webdataset/webdataset), which is recommended for larger scale datasets. The expected format is a series of `.tar` files. Each of these `.tar` files should contain two files for each training example, one for the image and one for the corresponding text. Both files should have the same name but different extensions. For instance, `shard_001.tar` could contain files such as `abc.jpg` and `abc.txt`. You can learn more about `webdataset` at [https://github.com/webdataset/webdataset](https://github.com/webdataset/webdataset). We use `.tar` files with 1000 data points each, which we create using [tarp](https://github.com/webdataset/tarp).

You can download the YFCC dataset from [Multimedia Commons](http://mmcommons.org/).
Similar to OpenAI, we used a subset of YFCC to reach the aforementioned accuracy numbers.
The indices of images in this subset are in [OpenAI's CLIP repository](https://github.com/openai/CLIP/blob/main/data/yfcc100m.md).


## Training CLIP

### Install dependencies

```
conda env create -f environment.yml
source activate open_clip
```

### Add directory to pythonpath:

```
cd open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"
```


### Sample running code:

```
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
    --workers=8
```

Note: `imagenet-val` is the path to the *val* set of ImageNet for zeroshot evaluation, not the train set!
You can remove this argument if you do not want to perform zeroshot on imagenet throughout training. Note that the `val` folder should contain subfolders, if it doesn't please use [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

This command should produce the following training curve:

![CLIP zero shot training curve](/docs/clip_zeroshot.png)

More detailed curves are given in [/docs/clip_conceptual_captions.md](/docs/clip_conceptual_captions.md)

### Launch tensorboard:
```
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

## Scaling trends

The plot below shows how zero-shot performance of CLIP models varies as we scale the number of samples used for training. Zero-shot performance increases steadily for both ImageNet and [ImageNetV2](https://arxiv.org/abs/1902.10811), and is far from saturated at ~15M samples.

<img src="docs/scaling.png" width="700">

## Why are low-accuracy CLIP models interesting?

**TL;DR:** CLIP models have high effective robustness, even at small scales.

CLIP models are particularly intriguing because they are more robust to natural distribution shifts.
This phenomena is illustrated by the figure below, with ImageNet accuracy on the x-axis
and [ImageNetV2](https://arxiv.org/abs/1902.10811) (a reproduction of the ImageNet) accuracy on the y-axis.
Standard training denotes training on the ImageNet train set while the CLIP zero-shot models
are shown with stars.

![CLIP scatter plot](/docs/effective_robustness.png)

As observed by [Taori et al., 2020](https://arxiv.org/abs/2007.00644), in-distribution
and out-of-distribution accuracy follow a predictable linear trend. Effective robustness
measures movement above this red line. Even though the models trained with
this codebase are much lower accuracy than those trained by OpenAI, they lie on the same
trend of improved effective robustness (purple line). Therefore, we can study what makes
CLIP robust without needing industry compute.

For more more information on effective robustness please see:

- [Recht et al., 2019](https://arxiv.org/abs/1902.10811).
- [Taori et al., 2020](https://arxiv.org/abs/2007.00644).
- [Miller et al., 2021](https://arxiv.org/abs/2107.04649).

## The Team


[Gabriel Ilharco*](http://gabrielilharco.com/), [Mitchell Wortsman*](https://mitchellnw.github.io/), [Nicholas Carlini](https://nicholas.carlini.com/), [Rohan Taori](https://www.rohantaori.com/), [Achal Dave](http://www.achaldave.com/), [Vaishaal Shankar](http://vaishaal.com/), [
Hongseok Namkoong](https://hsnamkoong.github.io/), [John Miller](https://people.eecs.berkeley.edu/~miller_john/), [Hannaneh Hajishirzi](https://homes.cs.washington.edu/~hannaneh/), [Ali Farhadi](https://homes.cs.washington.edu/~ali/), [Ludwig Schmidt](https://people.csail.mit.edu/ludwigs/)

Special thanks to Jong Wook Kim and Alec Radford!
