# CUB 2011 Quantitative Trait Evaluation

This document describes how to evaluate a pre-trained CLIP model on predicting traits for an individual image.

## Table of Contents

1. [Download Data](#download-data)
2. [Prepare Data](#prepare-data)
3. [Run Evaluation](#run-evaluation)

## Download Data

Download and extract the data from [CalTech Vision Lab](https://www.vision.caltech.edu/datasets/cub_200_2011/).

```
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xf CUB_200_2011.tgz
```

Move the `attributes.txt` file into the `attributes/` directory.

```
mv attributes.txt CUB_200_2011/attributes
```


## Prepare Data

Make a train/test split of the images, resized to whatever size you want (we used 224x224).

Modify the following variables in the script:

* `cub_root` should point to a directory with `images.txt`, `classes.txt` and the like. It's the parent CUB 2011 directory.
* `resize_size` should be an (x, y) tuple that is the output image size.
* `output_root` should be where you want to save the data. It will write a `train/` and a `test/` directory to `output_root`.

```
python scripts/cub2011_to_splits.py
```

If you have errors, try running it as a module:

```
python -m script.cub2011_to_splits
```

## Run Evaluation

Run the evaluation, pointing the script to a pretrained model.

Modify the following variables in the script `src/evaluation/cub_attrs.py`:

* `cub_img_root` should point to the `output_root` with `train/` and `test/` directories.
* `cub_label_root` should be the `cub_root` variable, pointing to a directory with `images.txt` and the like.
* `natural_attrs_path` shouldn't need to be changed. It should point to a file that has "natural" (language-like) templates for attributes.
* `template_option` can be one of `"bird"`, `"taxon"` or `"common"`. `"bird"` simply replaces the `{}` in the natural templates with the word "bird". `"taxon"` uses the taxonomic name, and `"common"` uses the common name. **Only `"bird"` is supported currently.**

Then run:

```
CUDA_VISIBLE_DEVICES=7 python -m src.evaluation.cub_attrs \
  --model ViT-B-16 \
  --pretrained openai \
  --batch-size 1024
```

This runs an OpenAI pretrained model on the evaluation.
