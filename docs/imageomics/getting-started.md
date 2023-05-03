## Clone

```sh
git clone https://github.com/Imageomics/open_clip.git
cd open_clip
```

## Make a Virtual Environment

(or conda environment)

I use Python 3.10.9, pyenv and the fish shell:

```sh
pyenv local 3.10.9
python -m venv .venv
source .venv/bin/activate.fish  # fish shell
```

## Install Requirements

```
pip install --upgrade pip  # upgrade pip
pip install --editable .  # install this package
```

Install wandb:

```sh
pip install wandb
wandb login
```

## Download Data

### iNaturalist 2021

Download 200+GB of images:

```sh
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz
```

Then you'll have to untar them into folders of raw images.
On strawberry0, the raw images are in `/local/scratch/cv_datasets/inat21/raw`.
On OSC, the raw images are in `/fs/scratch/PAS2136/hierarchical-vision/datasets/inat21/raw`.

### Common Names

**NOTE: the common name mapping is already committed to git. So you only need to run this step if you want to regenerate it.**

Download the taxa and the common names:

```sh
wget https://www.inaturalist.org/taxa/inaturalist-taxonomy.dwca.zip
unzip inaturalist-taxonomy.dwca.zip
```

Move it to the `data/inat/` directory.
Then you can create the scientific name to common name mapping.
You'll have to edit the `archive_path` and `inat_root` variables in `scripts/inat_common_names.py`.

```sh
python scripts/inat_common_names.py
```

### WebDataset Format

Then you have to convert them to the WebDataset format.
Edit the config options in `scripts/inat21_to_wds.py`
The options you have to likely change are `inat_root` and `output_root`.
This script will use the 9K classes chosen for pretraining.

```py
python scripts/inat21_to_wds.py
```

This will take a while.

## Training

`finetune.bash` will run the finetuning job.
You will have to edit:

* `.venv/bin/python` to point at your Python interpreter.
* `--train-data` to point at your wds (web data set) shards.
* `--nproc_per_node`, `--batch-size` and `--accum-freq` to make sure your total batch size is large enough.
* `--model` for different architectures.

## Evaluation

`evaluate.bash` will evaluate a model.
There are two evaluation scripts:

1. `src/evaluation/zero_shot_iid.py`
2. `src/evaluation/rich_text_descriptions.py`: This script is very rough and likely needs to be changed.

You will have to edit:

* `.venv/bin/python` to point at your Python interpreter.
* `src.evaluation.zero_shot_iid` to point at your eval script, if it's different.

In `src/evaluation/zero_shot_iid.py`, you'll have to edit `inat_root` and `common_names`, depending on whether you want to use common names or scientific names.
