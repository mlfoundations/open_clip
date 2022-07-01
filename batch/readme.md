# OpenCLIP Batch Jobs

This brief readme gives an overview on how to run training jobs using OpenCLIP in NYU's HPC environment.

## Clone the Repo

Navigate to your working directory, eg, scratch/<USERNAME>

git clone https://github.com/NYU-DICE-Lab/open_clip

## Overlays

You will need to generate ext3 images use as overlays. Please follow the procedure described by [HPC](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda).

For NVIDIA CUDA, use this version of PyTorch in your image: pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

If you do not name your image openclip_env_cuda.ext3, you will need to modify run-singularity.bash accordingly.

For AMD ROCM, use this version of PyTorch in your image: pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm4.5.2

If you do not name your image openclip_env_rocm.ext3, you will need to modify run-singularity-rocm.bash accordingly.

and don't forget:

```
pip install -r requirements-training.txt
```

## Basic sbatch template

You now have everything you need to run a job.

You can find example templates in the batch directory. These can be modified according to the particular needs of your run.

You will need to sync this repo to your scratch directory prior to first run. All batch scripts expect to be run from the open_clip directory.

Please be sure to replace any PATH variables as appropriate for your particular environment.

## GPUs

### AMD

AMD Notes:

16GB VRAM is available on the AMD cards.

16 GPUs can be accessed using 2 nodes, 8 GPUs per node.

```bash
#SLURM
#SBATCH --gres=gpu:mi50:*

#SRUN
/bin/bash src/script/run-singularity-rocm.bash \
```

### NVIDIA

VRAM may be anywhere from 16GB to 40GB, depending on which cards you request or are assigned.

16 GPUs can be accessed using 4 nodes, 4 GPUs per node.

```bash
#SLURM
#SBATCH --gres=gpu:*

#SRUN
/bin/bash src/script/run-singularity.bash \
```

## Datasets (Train)

### CC12M

```bash
# OPENCLIP
--dataset-type webdataset \
--train-data "/vast/work/public/ml-datasets/cc12m/{00000..01243}.tar" \
--train-num-samples 10968539 \
```

### LAION400M

```bash
# OPENCLIP
## Feel free to adjust train-num-samples if you want to train on a subset of LAION
--dataset-type webdataset \
--train-data "/vast/work/public/ml-datasets/laion400m/{00000..41400}.tar" \
--train-num-samples 400000000 \
```

### YFCC15M

```bash
# SLURM
#SBATCH --mem=128GB

# run-singularity.bash
$(for sqf in /vast/work/public/ml-datasets/yfcc15m/data/*.sqf; do echo "--overlay $sqf:ro"; done) \

# OPENCLIP
--train-data="PATH/TO/yfcc-small-metadata.csv" \
--csv-separator "," \
```

## Datasets (Validation)

### ImageNet Validation

You will need to request access from NYU HPC before you can use ImageNet

```bash
# SINGULARITY
--overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \

# OPENCLIP
--imagenet-val "/imagenet/val/" \
```

### Imagenet V2 Validation

ImageNet V2 should download and install automatically the first time your script calls it. Be sure to provide it with a path with write access.

```bash
--imagenet-v2 "INSTALL_PATH" \
```

## Arguments to OpenCLIP

The easiest way to get a sense of the arguments OpenCLIP accepts is to read the arg parser.

```
src/training/params.py
```