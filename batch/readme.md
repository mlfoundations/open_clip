# Basic sbatch template

This can be modified as needed, depending on your needs.

```bash
#!/bin/bash -x
#SBATCH --output=logs/out_%A_%j.log
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --mem=64GB
#SBATCH --gpus-per-node=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=lit_tf_efficientnetv2_xl_in21ft1k_laion400m_2node_b256_sing
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge;

#debug flags
echo $SLURM_JOB_NAME

#env vars
export PATH=./penv/bin:/home/USER/.local/bin:$PATH;
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
echo "MASTER_ADDR="$MASTER_ADDR

#run command
srun --cpu_bind=v --accel-bind=v \
    /bin/bash src/script/run-singularity.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py \
    --save-frequency 1 \
    --report-to tensorboard \
    --dataset-type webdataset \
    --train-data "/vast/work/public/ml-datasets/laion400m/{00000..41400}.tar" \
    --train-num-samples 400000000 \
    --imagenet-val "/imagenet/val/" \
    --warmup 10000 \
    --batch-size=256 \
    --wd=0.1 \
    --epochs=2 \
    --workers=4 \
    --model=timm-tf_efficientnetv2_xl_in21ft1k \
    --pretrained-image \
    --lock-image \
    --seed 0 \
    --local-loss \
    --gather-with-grad'
```

## GPUs

### AMD

```bash
#SLURM
#SBATCH --gres=gpu:mi50:1

#SRUN
/bin/bash src/script/run-singularity-rocm.bash \
```

### NVIDIA

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

### YFCC15M

```bash
# SINGULARITY
$(for sqf in /vast/work/public/ml-datasets/yfcc15m/data/*.sqf; do echo "--overlay $sqf:ro"; done) \
# OPENCLIP
--train-data="PATH/TO/yfcc-small-metadata.csv" \
--csv-separator "," \
```

### LAION400M

```bash
# OPENCLIP
## Feel free to adjust train-num-samples if you want to train on a subset of LAION
--dataset-type webdataset \
--train-data "/vast/work/public/ml-datasets/laion400m/{00000..41400}.tar" \
--train-num-samples 400000000 \
```

## Datasets (Validation)

### ImageNet Validation

You will need to request access from HPC to use ImageNet

```bash
# SINGULARITY
--overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \

# OPENCLIP
--imagenet-val "/imagenet/val/" \
```