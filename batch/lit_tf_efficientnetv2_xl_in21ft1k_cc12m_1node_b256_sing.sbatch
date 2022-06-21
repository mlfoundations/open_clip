#!/bin/bash -x
#SBATCH --output=logs/out_%A_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --mem=64GB
#SBATCH --gpus-per-node=2
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=lit_tf_efficientnetv2_xl_in21ft1k_cc12m_1node_b256_sing
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge;

#debug flags
echo $SLURM_JOB_NAME

#env vars
export PATH=./penv/bin:/home/bf996/.local/bin:$PATH;
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
    --report-to wandb \
    --dataset-type webdataset \
    --train-data "/vast/work/public/ml-datasets/cc12m/{00000..01243}.tar" \
    --train-num-samples 10968539 \
    --imagenet-val "/imagenet/val/" \
    --imagenet-v2 "imagenetv2-mff-val" \
    --zeroshot-frequency=1 \
    --warmup 2000 \
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