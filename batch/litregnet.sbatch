#!/bin/bash -x
#SBATCH --output=logs/out_%A_%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:mi50:8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=lit_regnetx_320_laion400m_4node_b384
#SBATCH --dependency=after:21593845

module purge;

#debug flags
echo $SLURM_JOB_NAME

# COMMENTS
# resume point 2022_06_17-23_06_30 was trained on 350m LAION samples for the first 2 epochs

#env vars
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
echo "MASTER_ADDR="$MASTER_ADDR

#run command
srun --cpu_bind=v --accel-bind=v \
    /bin/bash src/script/run-singularity-rocm.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py \
    --save-frequency 1 \
    --report-to wandb \
    --dataset-type webdataset \
    --train-data "/vast/work/public/ml-datasets/laion400m/{00000..41400}.tar" \
    --train-num-samples 400000000 \
    --imagenet-val "/imagenet/val/" \
    --zeroshot-frequency=1 \
    --warmup 2000 \
    --batch-size=384 \
    --wd=0.1 \
    --epochs=32 \
    --workers=4 \
    --model=timm-regnetx_320 \
    --pretrained-image \
    --lock-image \
    --seed 0 \
    --resume "/scratch/bf996/open_clip/logs/2022_06_26-05_54_37-model_timm-regnetx_320-lr_0.0005-b_384-j_4-p_amp/checkpoints/epoch_4.pt" \
    --local-loss \
    --gather-with-grad'

