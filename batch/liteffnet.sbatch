#!/bin/bash -x

#SBATCH --output=logs/out_%A_%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:mi50:8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=lit_tf_efficientnetv2_xl_in21ft1k_cc12m_1node_b256

module purge;

#debug flags
echo $SLURM_JOB_NAME

#env vars
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
echo "MASTER_ADDR="$MASTER_ADDR

#comments

#trained on cc12m for epochs 1-2

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
    --batch-size=256 \
    --wd=0.1 \
    --epochs=32 \
    --workers=4 \
    --model=timm-tf_efficientnetv2_xl_in21ft1k \
    --pretrained-image \
    --lock-image \
    --resume "logs/2022_06_20-14_24_17-model_timm-tf_efficientnetv2_xl_in21ft1k-lr_0.0005-b_256-j_4-p_amp/checkpoints/epoch_2.pt" \
    --seed 0 \
    --local-loss \
    --gather-with-grad'