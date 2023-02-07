#!/bin/bash
#SBATCH --partition=g40423
#SBATCH --job-name=gtopenclip
#SBATCH --nodes 4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --output=%x_%j.out
#SBATCH --account=laion
#SBATCH --open-mode=append
#SBATCH --exclusive

module load openmpi
module load cuda/11.7

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

cd /fsx/home-gamaga/open_clip/src
export PYTHONPATH="$PYTHONPATH:/fsx/home-gamaga/open_clip/src"

EXP_NAME="gt-B-32-cc12m-lr5e-4-bs8k"

srun --comment laion --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 1 \
    --train-data="s3://s-laion/cc12m/shards/{00000..01242}.tar" \
    --train-num-samples 10185253 \
    --dataset-type webdataset \
    --dataset-resampled \
    --warmup 2000 \
    --batch-size=256 \
    --epochs=32 \
    --lr 5e-4 \
    --workers=8 \
    --report-to wandb \
    --name ${EXP_NAME} \
    --logs /fsx/home-gamaga/logs/ \
    --model ViT-B-32 \
    --seed 0 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --precision amp_bfloat16 \
    --wandb-project-name open_clip6 \
    --resume "latest" \
    --imagenet-val "/fsx/rom1504/imagenetval/imagenet_validation" \
    --remote-sync s3://s-laion/gamaga/logs