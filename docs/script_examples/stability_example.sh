#!/bin/bash
#SBATCH --partition=g40423
#SBATCH --job-name=testopenclip
#SBATCH --nodes 30
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --output=%x_%j.out
#SBATCH --comment=laion
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

cd /admin/home-mitchellw/open_clip/src
export PYTHONPATH="$PYTHONPATH:/admin/home-mitchellw/open_clip/src"

EXP_NAME="test-B-32-laion5b-lr1e-3-bs90k"

srun --comment laion --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 1 \
    --train-data="pipe:aws s3 cp s3://s-datasets/laion5b/{laion2B-data/{000000..231349}.tar,laion2B-multi-data/{000000..226687}.tar,laion1B-nolang-data/{000000..127231}.tar} -" \
    --train-num-samples 135646078 \
    --dataset-type webdataset \
    --dataset-resampled \
    --warmup 2000 \
    --batch-size=375 \
    --epochs=97 \
    --lr 1e-3 \
    --workers=8 \
    --report-to wandb \
    --name ${EXP_NAME} \
    --logs /scratch/logs/ \
    --model ViT-B-32 \
    --seed 0 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --precision amp_bfloat16 \
    --wandb-project-name open_clip6 \
    --resume "latest" \
    --remote-sync s3://s-laion/mitchellw/logs