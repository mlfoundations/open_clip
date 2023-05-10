#!/bin/bash
#SBATCH --partition=g80
#SBATCH --account=laion
#SBATCH --job-name=TextTextCLIP
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
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


cd /admin/home-jianz/open_clip/src
export PYTHONPATH="$PYTHONPATH:/admin/home-jianz/open_clip/src"

EXP_NAME=""

#srun --comment laion --cpu_bind=v --accel-bind=gn torchrun --nproc_per_node 4 --max_restarts=3 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 -m training.main \
srun --comment laion --cpu_bind=v --accel-bind=gn python3 -m training.main \
    --save-frequency 1 \
    --dataset-type="webdataset" \
    --text-a-key="text_a" \
    --text-b-key="text_b" \
    --report-to wandb \
    --wandb-project-name="TextTextCLIP" \
    --train-data="/fsx/home-jianz/mycache/huggingface/hub/datasets--lingjzhu--laion-multi-2B/snapshots/7ec0d572ac4d8da1e6997ed32e383ab63967e05d/laion-multi-2B-{000..127}.tar" \
    --train-num-samples 135646078 \
    --warmup 2000 \
    --batch-size=2048 \
    --precision amp_bfloat16 \
    --lr=0.001 \
    --wd=0.2 \
    --epochs=97 \
    --workers=1 \
    --model="Siamese-xlm-roberta-large" \
    --seed 0 \
    --log-every-n-steps 5 \
    --local-loss \
    --gather-with-grad \
    --ddp-static-graph \
    --grad-checkpointing \
    --model_type="Siamese_CLANP" \
    --debug \
    --sts-val-data="lingjzhu/sts17-crosslingual" 

