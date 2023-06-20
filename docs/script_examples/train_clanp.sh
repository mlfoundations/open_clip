#!/bin/bash
#SBATCH --partition=g40
#SBATCH --account=laion
#SBATCH --job-name=CLANP
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=10
#SBATCH --output=logs/%x_%j.out
#SBATCH --comment=laion
#SBATCH --open-mode=append
#SBATCH --exclusive
source /etc/profile.d/modules.sh
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



srun --comment laion --cpu_bind=v --accel-bind=gn python3 -m training.main \
    --save-frequency 1 \
    --dataset-type="webdataset" \
    --text-a-key="txt" \
    --text-b-key="txt" \
    --report-to wandb \
    --wandb-project-name="CLANP" \
    --train-data="pipe:aws s3 --cli-connect-timeout 1000 --cli-read-timeout 1000 cp s3://s-laion/pile/pile_wds/pile-{00000..01159}.tar -" \
    --train-num-samples=1000000 \
    --warmup 5000 \
    --batch-size=256 \
    --lr=0.00001 \
    --wd=0.01 \
    --epochs=100 \
    --workers=10 \
    --model="Siamese-xlm-roberta-large" \
    --save-most-recent \
    --seed 42 \
    --local-loss \
    --gather-with-grad \
    --ddp-static-graph \
    --grad-checkpointing \
    --precision amp_bfloat16 \
    --model-type "SiameseCLANP" \
    --unsupervised-pretraining \
    --context-length=256 \
    --logs /fsx/home-jianz/logs \
    --resume /fsx/home-jianz/logs/2023_05_25-09_21_35-model_Siamese-xlm-roberta-large-lr_1e-05-b_256-j_10-p_amp_bfloat16/checkpoints/epoch_latest.pt

