#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=PAS1576
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=48
#SBATCH --job-name=baseline
#SBATCH --time=84:00:00
#SBATCH --partition=gpu

source $HOME/projects/open_clip/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3

.venv/bin/torchrun --nproc-per-node=4 -m training.main \
  --save-frequency 1 \
  --report-to wandb \
  --wandb-project-name open_clip \
  --train-data '/fs/ess/PAS2136/open_clip/data/evobio10m-v1/224x224/shard-{000000..000194}.tar' \
  --dataset-type webdataset \
  --dataset-resampled \
  --warmup 1000 \
  --batch-size 2048 \
  --accum-freq 2 \
  --epochs 40 \
  --workers 8 \
  --model ViT-B-16 \
  --log-every-n-steps 10 \
  --eps 1e-6 \
  --lr 1e-4 \
  --resume latest \
  --name baseline \
  --seed 42 \
  --local-loss \
  --gather-with-grad \
  --grad-checkpointing \
  --precision amp \
  --logs /fs/ess/PAS2136/open_clip/logs/

