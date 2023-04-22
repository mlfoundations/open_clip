#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=text-text-audio-clip
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=LOGS/%x_%j.out
#SBATCH --comment=laion
#SBATCH --open-mode=append
#SBATCH --exclusive

module load openmpi
module load cuda/11.7

export MASTER_ADDR=`hostname`
export MASTER_PORT=12802
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG=info

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

cd /fsx/marianna/clap/open_clip/src

source /fsx/marianna/clap/my-project-env/bin/activate && srun --comment laion python -m training.main \
    --save-frequency 1 \
    --train-data="pipe: aws s3 cp s3://s-laion-audio/webdataset_tar/VGGSound/train/{0..99}.tar -" \
    --train-num-samples 100000 \
    --dataset-type webdataset \
    --dataset-resampled \
    --warmup 1000 \
    --batch-size=32 \
    --epochs=4 \
    --lr 3e-3 \
    --workers=1 \
    --report-to wandb \
    --text-a-key="json" \
    --text-b-key="flac" \
    --logs /scratch/tta-logs/ \
    --model="bert-base-longform-context" \
    --seed 0 \
    --log-every-n-steps 5 \
    --local-loss \
    --gather-with-grad \
    --ddp-static-graph \
    --grad-checkpointing \
    --model_type="text-audio" \
    --wandb-project-name text_text_audio__clip \
    --precision amp_bfloat16 \
    --resume "latest" \
