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

cd /admin/home-mitchellw/open_clip/src
export PYTHONPATH="$PYTHONPATH:/admin/home-mitchellw/open_clip/src"

EXP_NAME="test-B-32-laion5b-lr1e-3-bs90k"

srun --comment laion --cpu_bind=v --accel-bind=gn python -m open_clip_train.main \
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
