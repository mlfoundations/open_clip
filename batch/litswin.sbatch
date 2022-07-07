#!/bin/bash -x
#SBATCH --output=logs/SLURM/out_%A_%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#bzatch --wait-all-nodes=1
#SBATCH --job-name=lit_swin_laion400m_4node_b512.sbatch
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge;

#debug flags
echo $SLURM_JOB_NAME

# COMMENTS
# Resume point 2022_06_17-06_05_36 was trained on 50m LAION samples for the first 2 epochs

#env vars
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
echo "MASTER_ADDR="$MASTER_ADDR

#run command
srun --cpu_bind=v --accel-bind=v \
    /bin/bash src/script/run-singularity.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:$PWD/src"; python src/training/main.py --save-frequency 1 --report-to wandb --dataset-type webdataset --train-data "/vast/work/public/ml-datasets/laion400m/{00000..41400}.tar" --train-num-samples 400000000 --imagenet-val "/imagenet/val/" --zeroshot-frequency=1 --warmup 2000 --batch-size=256 --wd=0.1 --epochs=32 --workers=4 --model=timm-swin_base_patch4_window7_224 --pretrained-image --lock-image --seed 0 --resume "/scratch/bf996/open_clip/logs/swin-ep5/checkpoints/epoch_5.pt" --local-loss --gather-with-grad'
