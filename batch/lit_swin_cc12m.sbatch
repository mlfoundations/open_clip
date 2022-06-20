#!/bin/bash -x
#SBATCH --output=out_%A_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=open_clip_LiT
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge;
module load anaconda3/2020.07;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
cd $SCRATCH/open_clip/
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate ./penv;
export PATH=./penv/bin:$PATH;
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export PYTHONPATH="$PYTHONPATH:$PWD/src"

srun --cpu_bind=v --accel-bind=gn python -u src/training/main.py \
    --save-frequency 1 \
    --dataset-type webdataset \
    --train-data '/vast/work/public/ml-datasets/cc12m/{00000..01243}.tar' \
    --train-num-samples 10968539 \
    --warmup 2000 \
    --batch-size=256 \
    --wd=0.1 \
    --epochs=5 \
    --workers=4 \
    --model=timm-swin_base_patch4_window7_224 \
    --pretrained-image \
    --debug \
    --lock-image \
    --seed 0 \
    --local-loss \
    --gather-with-grad