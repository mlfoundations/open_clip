#!/bin/bash -x
#SBATCH --output=out_%A_%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --mem=64GB
#SBATCH --gpus-per-node=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=lit_swin_laion400m_4node_tensorboard_b512
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge;
module load anaconda3/2020.07;

echo $SLURM_JOB_NAME

#env vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
cd $SCRATCH/open_clip/
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate ./penv;
pip install tensorboard;
export PATH=./penv/bin:$PATH;
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export MASTER_PORT=12802
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export PYTHONPATH="$PYTHONPATH:$PWD/src"

#run command
srun --cpu_bind=v --accel-bind=v python src/training/main.py \
    --save-frequency 1 \
    --report-to tensorboard \
    --dataset-type webdataset \
    --train-data '/vast/work/public/ml-datasets/laion400m/{00000..41400}.tar' \
    --train-num-samples 400000000 \
    --warmup 2000 \
    --batch-size=512 \
    --wd=0.1 \
    --epochs=2 \
    --workers=4 \
    --model=timm-swin_base_patch4_window7_224 \
    --pretrained-image \
    --lock-image \
    --seed 0 \
    --local-loss \
    --gather-with-grad