#!/bin/bash -x

### Job on 1 node. Expect speed: 1600 images/second
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=350000M
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1

#SBATCH --partition=a100-preemptable
#SBATCH --time=3-00:00
#SBATCH --no-requeue

#SBATCH --job-name=VITB32-200M
#SBATCH --error=/mnt/lustre/bethge/mwolff70/new_bow/training_jobs/%j.out # jobid_taskid_arrayid
#SBATCH --output=/mnt/lustre/bethge/mwolff70/new_bow/training_jobs/%j.out

#SBATCH --exclude=r2s-n31    # there is an exclude, be careful!!!!

name=VITB32_bowv1
model=ViT-B-32-bow
batch_size=600  # 700 is Max for 1 A100 node without grad accumulation
accum_freq=7

logdir="/mnt/lustre/bethge/mwolff70/new_bow/logs/"
save_dir_name=${name}_$(date +%y%m%d_%H%M%S)
lr=5e-4
epochs=32

set -e # exit when any command fails

export MASTER_PORT=12802

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd  /mnt/lustre/bethge/mwolff70/new_bow/code/open_clip/
export PYTHONPATH="$PYTHONPATH:$PWD/src"

srun --cpu_bind=v --accel-bind=gn \
  singularity exec --nv \
    -B /scratch_local \
    -B /mnt/lustre \
    /mnt/lustre/bethge/mwolff70/singularity_images/openclip/openclip.sif \
      python -u src/training/main.py \
        --save-frequency 1 \
        --report-to tensorboard \
        --train-data='/mnt/lustre/DATASETS/laion400m/laion400m-data/{00000..41455}.tar' \
        --dataset-type webdataset \
        --warmup 2000 \
        --batch-size ${batch_size} \
        --epochs ${epochs} \
        --workers 8 \
        --model ${model} \
        --seed 0 \
        --local-loss \
        --gather-with-grad \
        --lr ${lr} \
        --force-quick-gelu \
        --name ${save_dir_name}\
        --accum-freq ${accum_freq}\
        --logs ${logdir} \
        --train-num-samples 1000000