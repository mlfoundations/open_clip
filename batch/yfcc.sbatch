#!/bin/bash -x
#SBATCH --output=open_clip/logs/SLURM/out_%A_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gpus-per-node=1
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=lit_singularity_1gpu_yfcc
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge

singularity exec --nv \
    $(for sqf in /vast/work/public/ml-datasets/yfcc15m/data/*.sqf; do echo "--overlay $sqf:ro"; done) \
	--overlay /scratch/bf996/singularity_containers/openclip_env_cuda.ext3:ro \
	/scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
	/bin/bash -c 'source /ext3/env.sh; export PYTHONPATH="$PYTHONPATH:$PWD/open_clip/src";\
    python -u open_clip/src/training/main.py \
    --train-data="$PWD/datasets/yfcc15m/yfcc-small-metadata.csv" \
    --csv-separator "," \
    --save-frequency 1 \
    --report-to tensorboard \
    --warmup 2000 \
    --batch-size=256 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=2 \
    --workers=8 \
    --model=RN50'