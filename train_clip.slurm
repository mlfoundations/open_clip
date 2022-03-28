#!/bin/bash

#SBATCH --output=out_%A_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_openclip

module purge

singularity exec --nv \
	--overlay /scratch/aaj458/singularity_containers/openclip.ext3:ro \
	--overlay /vast/work/public/ml-datasets/conceptual-captions/cc_data.sqf:ro \
	/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	/bin/bash -c 'source /ext3/env.sh; conda activate openclip; \
				  export PYTHONPATH="$PYTHONPATH:$PWD/src"; \
				  python -u src/training/main.py \
    					--save-frequency 1 \
    					--zeroshot-frequency 1 \
    					--report-to tensorboard \
    					--train-data="/cc_data/Train_GCC-training_output.csv"  \
    					--val-data="/cc_data/Validation_GCC-1.1.0-Validation_output.csv"  \
    					--csv-img-key filepath \
    					--csv-caption-key title \
    					--imagenet-val=/imagenet/val/ \
    					--warmup 10000 \
    					--batch-size=128 \
    					--lr=1e-3 \
    					--wd=0.1 \
    					--epochs=30 \
    					--workers=8 \
    					--model RN50'
