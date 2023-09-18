#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=512gb
#SBATCH --account=PAS2136

echo $SLURM_JOB_NAME

source $HOME/projects/open_clip/pitzer-venv/bin/activate

python scripts/evobio10m/make_wds.py --tag v2 --split val
python scripts/evobio10m/make_wds.py --tag v2 --split train
