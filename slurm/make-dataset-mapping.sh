#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=256gb
#SBATCH --account=PAS2136

echo $SLURM_JOB_NAME

source $HOME/projects/open_clip/pitzer-venv/bin/activate

python scripts/evobio10m/make_mapping.py --tag v2
