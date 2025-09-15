#!/bin/bash

#SBATCH --job-name=NF_train_array
#SBATCH -q express
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem=15G
#SBATCH --output=logs/output_%A_%a.out
#SBATCH --error=logs/error_%A_%a.err
#SBATCH -t 4-0:0:0
#SBATCH --array=0-79

source PATH_TO_BASHRC_FILE
conda activate myenv

# Create required directories if not exist
mkdir -p logs models_eP metrics_eP plots_eP

# Run training for the given array task
python train_nf_job.py ${SLURM_ARRAY_TASK_ID}
