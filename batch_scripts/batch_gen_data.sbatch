#!/bin/bash 
#SBATCH --partition=short-serial-4hr
#SBATCH --account=short4hr
#SBATCH --job-name=gen_val_batches
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=04:00:00
#SBATCH --array=0-30
source /home/users/doran/software/envs/pytorch/bin/activate
python ./generate_data_batches.py ${SLURM_ARRAY_TASK_ID} val

