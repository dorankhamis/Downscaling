#!/bin/bash 
#SBATCH --partition=short-serial-4hr
#SBATCH --account=short4hr
#SBATCH --job-name=UK_dayruns
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=16000
#SBATCH --array=0-365

source /home/users/doran/software/envs/pytorch/bin/activate
python ../run_UK_tests_day.py 2019 ${SLURM_ARRAY_TASK_ID}


