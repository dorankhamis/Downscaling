#!/bin/bash 
#SBATCH --partition=short-serial-4hr
#SBATCH --account=short4hr
#SBATCH --job-name=UK_5
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=10000
#SBATCH --array=1-12

source /home/users/doran/software/envs/pytorch/bin/activate
python ./run_UK_tests.py 5 2015 ${SLURM_ARRAY_TASK_ID}



