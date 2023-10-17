#!/bin/bash 
#SBATCH --partition=short-serial 
#SBATCH --job-name=traindwnscale
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=10000
#SBATCH --array=0-5  ##0-2 ## 3-5

source /home/users/doran/software/envs/pytorch/bin/activate
python ../train_script2.py ${SLURM_ARRAY_TASK_ID}
