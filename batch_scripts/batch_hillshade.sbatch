#!/bin/bash 
#SBATCH --partition=short-serial
####SBATCH --account=short4hr
#SBATCH --job-name=gen_shading_masks
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=12:00:00
####SBATCH --array=0-365 # day
#SBATCH --array=0-7999%2000 # chunks
source /home/users/doran/software/envs/pytorch/bin/activate
python ../terrain_shading.py ${SLURM_ARRAY_TASK_ID} 8000

