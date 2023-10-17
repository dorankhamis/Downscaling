#!/bin/bash 
#SBATCH --partition=short-serial-4hr
#SBATCH --account=short4hr
#SBATCH --job-name=merge_shading_masks
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=3:00:00
#SBATCH --array=0-365 # day

source /home/users/doran/software/envs/pytorch/bin/activate
python ../merge_shading_maps.py ${SLURM_ARRAY_TASK_ID}

