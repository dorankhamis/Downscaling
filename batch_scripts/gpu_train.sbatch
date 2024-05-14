#!/bin/bash 
#SBATCH --partition=orchid 
#SBATCH --account=orchid
#SBATCH --gres=gpu:1
#SBATCH --job-name=dwnscale
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=14000
####SBATCH --array=0,1,2,3,4,5,6
#SBATCH --array=6

source /home/users/doran/software/envs/pytorch/bin/activate
python ../train_script3.py ${SLURM_ARRAY_TASK_ID}
