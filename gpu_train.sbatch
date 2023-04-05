#!/bin/bash 
#SBATCH --partition=orchid 
#SBATCH --account=orchid
#SBATCH --gres=gpu:1
#SBATCH --job-name=dwnscale_train
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --time=4:00:00

source /home/users/doran/software/envs/pytorch/bin/activate
python ./train_script.py
