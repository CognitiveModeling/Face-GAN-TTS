#!/bin/bash
#SBATCH -J evaluate_tts
#SBATCH --partition=2080-galvani
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx2080ti:8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=evaluation-%j.out
#SBATCH --error=evaluation-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=debie1997@yahoo.de
#SBATCH --export=ALL 

# Load environment
source /home/butz/bst080/miniconda3/etc/profile.d/conda.sh
conda activate /home/butz/bst080/miniconda3/envs/train_env  #/mnt/qb/home/butz/bst080/miniconda3/envs/label_env

export PYTHONPATH="/mnt/lustre/work/butz/bst080/faceGANtts:$PYTHONPATH"
srun python /mnt/lustre/work/butz/bst080/faceGANtts/evaluation/eval.py 
