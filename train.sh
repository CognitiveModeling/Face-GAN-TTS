#!/bin/bash
#SBATCH --job-name=Facetts_LRS3ckpts_gamma0.01_denoise0_lr1ende9_lr1e8_noGAN_gpu1_worker2
#SBATCH --partition=a100-fat-galvani 
#SBATCH --gres=gpu:4 
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=3-00:00:00
#SBATCH --output=Facetts_LRS3ckpts_gamma0.01_denoise0_lr1ende9_lr1e8_noGAN_gpu1_worker2_%j.out
#SBATCH --error=Facetts_LRS3ckpts_gamma0.01_denoise0_lr1ende9_lr1e8_noGAN_gpu1_worker2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=debie1997@yahoo.de
#SBATCH --exclude=galvani-cn112,galvani-cn203

# --- ENV SETUP ---
module purge
source /home/butz/bst080/miniconda3/etc/profile.d/conda.sh
conda activate /home/butz/bst080/miniconda3/envs/train_env

hostname
nvidia-smi
which python

# --- ENVIRONMENT VARS FOR Sacred CONFIG ---
export use_gan=0
export resume_from=./ckpts/facetts_lrs3.pt
export end_lr=1e-9
export num_gpus=1
export early_stopping_patience=9999
export gamma=0.01
export num_workers=2
export denoise_factor=0
export warmup_steps=0

# --- RUN ---
python /mnt/lustre/work/butz/bst080/faceGANtts/train.py




# #!/bin/bash
# #SBATCH --job-name=FaceGANtts
# #SBATCH --partition=a100-fat-galvani
# #SBATCH --gres=gpu:4 
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=50G
# #SBATCH --time=3-00:00:00
# #SBATCH --output=FaceGANtts_acc_%j.out
# #SBATCH --error=FaceGANtts_acc_%j.err
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=debie1997@yahoo.de
# #SBATCH --exclude=galvani-cn112,galvani-cn203

# module purge
# source /home/butz/bst080/miniconda3/etc/profile.d/conda.sh
# conda activate /home/butz/bst080/miniconda3/envs/train_env

# hostname
# nvidia-smi
# which python

# python /mnt/lustre/work/butz/bst080/faceGANtts/train.py
