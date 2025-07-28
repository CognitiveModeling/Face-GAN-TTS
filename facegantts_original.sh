#!/bin/bash
#SBATCH --job-name=FaceGANtts_ablation
#SBATCH --partition=a100-fat-galvani
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-14

module purge
source /home/butz/bst080/miniconda3/etc/profile.d/conda.sh
conda activate /home/butz/bst080/miniconda3/envs/train_env

declare -a NAMES=(
"orig_facetts_scratch_lr1e-4_denoise_0"
"orig_facetts_lrs3_pretrained_endlr_1e-9_denoise_0" 
"orig_facetts_scratch_lr1e-4_denoise_0_gamma_0.01"  
"orig_facetts_lrs3_pretrained_endlr_1e-9_denoise_0_gamma_0.01"
"gan_denoise_0"
"gan_warmup_steps_0"
"orig_facetts_lrs3_pretrained_endlr_1e-9_denoise_0_warmup_steps_0" 
"orig_facetts_scratch_lr1e-4"
"orig_facetts_lrs3_pretrained_endlr_1e-9" 
"orig_facetts_pretrained_lr1e-4"
"spectral_norm"
"warmup_disc_epochs_5"
"warmup_disc_epochs_15"
"freeze_gen_epochs_10"
)


declare -a CONFIGS=(
"use_gan=0 resume_from=./ckpts/no learning_rate=1e-4 early_stopping_patience=9999 denoise_factor=0.0"
"use_gan=0 resume_from=./ckpts/facetts_lrs3.pt end_lr=1e-9 early_stopping_patience=9999 denoise_factor=0.0"
"use_gan=0 resume_from=./ckpts/no learning_rate=1e-4 early_stopping_patience=9999 gamma=0.01 denoise_factor=0.0"
"use_gan=0 resume_from=./ckpts/facetts_lrs3.pt end_lr=1e-9 early_stopping_patience=9999 gamma=0.01 denoise_factor=0.0"
"use_gan=1 denoise_factor=0.0 "
"use_gan=1 warmup_steps=0"
"use_gan=0 resume_from=./ckpts/facetts_lrs3.pt end_lr=1e-9 early_stopping_patience=9999 denoise_factor=0.0 warmup_steps=0"
"use_gan=0 resume_from=./ckpts/no learning_rate=1e-4 early_stopping_patience=9999"
"use_gan=0 resume_from=./ckpts/facetts_lrs3.pt end_lr=1e-9 early_stopping_patience=9999"
"use_gan=0 resume_from=./ckpts/facetts_lrs3.pt learning_rate=1e-4 early_stopping_patience=9999"
"use_spectral_norm=1"
"warmup_disc_epochs=5"
"warmup_disc_epochs=15"
"freeze_gen_epochs=10"
)

RUN_NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
CFG_STRING=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

mkdir -p ablations_facetts
LOG_BASENAME=ablations_facetts/${RUN_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
exec > "${LOG_BASENAME}.out" 2> "${LOG_BASENAME}.err"

export working_dir=/mnt/lustre/work/butz/bst080/experiments/${RUN_NAME}_$SLURM_JOB_ID
mkdir -p "$working_dir"

for KV in $CFG_STRING ; do
    IFS='=' read VAR VAL <<< "$KV"
    export "$VAR"="$VAL"
done

echo "=== $RUN_NAME ==="
echo "working_dir  : $working_dir"
echo "env settings : $CFG_STRING"
echo "--------------"

srun python /mnt/lustre/work/butz/bst080/faceGANtts/train.py
