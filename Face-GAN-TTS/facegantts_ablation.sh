#!/bin/bash
#SBATCH --job-name=FaceGANtts_ablation
#SBATCH --partition=a100-fat-galvani
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-66

module purge
source /home/butz/bst080/miniconda3/etc/profile.d/conda.sh
conda activate /home/butz/bst080/miniconda3/envs/train_env

declare -a NAMES=(
"disc_loss_hinge"             # 0
"orig_facetts_lrs3ckpt_lr1e-4"   # 1
"orig_facetts_scratch_endlr_1e-9" # 2
"pitch"                       # 3
"energy"                      # 4
"fm"                          # 5
"pitch_energy"                # 6
"pitch_fm"                    # 7
"energy_fm"                   # 8
"pitch_energy_fm"             # 9
"disc_mse"                    # 10
"disc_bce"                    # 11
"disc_ls"                     # 12
"spectral_norm"               # 13
"disc_loss_bce"               # 14
"disc_loss_ls"                # 15
"lambda_adv_0.2"              # 16
"disc_lr_1e-3"                # 17
"lr_1e-6"                     # 18
"warmup_2"                    # 19
"warmup_5"                    # 20
"warmup_15"                   # 21
"freeze_2"                    # 22
"freeze_5"                    # 23
"freeze_10"                   # 24
"gamma_0.01"                  # 25
"gamma_0.05"                  # 26
"denoise_0.2"                 # 27
"denoise_0.4"                 # 28
"denoise_0.5"                 # 29
"denoise_0.6"                 # 30
"r1_penalty_0"                # 31
"r1_gamma_5"                  # 32
"r1_gamma_10"                 # 33
"r1_gamma_20"                 # 34
"kernel_width_3"              # 35
"kernel_width_7"              # 36
"kernel_width_9"              # 37
"kernel_width_12"             # 38
"kernel_width_15"             # 39
"kernel_height_3"             # 40
"kernel_height_5"             # 41
"kernel_height_7"             # 42
"kernel_height_9"             # 43
"kernel_height_15"            # 44
"disc_padding_4"              # 45
"disc_padding_7"              # 46
"disc_stride_2"               # 47
"disc_stride_3"               # 48
"disc_lrelu_0.1"              # 49
"disc_lrelu_0.2"              # 50
"disc_lrelu_0.3"              # 51
"lambda_adv_0.1"              # 52
"lambda_adv_0.3"              # 53
"lambda_adv_0.5"              # 54
"lambda_adv_0.9"              # 55
"lambda_adv_1.0"              # 56
"disc_lr_1e-6"                # 57
"disc_lr_1e-5"                # 58
"disc_lr_1e-4"                # 59
"lr_1e-7"                     # 60
"lr_1e-4"                     # 61
"gamma_0.02"                  # 62
"kernel_width_5"              # 63
"kernel_height_12"            # 64
"disc_padding_6"              # 65
"disc_stride_1"               # 66
)

declare -a CONFIGS=(
"disc_loss_type=hinge"
"use_gan=0 resume_from=./ckpts/facetts_lrs3.pt end_lr=1e-9 early_stopping_patience=9999"
"use_gan=0 resume_from=./ckpts/no learning_rate=1e-4 early_stopping_patience=9999"
"use_pitch_loss=1"
"use_energy_loss=1"
"use_fm_loss=1"
"use_pitch_loss=1 use_energy_loss=1"
"use_pitch_loss=1 use_fm_loss=1"
"use_energy_loss=1 use_fm_loss=1"
"use_pitch_loss=1 use_energy_loss=1 use_fm_loss=1"
"disc_loss_type=mse"
"disc_loss_type=bce"
"disc_loss_type=ls"
"use_spectral_norm=1"
"disc_loss_type=bce"
"disc_loss_type=ls"
"lambda_adv=0.2"
"disc_learning_rate=0.001"
"learning_rate=1e-06"
"warmup_disc_epochs=2"
"warmup_disc_epochs=5"
"warmup_disc_epochs=15"
"freeze_gen_epochs=2"
"freeze_gen_epochs=5"
"freeze_gen_epochs=10"
"gamma=0.01"
"gamma=0.05"
"denoise_factor=0.2"
"denoise_factor=0.4"
"denoise_factor=0.5"
"denoise_factor=0.6"
"use_r1_penalty=0"
"r1_gamma=5"
"r1_gamma=10"
"r1_gamma=20"
"kernel_width=3"
"kernel_width=7"
"kernel_width=9"
"kernel_width=12"
"kernel_width=15"
"kernel_height=3"
"kernel_height=5"
"kernel_height=7"
"kernel_height=9"
"kernel_height=15"
"disc_padding=4"
"disc_padding=7"
"disc_stride=2"
"disc_stride=3"
"disc_lrelu_slope=0.1"
"disc_lrelu_slope=0.2"
"disc_lrelu_slope=0.3"
"lambda_adv=0.1"
"lambda_adv=0.3"
"lambda_adv=0.5"
"lambda_adv=0.9"
"lambda_adv=1.0"
"disc_learning_rate=1e-6"
"disc_learning_rate=1e-5"
"disc_learning_rate=0.0001"
"learning_rate=1e-07"
"learning_rate=1e-4"
"gamma=0.02"
"kernel_width=5"
"kernel_height=12"
"disc_padding=6"
"disc_stride=1"
)

RUN_NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
CFG_STRING=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

mkdir -p ablations
LOG_BASENAME=ablations/${RUN_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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
