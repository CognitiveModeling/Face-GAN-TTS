#!/bin/bash
#SBATCH -J face_tts_full_eval
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH --output=face_tts_%j.out
#SBATCH --error=face_tts_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=debie1997@yahoo.de
#SBATCH --export=ALL

# -------------------------------------------------------------------------
# 1) Environment
# -------------------------------------------------------------------------
source /home/butz/bst080/miniconda3/etc/profile.d/conda.sh
conda activate /home/butz/bst080/miniconda3/envs/train_env

# 2) Run your commands
export USE_GAN=1
export PYTHONPATH="/mnt/lustre/work/butz/bst080/faceGANtts:$PYTHONPATH"

# -------------------------------------------------------------------------
# 2) Paths and model list (unchanged)
# -------------------------------------------------------------------------
declare -A MODELS=(
  #[v1538403_mse]="/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1538403/checkpoints/best_epoch_96_step_17848.ckpt"
  #[v1538404_bce]="/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1538404/checkpoints/best_epoch_96_step_17848.ckpt"
  #[v1538404_hinge]="/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1538393/checkpoints/best_epoch_96_step_17848.ckpt"
  #[v1538404_gamma0.01]="/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1538418/checkpoints/best_epoch_96_step_17848.ckpt"
  #[v1538404_fm]="/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1538398/checkpoints/best_epoch_96_step_17848.ckpt"
  #[FACETTS_finetuned_lr1e-8_gamma0.02_denoised]="/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1561277/checkpoints/epoch=096.ckpt"
  #[FACETTS_LRS3_only]="/mnt/lustre/work/butz/bst080/faceGANtts/ckpts/facetts_lrs3.pt"
#   [FACEGANTTS_denoise0]=/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1561273/checkpoints/best_epoch_96_step_17848.ckpt
#   [FACEGANTTS_denoise0.2]=/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1538420/checkpoints/best_epoch_96_step_17848.ckpt
  #[FACEGANTTS_warmup5]="/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1561280/checkpoints/epoch=096.ckpt"
  #[FACETTS_scratch_lr1e-4_gamma0.02_denoised]="/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1561276/checkpoints/epoch=096.ckpt"
  #[FACEGANTTS_denoise_0]="/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1561273/checkpoints/epoch=096.ckpt"
  #[FACEGANTTS_lrs3]="./ckpts/facetts_lrs3.pt"
#    [FACEGANTTS_1541668]=/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1541668/checkpoints/best_epoch_96_step_17848.ckpt
#    [FACEGANTTS_1543003]=/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1543003/checkpoints/best_epoch_96_step_17848.ckpt
#    [FACEGANTTS_1543102]=/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1543102/checkpoints/best_epoch_96_step_17848.ckpt
#    [FACEGANTTS_1543668]=/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1543668/checkpoints/best_epoch_96_step_17848.ckpt
#    [FACEGANTTS_1544057]=/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1544057/checkpoints/best_epoch_96_step_17848.ckpt
#    [FACEGANTTS_1544224]=/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1544224/checkpoints/best_epoch_96_step_17848.ckpt
#   [FACEGANTTS_1544227]=/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1544227/checkpoints/epoch=096.ckpt
    [FACEGANTTS_1561281]=/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1561281/checkpoints/epoch=096.ckpt
    [FACEGANTTS_1561282]=/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1561282/checkpoints/epoch=096.ckpt
)

OUT_ROOT=./inference_abl_extras #./inference_outputs
GT_DIR=/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/wav/test
mkdir -p "${OUT_ROOT}"

for TAG in "${!MODELS[@]}"; do
    echo -e "\n====================  ${TAG}  ====================\n"

    export resume_from_checkpoint="${MODELS[$TAG]}"

    # ------ Zielordner & passender Sacred-Parameter ----------------------
    export OUTPUT_DIR="${OUT_ROOT}/${TAG}"
    mkdir -p "${OUTPUT_DIR}"
    export DYNAMIC_EVAL_PATH="${OUTPUT_DIR}"

    if [[ "${USE_GAN}" -eq 1 ]]; then
        OUT_KEY="output_dir_gan"
    else
        OUT_KEY="output_dir_orig"
    fi
    # --------------------------------------------------------------------

    echo "Inference  →  ${OUTPUT_DIR}"
    srun --ntasks=1 \
         python /mnt/lustre/work/butz/bst080/faceGANtts/inference.py \
              with "${OUT_KEY}=${OUTPUT_DIR}"

    echo "Evaluation"
    srun --ntasks=1 \
         python /mnt/lustre/work/butz/bst080/faceGANtts/evaluation/eval.py \
              with use_gan="${USE_GAN}" \
                   ground_truth_dir="${GT_DIR}" \
                   "${OUT_KEY}=${OUTPUT_DIR}"
done

echo "All models processed.  Results are printed above and stored in the *.out file."







