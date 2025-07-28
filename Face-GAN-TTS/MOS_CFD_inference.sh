#!/bin/bash
#SBATCH -J facetts_all_pairs
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=allpairs_%j.out
#SBATCH --error=allpairs_%j.err

set -euo pipefail

# ---------- environment ----------
PY=/home/butz/bst080/miniconda3/envs/train_env/bin/python
export PYTHONPATH="/mnt/lustre/work/butz/bst080/faceGANtts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

which python
ls -l $(which python)

INF_SCRIPT=/mnt/lustre/work/butz/bst080/faceGANtts/temp_inference.py
OUT_ROOT=/mnt/lustre/work/butz/bst080/faceGANtts/test

# ---------- Deine sechs Text–Bild-Paare ----------
TEXTS=(
  "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/test/spk2565/00030.txt"
  "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/test/spk5934/00020.txt"
  "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/test/spk9201/00007.txt"
  "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/test/spk2077/00054.txt"
  "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/test/spk4763/00015.txt"
  "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/test/spk4763/00015.txt"
)
IMGS=(
  "/mnt/lustre/work/butz/bst080/data/Chigago_rescaled_Images/CFD-WF-222-092-N_face0.png"
  "/mnt/lustre/work/butz/bst080/data/Chigago_rescaled_Images/CFD-WM-018-002-N_face0.png"
  "/mnt/lustre/work/butz/bst080/data/Chigago_rescaled_Images/CFD-LM-243-075-N_face0.png"
  "/mnt/lustre/work/butz/bst080/data/Chigago_rescaled_Images/CFD-LM-210-156-N_face0.png"
  "/mnt/lustre/work/butz/bst080/data/Chigago_rescaled_Images/CFD-WF-027-003-N_face0.png"
  "/mnt/lustre/work/butz/bst080/data/Chigago_rescaled_Images/CFD-BF-019-001-N_face0.png"
)

# ---------- model + faceTTS-Varianten ----------
declare -A CHECKPOINTS=(
  [CFD_faceGANTTS]="/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1538393/checkpoints/best_epoch_96_step_17848.ckpt"
  [CFD_FACETTS_lrs3_pretrained]="/mnt/lustre/work/butz/bst080/faceGANtts/lightning_logs/version_1561277/checkpoints/epoch=096.ckpt"
)

# ---------- loop über Systeme und Text–Bild-Paare ----------
for SYSTEM in "${!CHECKPOINTS[@]}"; do
  export USE_GAN=$([[ "$SYSTEM" == "gan" ]] && echo 1 || echo 0)
  export resume_from_checkpoint="${CHECKPOINTS[$SYSTEM]}"
  OUT_DIR="${OUT_ROOT}/infer_${SYSTEM}"
  mkdir -p "$OUT_DIR"

  for IDX in "${!TEXTS[@]}"; do
    TEST_TXT="${TEXTS[$IDX]}"
    FACE_PATH="${IMGS[$IDX]}"
    # hier holen wir den Bildnamen ohne .png
    export FACE_TAG="$(basename "$FACE_PATH" .png)"

    echo -e "\n==========  $SYSTEM – Paar $IDX  ==========\n"
    echo "CKPT:     $resume_from_checkpoint"
    echo "TEXT:     $TEST_TXT"
    echo "FACE:     $FACE_PATH"
    echo "OUT_DIR:  $OUT_DIR"
    echo "USE_GAN:  $USE_GAN"
    echo "FACE_TAG: $FACE_TAG"
    echo "-----------------------------------------"

    srun --ntasks=1 \
         "$PY" "$INF_SCRIPT" \
              with \
              test_faceimg="$FACE_PATH" \
              test_txt="$TEST_TXT" \
              use_custom=1 \
              output_dir_orig="$OUT_DIR" \
              output_dir_gan="$OUT_DIR"
  done
done
