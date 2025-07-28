#!/bin/bash
#SBATCH -J face_tts_full_eval
#SBATCH --partition=2080-galvani
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=plot_mel_f0_%j.out
#SBATCH --error=plot_mel_f0_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=debie1997@yahoo.de


# load your environment
source /home/butz/bst080/miniconda3/etc/profile.d/conda.sh
conda activate /home/butz/bst080/miniconda3/envs/train_env

# make sure repo is on PYTHONPATH
export PYTHONPATH="/mnt/lustre/work/butz/bst080/faceGANtts:$PYTHONPATH"

# paths
GT_WAV="/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/wav/test/spk1019/00014.wav"
GEN_WAV="/mnt/lustre/work/butz/bst080/faceGANtts/inference_outputs/v1538404_hinge/spk1019/00014.wav"
OUT_PNG="./plots/spk1019_00014_comparison.png"

mkdir -p "$(dirname "$OUT_PNG")"

# run
python melspec_plots.py \
    --ground_truth "$GT_WAV" \
    --generated    "$GEN_WAV" \
    --out          "$OUT_PNG"
