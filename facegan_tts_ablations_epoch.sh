resume_from = os.getenv("resume_from", "./ckpts/facetts_lrs3.pt")  

Damit kannst du jedes Experiment exakt an der letzten gespeicherten **Lightning-Checkpoint-Datei** (z. B. `epoch=95-step=*.ckpt`) weiterlaufen lassen.  
Damit nach dem Weiter­trainieren **immer** ein Checkpoint bei Epoche 96 landet, hast du zwei Möglichkeiten:

| Weg | was tun |
|-----|---------|
|**1 · Lightning-Standard nutzen**| lasse den Default-`ModelCheckpoint` aktiv (speichert immer am Ende jeder Epoche).<br>→ Keine Änderung nötig.|
|**2 · Eigene Schritt-Intervalle**| setze in der Umgebungs­variable <br>`save_step=1000`  (o. ä.) so, dass ein Speichern garantiert in Epoche 96 liegt. Das Feld wird im Training-Script ausgewertet. 

---

### Minimal-SBATCH-Skript für die sechs Runs  

Speichere z. B. als `run_reruns.sbatch` – der Array-Index 0-5 deckt genau die von dir gewünschten Konfigurationen ab.

```bash
#!/bin/bash
#SBATCH --job-name=FaceTTS_reruns
#SBATCH --partition=a100-fat-galvani
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-5

module purge
source /home/butz/bst080/miniconda3/etc/profile.d/conda.sh
conda activate /home/butz/bst080/miniconda3/envs/train_env

# -------- Konfigurationen ------------------------------------
declare -a NAMES=(
  "nogan_lrs3_resume"      # 0
  "nogan_no_lr1e-4_resume" # 1
  "spectral_norm"          # 2
  "denoise_0.0"            # 3
  "warmup_disc_15"         # 4
  "warmup_disc_5"          # 5
)

declare -a CONFIGS=(
  # 0  – GAN aus, LRS3-Start, extrem kleines End-LR
  "use_gan=0 resume_from=/mnt/lustre/work/butz/bst080/experiments/nogan_lrs3/epoch=95.ckpt end_lr=1e-9 early_stopping_patience=9999"

  # 1  – GAN aus, kein LRS3 Start, Lern­rate 1e-4
  "use_gan=0 resume_from=/mnt/lustre/work/butz/bst080/experiments/nogan_no/epoch=95.ckpt learning_rate=1e-4 early_stopping_patience=9999"

  # 2  – Spectral-Norm einschalten
  "use_spectral_norm=1 resume_from=/mnt/lustre/work/butz/bst080/experiments/spectral_norm/epoch=95.ckpt"

  # 3  – keine Denoise-Vorverarbeitung
  "denoise_factor=0.0 resume_from=/mnt/lustre/work/butz/bst080/experiments/denoise0/epoch=95.ckpt"

  # 4  – Discriminator erst ab Epoche 15
  "warmup_disc_epochs=15 resume_from=/mnt/lustre/work/butz/bst080/experiments/warm15/epoch=95.ckpt"

  # 5  – Discriminator erst ab Epoche 5
  "warmup_disc_epochs=5 resume_from=/mnt/lustre/work/butz/bst080/experiments/warm05/epoch=95.ckpt"
)
# --------------------------------------------------------------

RUN_NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
CFG_STRING=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Logs
mkdir -p ablations_epoch
LOG_BASE=ablations_epoch/${RUN_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

# Arbeits­verzeichnis
export working_dir=/mnt/lustre/work/butz/bst080/experiments/${RUN_NAME}_${SLURM_JOB_ID}
mkdir -p "$working_dir"

# alle Key-Value-Paare in die Umgebungs­variablen kippen
for KV in $CFG_STRING ; do
  IFS='=' read VAR VAL <<< "$KV"
  export "$VAR"="$VAL"
done

echo "=== $RUN_NAME ==="
echo "working_dir : $working_dir"
echo "env settings: $CFG_STRING"
echo "----------------"

srun python /mnt/lustre/work/butz/bst080/faceGANtts/train.py \
     --log_dir "$working_dir" \
     --resume_from "$resume_from"
