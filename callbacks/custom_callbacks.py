import os
import torch
import pytorch_lightning as pl
import subprocess
import shutil
import time
import re
import torchaudio
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

def log_audio_and_metrics(trainer, step_dir, step):
    # TensorBoard Logging
    epoch = trainer.current_epoch 
    eval_path = os.path.join(step_dir, "eval_output.txt")
    if os.path.exists(eval_path) and hasattr(trainer, "logger"):
        with open(eval_path, "r") as f:
            content = f.read()

        def extract(name):
            match = re.search(rf"{name}:\s*([\d\.eE+-]+)", content)
            return float(match.group(1)) if match else None

        metrics = {
            "eval/composite": extract("Composite Metric"),
            "eval/speaker_sim": extract("Speaker Similarity"),
            "eval/f0_rmse": extract("F0 RMSE"),
            "eval/mcd": extract("MCD"),
            "eval/stft": extract("STFT Distance"),
            "eval/l1mel": extract("L1 Mel Distance")
        }
        tb = trainer.logger.experiment
        for key, val in metrics.items():
            if val is not None:
                tb.add_scalar(key, val, global_step=step)
                tb.add_scalar(f"{key}_epoch", val, global_step=epoch)
                #print(f"[LOGGED] {key} = {val:.4f} (step={step}, epoch={epoch})")

    # Audio Logging
    if hasattr(trainer, "logger"):
        wavs_dir = step_dir
        if os.path.exists(wavs_dir):
            for root, _, files in os.walk(wavs_dir):
                for f in sorted(files):
                    if f.endswith(".wav"):
                        sample_path = os.path.join(root, f)
                        try:
                            wav, sr = torchaudio.load(sample_path)
                            tag = f"audio_sample/{os.path.basename(root)}_{f.replace('.wav', '')}"
                            trainer.logger.experiment.add_audio(tag, wav[0], global_step=step, sample_rate=sr)
                            #print(f"[AUDIO LOGGED] {tag}")
                        except Exception as e:
                            print(f"[WARNING] Could not log audio {f}: {e}")
                        break  # only one WAV per Sample

def run_inference_and_eval(output_dir, checkpoint_path):
    os.environ["output_dir"] = os.path.abspath(output_dir)
    os.environ["DYNAMIC_EVAL_PATH"] = os.path.abspath(output_dir)
    os.environ["resume_from_checkpoint"] = checkpoint_path

    subprocess.run([
        "bash", "-c",
        f"export CUDA_VISIBLE_DEVICES=3 && "  # only GPU 3 to avoid OOM
        f"source /home/butz/bst080/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate train_env && "
        f"python inference.py"
    ])

    eval_proc = subprocess.run([
        "bash", "-c",
        "export CUDA_VISIBLE_DEVICES=3 && "  # only GPU 3 to avoid OOM
        "cd /mnt/lustre/work/butz/bst080/faceGANtts && "
        "export PYTHONPATH=$(pwd):$PYTHONPATH && "
        "source /home/butz/bst080/miniconda3/etc/profile.d/conda.sh && "
        "conda activate train_env && "
        "python evaluation/eval.py"
    ], capture_output=True, text=True)

    if eval_proc.returncode != 0:
        print(f"[ERROR] Eval subprocess failed with return code {eval_proc.returncode}")
        print(f"[STDERR] {eval_proc.stderr[:300]}")

    # Wait for eval_output.txt 
    eval_path = os.path.join(output_dir, "eval_output.txt")
    for i in range(60):  # max. 60 sec
        if os.path.exists(eval_path):
            #print(f"[INFO] eval_output.txt found after {i+1}s")
            break
        time.sleep(1)
    else:
        print("[WARNING] eval_output.txt NOT found after 60s")


def make_inference_dir(epoch: int, step: int):
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_id")
    working_dir = os.environ.get("HP_WORKING_DIR", "")  # neu: expliziten Pfad zum aktuellen run
    base_dir = working_dir if working_dir else f"inference_{slurm_job_id}"
    try:
        epoch_int = int(epoch)
        step_int = int(step)
        epoch_str = f"epoch_{epoch_int:03d}_step_{step_int}"
    except ValueError:
        # Fallback: benutze strings direkt
        epoch_str = f"{epoch}_{step}"
    path = os.path.join(base_dir, f"inference_{slurm_job_id}", epoch_str)
    os.makedirs(path, exist_ok=True)
    return path

class SaveEpochZeroCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == 0 and trainer.global_rank == 0:
            torch.cuda.empty_cache() 
            ckp_callback = trainer.checkpoint_callback
            dirpath = ckp_callback.dirpath
            ckpt_path = os.path.join(dirpath, "epoch=0.ckpt")
            os.makedirs(dirpath, exist_ok=True)
            #print(f"[INFO] Saving 'epoch=0' checkpoint to {ckpt_path}")
            torch.save({
                'epoch': trainer.current_epoch,
                'global_step': trainer.global_step,
                'state_dict': pl_module.state_dict()
            }, ckpt_path)

            step_dir = make_inference_dir(trainer.current_epoch, trainer.global_step)
            run_inference_and_eval(step_dir, ckpt_path)
            log_audio_and_metrics(trainer, step_dir, trainer.global_step)
            
class EarlyStoppingCallback(pl.callbacks.EarlyStopping):
    def __init__(self, patience: int, min_delta: float):
        super().__init__(
            monitor="val/total_loss",  
            mode="min",                
            patience=patience,               
            min_delta=min_delta,             
            verbose=True              
        )

class SaveBestCheckpointPath(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return

        best_path = trainer.checkpoint_callback.best_model_path
        if best_path and os.path.exists(best_path):
            basename = os.path.basename(best_path)
            epoch_str = "epoch_unknown"
            step_str = "step_unknown"
            for part in basename.split("-"):
                if part.startswith("epoch="):
                    epoch_str = part.replace("epoch=", "").split(".")[0]
                if part.startswith("step="):
                    step_str = part.replace("step=", "").split(".")[0]
            dest_name = f"best_epoch_{epoch_str}_step_{step_str}.ckpt"
            dest_path = os.path.join(os.path.dirname(best_path), dest_name)

            if os.path.exists(dest_path):
                os.remove(dest_path)
            shutil.copy(best_path, dest_path)

            step_dir = make_inference_dir(int(epoch_str), int(step_str))
            run_inference_and_eval(step_dir, best_path)
            log_audio_and_metrics(trainer, step_dir, trainer.global_step)

class StepwiseEvalCallback(pl.Callback):
    def __init__(self, config):
        super().__init__()
        self.eval_interval = config["eval_interval"]
        os.environ["USE_GAN"] = str(config["use_gan"])
        self.slurm_job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_id")
        self.base_inference_dir = f"inference_{self.slurm_job_id}"

    def on_validation_end(self, trainer, pl_module):
        step = trainer.global_step
        eval_int = self.eval_interval

        if trainer.global_rank == 0 and (trainer.current_epoch == 1 or (step > 0 and step % eval_int == 0)):
            torch.cuda.empty_cache()  
            ckpt_dir = trainer.checkpoint_callback.dirpath
            has_checkpoints = os.path.exists(ckpt_dir) and any(f.endswith(".ckpt") for f in os.listdir(ckpt_dir))
            if not has_checkpoints:
                print(f"[StepwiseEvalCallback] No Checkpoint found -> skipping evaluation by step={step}")
                return

            #print(f"[StepwiseEvalCallback] step={step}: Inference + Eval via Subprozess...")

            step_dir = make_inference_dir(trainer.current_epoch, step)
            run_inference_and_eval(step_dir, trainer.checkpoint_callback.last_model_path)

            log_audio_and_metrics(trainer, step_dir, step)

class CompositeBestMelCallback(pl.Callback):
    """
    - searches for the checkpoint with the lowest composite metric value among the last N saved checkpoints
    after the end of training
    - saves a Mel spectrogram comparison image (Reference vs. Generated) in the same folder for this checkpoint
    """
    def __init__(self, config, last_n: int = 10):
        super().__init__()
        self.cfg = config
        self.last_n = last_n

    # --------------------------------------------------------------
    def _read_composite(self, eval_txt: str) -> float | None:
        """Parse 'Composite Metric:' Zeile aus eval_output.txt"""
        import re
        with open(eval_txt) as f:
            m = re.search(r"Composite Metric:\s*([\-+eE0-9\.]+)", f.read())
        return float(m.group(1)) if m else None

    # --------------------------------------------------------------
    def on_fit_end(self, trainer, pl_module):
        if trainer.global_rank != 0: 
            torch.cuda.empty_cache()       
            return

        ckpt_dir = trainer.checkpoint_callback.dirpath
        ckpts = sorted(
            [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
             if f.endswith(".ckpt")],
            key=os.path.getmtime, reverse=True
        )[: self.last_n]

        best_ckpt, best_comp, best_inf_dir = None, float("inf"), None
        for ckpt in ckpts:
            step_dir = make_inference_dir("post", os.path.basename(ckpt)[:-5])
            run_inference_and_eval(step_dir, ckpt)        # create eval_output.txt & WAVs

            comp_path = os.path.join(step_dir, "eval_output.txt")
            comp_val  = self._read_composite(comp_path) if os.path.exists(comp_path) else None
            if comp_val is not None and comp_val < best_comp:
                best_ckpt, best_comp, best_inf_dir = ckpt, comp_val, step_dir

        if best_ckpt is None:
            print("[CompositeBestMel] No Composite-Metrik found – Stop")
            return
        #print(f"[CompositeBestMel] → best CKPT: {os.path.basename(best_ckpt)} "
              #f"(Composite = {best_comp:.4f})")

        # -------- Mel-Spectrogram ---------------------------------------
        sr      = self.cfg["sample_rate"]
        hop_len = self.cfg["hop_len"]
        n_fft   = self.cfg["n_fft"]
        n_mels  = self.cfg["n_mels"]

        gen_wavs = [p for p in os.listdir(best_inf_dir) if p.endswith(".wav")]
        if not gen_wavs:
            print("[CompositeBestMel] No generated WAVs found – Stop")
            return
        gen_path = os.path.join(best_inf_dir, sorted(gen_wavs)[0])

        # try identical file in Ground Truth folder, otherwise use first WAV
        ref_root = self.cfg["ground_truth_dir"]
        ref_path = os.path.join(ref_root, os.path.basename(gen_path))
        if not os.path.exists(ref_path):
            ref_wavs = sorted([p for p in os.listdir(ref_root) if p.endswith(".wav")])
            ref_path = os.path.join(ref_root, ref_wavs[0])

        def wav2mel(wav_file):
            y, _   = librosa.load(wav_file, sr=sr)
            mel    = librosa.feature.melspectrogram(
                        y=y, sr=sr, n_fft=n_fft, hop_length=hop_len,
                        n_mels=n_mels, power=1.0)
            return librosa.power_to_db(mel, ref=np.max)

        mel_ref = wav2mel(ref_path)
        mel_gen = wav2mel(gen_path)

        save_path = os.path.join(os.path.dirname(best_ckpt), "mel_comparison.png")
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        librosa.display.specshow(mel_ref, sr=sr, hop_length=hop_len,
                                 x_axis="time", y_axis="mel")
        plt.title("Reference Mel")

        plt.subplot(2, 1, 2)
        librosa.display.specshow(mel_gen, sr=sr, hop_length=hop_len,
                                 x_axis="time", y_axis="mel")
        plt.title("Generated Mel")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"[CompositeBestMel] Mel-comparison saved here {save_path}")
        
class SaveEpoch96Callback(pl.Callback):
    """Save + evaluate at the **end of epoch 96** (exact clone of SaveEpochZeroCallback)."""
    TARGET_EPOCH = 96                      # change only this line if you need another epoch

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.TARGET_EPOCH and trainer.global_rank == 0:
            torch.cuda.empty_cache()

            dirpath   = trainer.checkpoint_callback.dirpath
            ckpt_path = os.path.join(dirpath, f"epoch={self.TARGET_EPOCH:03d}.ckpt")
            trainer.save_checkpoint(ckpt_path)           # Lightning helper

            step_dir = make_inference_dir(
                trainer.current_epoch, trainer.global_step
            )
            run_inference_and_eval(step_dir, ckpt_path)
            log_audio_and_metrics(trainer, step_dir, trainer.global_step)