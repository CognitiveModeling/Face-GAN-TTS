import os
import torch
import numpy as np
import pytorch_lightning as pl
import torchaudio
from tqdm import tqdm
from fastdtw import fastdtw  
from scipy.spatial.distance import cosine, euclidean
import types
import librosa
import noisereduce as nr

from model.syncnet_hifigan import SyncNet
from utils.mel_spectrogram import mel_spectrogram

from mel_cepstral_distance import compare_audio_files #https://github.com/stefantaubert/mel-cepstral-distance/blob/main/README.md
from nnAudio.features import STFT #https://github.com/KinWaiCheuk/nnAudio
from discrete_speech_metrics import LogF0RMSE #https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics
#from discrete_speech_metrics import PESQ
import discrete_speech_metrics.logf0rmse as logf0mod
from discrete_speech_metrics import UTMOS

from config import ex


def compute_speaker_similarity(syncnet_model, ref_mel, syn_mel, config):
    """
    Compute speaker similarity as cosine similarity between SyncNet audio embeddings.
    """

    # Forward through SyncNet
    emb_ref_seq = syncnet_model.forward_aud(ref_mel.unsqueeze(1).cuda())  # (1, emb_dim, T)
    emb_syn_seq = syncnet_model.forward_aud(syn_mel.unsqueeze(1).cuda())

    # Mean pool across time (dim=2)
    emb_ref = emb_ref_seq.mean(dim=2).squeeze(0).cpu().numpy()
    emb_syn = emb_syn_seq.mean(dim=2).squeeze(0).cpu().numpy()

    # Normalize and cosine similarity
    emb_ref /= np.linalg.norm(emb_ref) + 1e-8
    emb_syn /= np.linalg.norm(emb_syn) + 1e-8
    cos_sim = np.dot(emb_ref, emb_syn)
    cos_distance = 1.0 - cos_sim
    return cos_distance, cos_sim

def normalize_audio(wav):
        return librosa.util.normalize(wav)

def _patched_score(self, gt_audio, gen_audio):
    import numpy as np
    import pyworld
    import pysptk
    import librosa

    def world_extract(x, sr):
        x = x.astype(np.float64)
        f0, timeaxis = pyworld.dio(x, sr)
        f0 = pyworld.stonemask(x, f0, timeaxis, sr)
        sp = pyworld.cheaptrick(x, f0, timeaxis, sr)
        ap = pyworld.d4c(x, f0, timeaxis, sr)
        mcep = pysptk.sp2mc(sp, order=24, alpha=0.42)
        return mcep, f0

    gt_audio = librosa.resample(gt_audio, orig_sr=self.sr, target_sr=16000)
    gen_audio = librosa.resample(gen_audio, orig_sr=self.sr, target_sr=16000)

    gt_mcep, gt_f0 = world_extract(gt_audio, 16000)
    gen_mcep, gen_f0 = world_extract(gen_audio, 16000)

    _, path = fastdtw(gen_mcep, gt_mcep, dist=euclidean)  # <- gepatcht!
    gen_f0_aligned = np.array([gen_f0[i] for i, _ in path])
    gt_f0_aligned = np.array([gt_f0[j] for _, j in path])
    nonzero_indices = np.logical_and(gt_f0_aligned > 0, gen_f0_aligned > 0)

    if np.sum(nonzero_indices) == 0:
        return 0.0
    gt_f0_log = np.log(gt_f0_aligned[nonzero_indices])
    gen_f0_log = np.log(gen_f0_aligned[nonzero_indices])
    return np.sqrt(np.mean((gt_f0_log - gen_f0_log) ** 2))

def compute_lsd_classic(ref, gen, stft_transform):
                ref_spec = stft_transform(torch.tensor(ref).unsqueeze(0)).abs()[0]  # [freq, time]
                gen_spec = stft_transform(torch.tensor(gen).unsqueeze(0)).abs()[0]

                ref_log = torch.log10(ref_spec + 1e-8)
                gen_log = torch.log10(gen_spec + 1e-8)

                min_frames = min(ref_log.shape[1], gen_log.shape[1])
                ref_log = ref_log[:, :min_frames]
                gen_log = gen_log[:, :min_frames]

                # RMSE pro Frame (dim=0 = freq)
                frame_errors = torch.sqrt(torch.mean((ref_log - gen_log) ** 2, dim=0))
                lsd = torch.mean(frame_errors).item()
                return lsd

def find_wav_files(root_dir):
        return sorted([
            os.path.join(root, f) for root, _, files in os.walk(root_dir)
            for f in files if f.endswith(".wav")
        ])

def denoise_and_fadeout_reference(wav, sr, config):
    # Apply denoising (same as during training)
    wav_denoised = nr.reduce_noise(
        y=wav,
        sr=sr,
        stationary=True,
        prop_decrease=config["denoise_factor"],
        n_fft=config["n_fft"],
        win_length=config["win_len"],
        hop_length=config["hop_len"]
    )

    # Apply 50ms linear fade-out
    fade_len = int(0.05 * sr)
    if len(wav_denoised) >= fade_len:
        fade = np.linspace(1, 0, fade_len)
        wav_denoised[-fade_len:] *= fade

    return wav_denoised

@ex.automain
def main(_config):
    pl.seed_everything(_config["seed"])
    print("[INFO] Starting evaluation", flush=True)

    reference_audio_dir = _config.get("ground_truth_dir")
    syncnet = SyncNet(_config).cuda().eval()
    use_gan = _config["use_gan"]

    
    eval_output_dir = os.getenv("DYNAMIC_EVAL_PATH", None)
    generated_audio_dir = eval_output_dir if eval_output_dir is not None else _config.get("output_dir_gan" if use_gan else "output_dir_orig")

    generated_wavs = find_wav_files(generated_audio_dir)
    reference_wavs = find_wav_files(reference_audio_dir)
    
    ref_dict = {
        os.path.relpath(p, reference_audio_dir): p
        for p in reference_wavs
    }
    gen_dict = {
        os.path.relpath(p, generated_audio_dir): p
        for p in generated_wavs
    }

    common = sorted(set(ref_dict.keys()) & set(gen_dict.keys()))

    reference_wavs = [ref_dict[k] for k in common]
    generated_wavs = [gen_dict[k] for k in common]

    speaker_similarities, f0_errors, mcd_values, stft_distances, utmos_list = [], [], [], [], []

    # Evaluation parameters
    sample_rate = _config["sample_rate"]
    num_mels = _config["n_mels"]
    n_fft = _config["n_fft"]
    hop_length = _config["hop_len"]
    win_length = _config["win_len"]
    f_min = _config["f_min"]
    f_max = _config["f_max"]

    stft_transform = STFT(sample_rate, n_fft, hop_length, win_length)
    with torch.no_grad():
        for idx, (gen_wav_path, ref_wav_path) in tqdm(enumerate(zip(generated_wavs, reference_wavs)), total=len(generated_wavs), desc="Evaluating"):

            # Convert waveforms to log-mel spectrograms (shape: [mel_bins, time_frames])
            gen_torch, sr_gen = torchaudio.load(gen_wav_path)
            ref_torch, sr_ref = torchaudio.load(ref_wav_path)

            gen_norm_audio = normalize_audio(gen_torch[0].numpy())
            ref_norm_audio = normalize_audio(ref_torch[0].numpy())

            # Denoise referenece audio - Train data already denoised
            ref_audio_denoise = denoise_and_fadeout_reference(ref_norm_audio, sample_rate, _config)

            gen_audio = torch.tensor(gen_norm_audio).unsqueeze(0)
            ref_audio = torch.tensor(ref_audio_denoise).unsqueeze(0)

            ref_mel = mel_spectrogram(ref_audio, n_fft, num_mels, sample_rate,
                                            hop_length, win_length, f_min, f_max, center=False)
            syn_mel = mel_spectrogram(gen_audio, n_fft, num_mels, sample_rate,
                                            hop_length, win_length, f_min, f_max, center=False)

            gen_np = gen_audio[0].numpy()
            ref_np = ref_audio[0].numpy()

            # Speaker similarity
            cos_dist, cos_sim = compute_speaker_similarity(syncnet, ref_mel, syn_mel, _config)
            speaker_similarities.append(cos_sim)

            # F0 RMSE
            metrics = LogF0RMSE(sr=sample_rate)
            metrics.score = types.MethodType(_patched_score, metrics)
            f0_rmse = metrics.score(ref_np, gen_np)
            f0_errors.append(f0_rmse)

            min_len = min(len(ref_np), len(gen_np))
            ref_aligned = ref_np[:min_len]
            gen_aligned = gen_np[:min_len]

            # # PESQ
            # pesq_metric = PESQ(sr=sample_rate)
            # pesq = pesq_metric.score(ref_aligned, gen_aligned)
            # pesq_list.append(pesq)

            # # UTMOS
            metrics = UTMOS(sr=sample_rate)
            utmos = metrics.score(gen_aligned)
            utmos_list.append(utmos)

            # Mel Cepstral Distortion (MCD)
            mcd, _ = compare_audio_files(ref_wav_path, gen_wav_path)
            mcd_values.append(mcd)

            # Log-spectral distance (STFT)
            stft_dist = compute_lsd_classic(ref_np, gen_np, stft_transform)
            stft_distances.append(stft_dist)
  
    # Compute means
    mean_speaker_similarity = np.mean(speaker_similarities)
    mean_f0_error = np.mean(f0_errors)
    mean_mcd = np.mean(mcd_values)
    mean_stft_distance = np.mean(stft_distances)
    #mean_pesq = np.mean(pesq_list)
    mean_utmos =np.mean(utmos_list)

    # Normalize each metric to [0,1]
    # (a) Speaker similarity: cos_sim ∈ [0,1] → error ∈ [0,1]
    norm_speaker = 1.0 - mean_speaker_similarity
    # (b) F0-RMSE: assume worst-case ~1.0 Nats → clip above 1.0
    f0_max = 1.0
    norm_f0 = mean_f0_error / f0_max
    if norm_f0 > 1.0:
        norm_f0 = 1.0

    # (c) MCD: we know MCD ∈ [4,12] → normalize to [0,1]
    norm_mcd = (mean_mcd - 4.0) / 8.0
    if norm_mcd < 0.0:
        norm_mcd = 0.0
    if norm_mcd > 1.0:
        norm_mcd = 1.0

    # (d) STFT distance: typical ∈ [0,2] → normalize to [0,1]
    norm_stft = mean_stft_distance / 2.0
    if norm_stft < 0.0:
        norm_stft = 0.0
    if norm_stft > 1.0:
        norm_stft = 1.0

    # Composite score combining relevant metrics (normalized)
    composite = (norm_speaker + norm_f0 + norm_mcd + norm_stft) / 4.0

    metrics = {
    "Composite Metric": composite,
    "Speaker Similarity": mean_speaker_similarity,
    "F0 RMSE": mean_f0_error,
    "MCD": mean_mcd,
    #"PESQ":mean_pesq,
    "UTMOS": mean_utmos,
    "STFT Distance": mean_stft_distance
    }

    # print("\n######## Evaluation Results ########")
    # for k, v in metrics.items():
    #     print(f"{k}: {v:.4f}", flush=True)

    out_dir = os.getenv("DYNAMIC_EVAL_PATH")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "eval_output.txt")
        try:
            with open(out_file, "w") as f:
                for k, v in metrics.items():
                    f.write(f"{k}: {v:.6f}\n")
            print(f"[INFO] Alle Metriken nach {out_file} geschrieben")
        except Exception as e:
            print(f"[ERROR] eval_output.txt konnte nicht geschrieben werden: {e}")