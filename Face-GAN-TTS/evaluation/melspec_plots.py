# plot_mels_two_figures.py
import os, torch, librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from config import ex
from utils.mel_spectrogram import mel_spectrogram
from evaluation.eval import denoise_and_fadeout_reference

# -------------------- Plotstil --------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
})

# -------------------- WAV-Pfadgruppen --------------------
GT = "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/wav/test/spk1263/00003.wav"

#  TTS-Varianten (5 Spektrogramme, 3 × 2-Grid)
TTS_WAVS   = [
    GT,
    "/mnt/lustre/work/butz/bst080/faceGANtts/test/inference_outputs/v1538404_hinge/spk1263/00003.wav",
    "/mnt/lustre/work/butz/bst080/faceGANtts/inference_1561276/inference_1561276/epoch_096_step_8924/spk1263/00003.wav",
    "/mnt/lustre/work/butz/bst080/faceGANtts/inference_1561277/inference_1561277/epoch_096_step_8924/spk1263/00003.wav",
    "/mnt/lustre/work/butz/bst080/faceGANtts/test/Only_LRS3Checkpoints/spk1263/00003.wav",
]
TTS_TITLES = [
    "(a) Ground Truth",
    "(b) Face-GAN-TTS",
    "(c) FACE-TTS Scratch",
    "(d) FACE-TTS Finetuned",
    "(e) FACE-TTS trained on LRS3",
]
TTS_PDF    = "./plots/compare_gt_gan_scratch_finetune_2x2_spk1263_00003.pdf"

#   Denoising-Stufen (4 Spektrogramme, 2 × 2-Grid)
DEN_WAVS   = [
    GT,
    "/mnt/lustre/work/butz/bst080/faceGANtts/inference_1561273/inference_1561273/epoch_096_step_17848/spk1263/00003.wav",
    "/mnt/lustre/work/butz/bst080/faceGANtts/inference_1538420/inference_1538420/epoch_096_step_17848/spk1263/00003.wav",
    "/mnt/lustre/work/butz/bst080/faceGANtts/inference_1538393/inference_1538393/epoch_096_step_17848/spk1263/00003.wav",
]
DEN_TITLES = [
    "(a) Ground Truth",
    "(b) Denoise 0.0",
    "(c) Denoise 0.2",
    "(d) Denoise 0.7",
]
DEN_PDF    = "./plots/compare_gt_denoise_levels_2x2_spk1263_00003.pdf"

os.makedirs("./plots", exist_ok=True)

# -------------------- Helper --------------------
def compute_mel(path, cfg, ref=False):
    sr, hop, n_fft, win, n_mels, fmin, fmax = (
        cfg[k] for k in ("sample_rate", "hop_len", "n_fft", "win_len", "n_mels", "f_min", "f_max")
    )
    wav, _ = librosa.load(path, sr=sr)
    wav = librosa.util.normalize(wav)
    if ref:
        wav = denoise_and_fadeout_reference(wav, sr, cfg)
    wav = torch.from_numpy(wav).unsqueeze(0).float()
    with torch.no_grad():
        mel = mel_spectrogram(wav, n_fft, n_mels, sr, hop, win, fmin, fmax, center=False)
    return mel.squeeze(0).cpu().numpy()

def plot_mels(wavs, titles, out_pdf, cfg):
    mels = [compute_mel(p, cfg, ref=(i==0)) for i, p in enumerate(wavs)]
    hop_s = cfg["hop_len"] / cfg["sample_rate"]
    times = [np.arange(m.shape[1]) * hop_s for m in mels]
    rows  = 2 if len(mels) == 4 else 3
    cols  = 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10 if rows==2 else 14),
                             sharey=True, constrained_layout=True)
    axes = axes.flatten()
    for ax, S, t, title in zip(axes, mels, times, titles):
        im = ax.imshow(S, origin="lower", aspect="auto",
                       extent=[t[0], t[-1], 0, S.shape[0]], cmap="viridis")
        ax.set_title(title, pad=10)
        ax.set_xlabel("Time (s)")
    # leere Achse bei 5 Plots
    if len(mels) == 5:
        axes[-1].axis("off")
    # Y-Beschriftung links
    for idx in range(0, len(mels), cols):
        axes[idx].set_ylabel("Mel bin")
        axes[idx].set_yticks(np.arange(0, mels[idx].shape[0]+1, 25))
    # gemeinsame Colorbar
    fig.colorbar(im, ax=axes[:len(mels)], format="%+2.0f dB").set_label("Amplitude (dB)")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

# -------------------- Sacred-Entry-Point --------------------
@ex.automain
def main(_config):
    plot_mels(TTS_WAVS, TTS_TITLES, TTS_PDF, _config)
    plot_mels(DEN_WAVS, DEN_TITLES, DEN_PDF, _config)
