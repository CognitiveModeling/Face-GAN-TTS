import sys
from pathlib import Path
import os
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# 2) Standard‑Imports                                                            
# ----------------------------------------------------------------------------‐
from config import ex  # Sacred‑Experiment mit allen Hyper‑Parametern

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa

from utils.mel_spectrogram import mel_spectrogram
from evaluation.eval import denoise_and_fadeout_reference

# -----------------------------------------------------------------------------
# 3) Pfade & Parameter                                                           
# -----------------------------------------------------------------------------
DEFAULT_INFER_ROOT = "/mnt/lustre/work/butz/bst080/faceGANtts/inference_1538393"
INFERENCE_ROOT = Path(os.environ.get("FACEGAN_INFER_ROOT", DEFAULT_INFER_ROOT))
TARGET_WAV_REL  = Path("spk1019/00014.wav")  # Unterpfad innerhalb jeder Epoche

# -------- Auswahl der Epochen -------------------------------------------------
SPECIFIC_EPOCHS = [0, 12, 24, 30, 55, 76, 96, 134, 156, 180]  # <- hier anpassen
EPOCH_INTERVAL  = 20   # Fallback, falls SPECIFIC_EPOCHS leer ist
# -----------------------------------------------------------------------------

OUT_DIR       = Path("./plots/epoch_mels")
OUT_PDF_NAME  = "mel_specs_selected_epochs_spk1019_00014.pdf"  # None → Einzel‑PDFs pro Epoche
INCLUDE_GT    = True  # Ground‑Truth ebenfalls anzeigen

# -----------------------------------------------------------------------------
# 4) Plot‑Style                                                                  
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 5) Hilfsfunktionen                                                             
# -----------------------------------------------------------------------------

def compute_mel(wav_path: Path, cfg: dict, apply_denoise: bool = False):
    """Berechnet das Log‑Mel‑Spektrum als (n_mels, T)‑Array."""

    sr       = cfg["sample_rate"]
    n_fft    = cfg["n_fft"]
    hop_size = cfg["hop_len"]
    win_size = cfg["win_len"]
    n_mels   = cfg["n_mels"]
    fmin     = cfg["f_min"]
    fmax     = cfg["f_max"]

    wav_np, _ = librosa.load(str(wav_path), sr=sr)
    wav_np    = librosa.util.normalize(wav_np)

    if apply_denoise:
        wav_np = denoise_and_fadeout_reference(wav_np, sr, cfg)

    wav_torch = torch.from_numpy(wav_np).unsqueeze(0).float()
    with torch.no_grad():
        mel = (
            mel_spectrogram(
                wav_torch, n_fft, n_mels, sr, hop_size, win_size, fmin, fmax, center=False
            ).cpu().numpy()
        )
    return np.squeeze(mel, axis=0)


def find_epoch_base(root: Path) -> Path:
    """Gibt das Verzeichnis zurück, das die epoch‑Ordner enthält."""

    if any((root / p).is_dir() and p.startswith("epoch_") for p in os.listdir(root)):
        return root
    for child in root.iterdir():
        if child.is_dir() and any((child / p).is_dir() and p.startswith("epoch_") for p in os.listdir(child)):
            return child
    raise RuntimeError(
        f"Kein Verzeichnis mit epoch_*‑Ordnern unter {root} gefunden. "
        "Setze ggf. FACEGAN_INFER_ROOT."
    )


def epoch_number(dir_path: Path) -> int:
    """Extrahiert die Nummer aus 'epoch_012_step_345'. Fehler → math.inf."""
    try:
        return int(dir_path.name.split("_")[1])
    except Exception:
        return math.inf


def discover_epoch_dirs(root: Path, specific_epochs: list[int], interval: int):
    """Gibt sortierte epoch‑Ordner gemäß *specific_epochs* oder *interval* zurück."""

    all_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
    if specific_epochs:
        whitelist = set(specific_epochs)
        sel = [d for d in all_dirs if epoch_number(d) in whitelist]
    else:
        sel = [d for d in all_dirs if epoch_number(d) % interval == 0]
    return sorted(sel, key=epoch_number)


from matplotlib import gridspec

from matplotlib import gridspec

def plot_grid(specs, titles, cfg, out_path: Path):
    n = len(specs)
    ncols = 2
    nrows = math.ceil(n / ncols)
    hop_s = cfg["hop_len"] / cfg["sample_rate"]

    fig = plt.figure(figsize=(16, 4.7 * nrows))
    gs = gridspec.GridSpec(nrows, ncols + 1, 
                           width_ratios=[1] * ncols + [0.07], 
                           wspace=0.1, hspace=0.5)  

    axes = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]

    for idx, (ax, S, title) in enumerate(zip(axes, specs, titles)):
        t = np.arange(S.shape[1]) * hop_s
        im = ax.imshow(
            S, origin='lower', aspect='auto',
            extent=[t[0], t[-1], 0, S.shape[0]],
            cmap='viridis'
        )
        ax.set_title(title, pad=10)
        ax.set_xlabel("Time (s)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

        if idx % ncols == 0:
            ax.set_ylabel("Mel bin")
            ax.set_yticks(np.arange(0, S.shape[0] + 1, 25)) 
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

    for ax in axes[n:]:
        ax.axis('off')

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(im, cax=cax, format="%+2.0f dB")
    cbar.set_label("Amplitude (dB)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------------------
# 6) Sacred‑Entry‑Point                                                         
# -----------------------------------------------------------------------------

@ex.automain
def main(_config):
    epoch_base = find_epoch_base(INFERENCE_ROOT)
    epoch_dirs = discover_epoch_dirs(epoch_base, SPECIFIC_EPOCHS, EPOCH_INTERVAL)
    if not epoch_dirs:
        raise RuntimeError("Keine passenden epoch_‑Ordner gefunden. Prüfe SPECIFIC_EPOCHS oder Pfad.")

    specs, titles = [], []

    # 1) Ground‑Truth hinzufügen (optional)
    if INCLUDE_GT:
        gt_root = Path(
            _config.get(
                "ground_truth_dir",
                "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/wav/test",
            )
        )
        gt_wav = gt_root / TARGET_WAV_REL
        if gt_wav.exists():
            specs.append(compute_mel(gt_wav, _config, apply_denoise=True))
            titles.append("Ground Truth")
        else:
            print(f"[WARN] Ground‑Truth file nicht gefunden: {gt_wav}")

    # 2) Durch Epochen iterieren
    for d in epoch_dirs:
        wav_path = d / TARGET_WAV_REL
        if not wav_path.exists():
            print(f"[WARN] {wav_path} fehlt – übersprungen")
            continue
        specs.append(compute_mel(wav_path, _config))
        titles.append(f"Epoch {epoch_number(d):03d}")

    if not specs:
        raise RuntimeError("Keine Spektrogramme erzeugt – Abbruch.")

    # 3) Ausgabe
    if OUT_PDF_NAME is None:
        for S, title in zip(specs, titles):
            out_file = OUT_DIR / f"{title.replace(' ', '_').lower()}.pdf"
            plot_grid([S], [title], _config, out_file)
    else:
        out_file = OUT_DIR / OUT_PDF_NAME
        plot_grid(specs, titles, _config, out_file)
        print(f"[INFO] Spektrogramm‑Grid in {out_file} gespeichert.")
