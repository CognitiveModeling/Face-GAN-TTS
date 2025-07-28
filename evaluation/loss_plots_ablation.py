import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

# ------------------------------------------------------------------
# 1) LaTeX / Computer-Modern fonts
# ------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
})

# ------------------------------------------------------------------
# 2) Paths
# ------------------------------------------------------------------
BASE_DIR  = r"C:\Users\debor\OneDrive\Desktop\face-GAN-TTS"
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 3) Epoch / Step to mark
# ------------------------------------------------------------------
EPOCH_CSV    = os.path.join(BASE_DIR, "run-version_1538393-tag-epoch.csv")
MARKED_EPOCH = 96
MARKED_STEP  = 17848

# ------------------------------------------------------------------
# 4) Metrics to plot (Train vs. Val)
# ------------------------------------------------------------------
metrics = [
    ("run-version_1538393-tag-train_g_loss_step.csv",
     "run-version_1538393-tag-val_total_loss.csv",
     "Generator loss",
     "Generator loss"),

    ("run-version_1538393-tag-train_adv_loss_step.csv",
     "run-version_1538393-tag-val_adv_loss.csv",
     "Adversarial loss",
     "Adversarial loss"),

    ("run-version_1538393-tag-train_d_loss_step.csv",
     None,
     "Discriminator loss",
     "Discriminator loss"),

    ("run-version_1538393-tag-train_disc_acc_step.csv",
     None,
     "Disc. accuracy",
     "Discriminator accuracy"),

     ("run-version_1538393-tag-train_diffusion_loss_step.csv",
     "run-version_1538393-tag-val_diffusion_loss.csv",
     "Diffusion loss",
     "Diffusion loss"),

    ("run-version_1538393-tag-train_duration_loss_step.csv",
     "run-version_1538393-tag-val_duration_loss.csv",
     "Duration loss",
     "Duration loss"),

    ("run-version_1538393-tag-train_prior_loss_step.csv",
     "run-version_1538393-tag-val_prior_loss.csv",
     "Prior loss",
     "Prior loss"),

    ("run-version_1538393-tag-train_spk_loss_step.csv",
     "run-version_1538393-tag-val_spk_loss.csv",
     "Speaker loss",
     "Speaker loss"),
]

# ------------------------------------------------------------------
# 4a) Y‑Axis exceptions: plots that should NOT start at 0
# ------------------------------------------------------------------
NO_ZERO_YLIM = {"Adversarial loss"}

# ------------------------------------------------------------------
# 5) Build Step → Epoch lookup
# ------------------------------------------------------------------
epoch_map = (
    pd.read_csv(EPOCH_CSV)
      .groupby("Step")["Value"].max()
      .rename("Epoch")
      .reset_index()
)

def load_and_agg(path, epoch_map):
    df = pd.read_csv(path).merge(epoch_map, on="Step", how="left")
    ep = df.groupby("Epoch")["Value"].mean().reset_index()
    return ep

# ------------------------------------------------------------------
# 6) Zwei Figures á 4 Subplots erzeugen
# ------------------------------------------------------------------
# Chunk metrics in Gruppen zu je 4
chunks = [metrics[i:i+4] for i in range(0, len(metrics), 4)]

for fig_idx, chunk in enumerate(chunks, start=1):
    # figsize: doppelte Breite (2×10), doppelte Höhe (2×4)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (train_f, val_f, ylabel, title) in zip(axes, chunk):
        # train
        train_ep = load_and_agg(os.path.join(BASE_DIR, train_f), epoch_map)
        ax.plot(train_ep["Epoch"], train_ep["Value"], label="Train", linewidth=2)

        # val (falls vorhanden)
        if val_f:
            val_path = os.path.join(BASE_DIR, val_f)
            if os.path.exists(val_path):
                val_ep = load_and_agg(val_path, epoch_map)
                ax.plot(val_ep["Epoch"], val_ep["Value"], label="Validation", linewidth=2)

        # Markierung
        ax.axvline(MARKED_EPOCH, color="red", linestyle=":", linewidth=1,
                   label=f"Epoch {MARKED_EPOCH}")

        # Label und Titel
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=10)

        # Grid, volle Zahlen und mehr Ticks
        ax.ticklabel_format(style="plain", axis="y")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(-3, 3))

        # --- Y‑Achse bei 0 starten (außer Ausnahmen) ---
        if ylabel not in NO_ZERO_YLIM:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(bottom=0, top=ymax)

        ax.legend()

    # Leeren Axis (falls weniger als 4 in der letzten Gruppe)
    if len(chunk) < 4:
        for ax in axes[len(chunk):]:
            ax.axis("off")

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"train_val_subplot_{fig_idx}.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"✓ saved {out_path}")

print("Alle Subplot-Figuren erzeugt in:", PLOTS_DIR)

# ------------------------------------------------------------------
# Eval‑Metriken
# ------------------------------------------------------------------
EVAL_FILES   = {
    "Composite Metric":        "run-version_1538393-tag-eval_composite_epoch.csv",
    "F0 RMSE":                 "run-version_1538393-tag-eval_f0_rmse_epoch.csv",
    "Mel-Cepstral Distortion": "run-version_1538393-tag-eval_mcd_epoch.csv",
    "Speaker Similarity":      "run-version_1538393-tag-eval_speaker_sim_epoch.csv",
    "Log-Spectral Distance":   "run-version_1538393-tag-eval_stft_epoch.csv",
}
MARKED_EPOCH = 96

# ------------------------------------------------------------------
# Split in first 4 und restliche 1
# ------------------------------------------------------------------

eval_items = list(EVAL_FILES.items())
first_four = eval_items[:4]
rest       = eval_items[4:]

# ------------------------------------------------------------------
# 1) 2×2-Subplot für die ersten vier
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax, (metric_name, filename) in zip(axes, first_four):
    df  = pd.read_csv(os.path.join(BASE_DIR, filename))
    agg = df.groupby("Step")["Value"].mean()
    x, y = agg.index, agg.values

    ax.plot(x, y, marker='o', linewidth=2)
    ax.axvline(MARKED_EPOCH, color='red', linestyle='--', linewidth=1,
               label=f"Epoch {MARKED_EPOCH}")

    ax.set_xlabel("Epoche")
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name, pad=10)

    # wissenschaftliche Notation für sehr kleine Werte
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(-3, 3))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend()

fig.tight_layout()
out1 = os.path.join(PLOTS_DIR, "eval_subplot_1.pdf")
fig.savefig(out1)
plt.close(fig)
print(f"✓ saved {out1}")

# ------------------------------------------------------------------
# 2) Einzelfigur für den Rest (letzte Metrik)
# ------------------------------------------------------------------
for metric_name, filename in rest:
    df  = pd.read_csv(os.path.join(BASE_DIR, filename))
    agg = df.groupby("Step")["Value"].mean()
    x, y = agg.index, agg.values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, marker='o', linewidth=2)
    ax.axvline(MARKED_EPOCH, color='red', linestyle='--', linewidth=1,
               label=f"Epoch {MARKED_EPOCH}")

    ax.set_xlabel("Epoche")
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name, pad=10)

    ax.ticklabel_format(style="scientific", axis="y", scilimits=(-3, 3))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.legend()
    fig.tight_layout()

    out2 = os.path.join(PLOTS_DIR, f"eval_single_{metric_name.replace(' ', '_').lower()}.pdf")
    fig.savefig(out2)
    plt.close(fig)
    print(f"✓ saved {out2}")

print("Alle Eval-Figuren erzeugt in:", PLOTS_DIR)
