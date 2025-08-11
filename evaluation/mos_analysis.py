import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import textwrap
import pathlib
import pingouin as pg
# -------------------------------------------------------------------------
# 0) MultiIndex-Header einlesen, damit die Q-Codes (Level 0) und Beschriftungen
#    (Level 1) getrennt vorliegen
# -------------------------------------------------------------------------
FILE_PATH = r"C:\Users\debor\OneDrive\Desktop\mos study\data_test479469_2025-07-23_09-51.xlsx"
df = pd.read_excel(FILE_PATH, header=[0,1])

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# print(df)
# -------------------------------------------------------------------------
# 1) MOS‑Daten extrahieren
# -------------------------------------------------------------------------
nat_df = df.xs('MOS-naturalness', axis=1, level=1, drop_level=False)
scr_df = df.xs('MOS-scratchiness', axis=1, level=1, drop_level=False)

# Level‑0 (z.B. "Q101", "Q102", ...) extrahieren
nat_df.columns = nat_df.columns.get_level_values(0)
scr_df.columns = scr_df.columns.get_level_values(0)

df_mos = pd.concat([nat_df, scr_df], axis=1)

mos_long = df_mos.melt(var_name="QID", value_name="Rating")
mos_long["ID"]   = mos_long["QID"].str[2:]  # z.B. Q104 → "04"
mos_long["Type"] = mos_long["QID"].str.startswith("Q2") \
                   .map({True:"Scratchiness", False:"Quality"})

id_to_group = {
    **{f"{i:02}": ("Face-TTS LRS2"  if i%2 else "Face-GAN-TTS LRS2") for i in range(1, 11)},
    **{f"{i:02}": ("Face-TTS CFD"   if i%2 else "Face-GAN-TTS CFD")  for i in range(11, 21)}
}
mos_long["Group"] = mos_long["ID"].map(id_to_group)
mos_long.dropna(subset=["Group","Rating"], inplace=True)

# -------------------------------------------------------------------------
# 2) Deskriptive Statistik (Mittelwert und Standardabweichung)
# -------------------------------------------------------------------------
desc_stats = mos_long.groupby(["Group", "Type"])["Rating"].agg(["mean", "std"]).round(2)
print("\n=== Deskriptive MOS-Statistik (Mittelwert ± SD) ===")
for (group, typ), row in desc_stats.iterrows():
    print(f"{typ:12} | {group:20} →  M = {row['mean']:.2f}, SD = {row['std']:.2f}")

# Als Pivot-Tabelle für CSV-Speicherung
summary = desc_stats.reset_index().pivot(index="Group", columns="Type", values="mean")
stds    = desc_stats.reset_index().pivot(index="Group", columns="Type", values="std")
summary.columns.name = None
stds.columns.name = None

# Zusammenführen (optional, für manuelle Kontrolle in Excel)
combined = summary.copy()
for col in summary.columns:
    combined[col] = summary[col].round(2).astype(str) + " ± " + stds[col].round(2).astype(str)

# Speichern (optional)
summary.to_csv("mos_means.csv")
stds.to_csv("mos_stds.csv")
combined.to_csv("mos_means_and_stds_combined.csv")


# -------------------------------------------------------------------------
# 3) Wilcoxon + Bonferroni pro Merkmal
# -------------------------------------------------------------------------
def w_test(df, g1, g2, metric):
    a = df[(df["Group"]==g1)&(df["Type"]==metric)]["Rating"].values
    b = df[(df["Group"]==g2)&(df["Type"]==metric)]["Rating"].values
    return wilcoxon(a,b) if len(a)==len(b) else (np.nan,np.nan)

pairs = [(f"Face-TTS {ds}", f"Face-GAN-TTS {ds}") for ds in ["LRS2","CFD"]] + \
        [(f"{m} CFD", f"{m} LRS2") for m in ["Face-TTS","Face-GAN-TTS"]]

res = []
for metric in ["Quality", "Scratchiness"]:
    for g1, g2 in pairs:
        # Bedingungen vertauschen: jetzt g2 zuerst, g1 als Vergleich
        a = mos_long.loc[(mos_long["Group"] == g2) & (mos_long["Type"] == metric), "Rating"]
        b = mos_long.loc[(mos_long["Group"] == g1) & (mos_long["Type"] == metric), "Rating"]
        
        # Wilcoxon-Test (Pingouin liefert W, p)
        out = pg.wilcoxon(a, b)

        W     = float(out["W-val"].iloc[0])
        p_raw = float(out["p-val"].iloc[0])
        r     = float(out["RBC"].iloc[0])

        # z-Wert per Normalapproximation
        n      = len(a)
        mean_W = n * (n + 1) / 4
        std_W  = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z      = (W - mean_W) / std_W

        res.append({
            "Metric": metric,
            "G1":      g2,       
            "G2":      g1,
            "W":       W,
            "z":       z,
            "r":       r,
            "p_raw":   p_raw
        })
res_df = pd.DataFrame(res)

# Bonferroni-Korrektur (k=4 je Merkmal)
res_df["p_adj"] = res_df.groupby("Metric")["p_raw"] \
                       .transform(lambda p: (p * 4).clip(upper=1))

# Ausgabe
print("\n=== Wilcoxon Signed-Rank (Bedingungen vertauscht) ===")
for _, r in res_df.iterrows():
    print(
        f"{r.Metric:12}: {r.G1:18} vs {r.G2:18} → "
        f"W={r.W:5.0f}, z={r.z:0.2f}, r={r.r:0.2f}, "
        f"p_raw={r.p_raw:0.4f}, p_adj={r.p_adj:0.4f}"
    )


# -------------------------------------------------------------------------
# 4) Boxplots 
# -------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 22
})
OUT = r"C:\Users\debor\OneDrive\Desktop\face-GAN-TTS\plots"
os.makedirs(OUT, exist_ok=True)

def mos_box(df, typ, fname):
    plt.figure(figsize=(12,6))
    sns.boxplot(
        data=df[df["Type"]==typ],
        x="Group", y="Rating",
        hue="Group",           
        palette="Set2",
        legend=False,          
        fliersize=0, linewidth=1.6,
        boxprops=dict(alpha=.8)
    )
    plt.ylabel("MOS Score")
    plt.title(f"MOS ({typ})")
    plt.ylim(1,5)
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, ls="--", lw=.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=300)
    plt.close()

mos_box(mos_long, "Quality",    "mos_quality.pdf")
mos_box(mos_long, "Scratchiness","mos_scratchiness.pdf")

# -------------------------------------------------------------------------
# 5) Demografie (via xs auf Level=1, statt df[col])
# -------------------------------------------------------------------------
DEM_VARS = {
    "gender": {
        1: "Female",
        2: "Male",
        3: "Other / prefer not to say"
    },
    "Age": {
        1: "under 20 yrs",
        2: "20 – 29 yrs",
        3: "30 – 39 yrs",
        4: "40 – 49 yrs",
        5: "50 yrs or older"
    },
    "native speaker or not": {
        1: "Beginner",
        2: "Intermediate",
        3: "Advanced",
        4: "Native / near-native"
    }
}

print("\n=== Respondent demographics ===")
rows = []
for col, mapping in DEM_VARS.items():
    # cross-section auf Level-1 (Beschriftung) holen Series
    dem_series = df.xs(col, axis=1, level=1, drop_level=True).squeeze()
    counts     = dem_series.value_counts().sort_index()
    total      = counts.sum()
    for code, label in mapping.items():
        n   = counts.get(code, 0)
        pct = 100 * n / total if total else 0
        rows.append([col, label, n, f"{pct:0.1f}%"])
        print(f"{col:<27} {label:<28} n={n:3d}  ({pct:4.1f} %)")
        

# -------------------------------------------------------------------------
# 4) Boxplots für MOS-Ratings nach Typ und Gruppe
# -------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 22
})
OUT = r"C:\Users\debor\OneDrive\Desktop\face-GAN-TTS\plots"
os.makedirs(OUT, exist_ok=True)

def mos_box(df, typ, fname):
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df[df["Type"] == typ],
        x="Group",
        y="Rating",
        hue="Group",
        palette="Set2",
        legend=False,
        fliersize=0,
        linewidth=1.6,
        boxprops=dict(alpha=.8)
    )
    plt.ylabel("MOS Score")
    plt.title(f"MOS ({typ})")
    plt.ylim(1, 5)
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, ls="--", lw=.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=300)
    plt.close()

# Boxplots generieren
mos_box(mos_long, "Quality", "mos_quality.pdf")
mos_box(mos_long, "Scratchiness", "mos_scratchiness.pdf")

import sys
from importlib.metadata import version, PackageNotFoundError

def print_versions():
    # Interpreter-Version
    print(f"Python version: {sys.version.split()[0]}")
    for pkg in ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "pingouin"]:
        try:
            ver = version(pkg)
        except PackageNotFoundError:
            ver = "not installed"
        print(f"{pkg} version: {ver}")

if __name__ == "__main__":
    print_versions()