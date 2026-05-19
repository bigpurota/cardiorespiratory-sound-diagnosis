"""
Exploratory data analysis figures.

Reads the manifest (``data/processed/manifest.csv``) and the lung cycle table
(``data/processed/lung_cycles.csv``) and renders the descriptive figures used in the
report: class distributions, duration histograms, heart per-database (A-E) counts, lung
per-class cycle counts, the ICBHI native sampling-rate distribution, and example
waveform + log-mel panels per class.

This module is read-only: it never modifies splits, the manifest, params, or any cache.
The matplotlib Agg backend is selected before importing pyplot so the script runs headless.
"""
from src import config  # noqa: F401 — import FIRST for the SEED=42 side effect (determinism)

import os

import matplotlib

matplotlib.use("Agg")  # headless backend, must precede the pyplot import
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import soundfile as sf  # noqa: E402
import librosa  # noqa: E402

from src.preprocess import load_resampled  # noqa: E402

__all__ = [
    "EDA_DIR",
    "load_tables",
    "plot_class_dist_heart",
    "plot_class_dist_lung",
    "plot_duration_hist_heart",
    "plot_duration_hist_lung",
    "plot_heart_per_db",
    "plot_lung_per_class",
    "plot_icbhi_native_sr",
    "plot_example_panels",
    "main",
]

# Output directory for all EDA figures.
EDA_DIR = os.path.join(config.RESULTS_DIR, "figures", "eda")

# Heart label mapping: -1 = normal, 1 = abnormal in the CinC 2016 manifest.
HEART_LABEL_NAMES = {-1.0: "normal", 1.0: "abnormal"}

# Fixed lung 4-class ordering for consistent figure colours and axes.
LUNG_CLASS_ORDER = ["normal", "crackle", "wheeze", "both"]

# Single accent colour (steel-blue, matching the "Blues" colormap used by the confusion
# matrices) so every report figure shares one visual style.
ACCENT = "#3b6ea5"


def _apply_style():
    """Apply the shared report style to matplotlib (idempotent)."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.7,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 11,
        "savefig.facecolor": "white",
    })


def _ensure_dir():
    os.makedirs(EDA_DIR, exist_ok=True)


def _save(fig, name):
    """Save *fig* as a non-empty PNG under EDA_DIR and close it."""
    _ensure_dir()
    out = os.path.join(EDA_DIR, name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def load_tables():
    """Load the manifest and lung-cycle tables (source of truth for EDA)."""
    manifest = pd.read_csv(os.path.join(config.DATA_PROCESSED, "manifest.csv"))
    cycles = pd.read_csv(os.path.join(config.DATA_PROCESSED, "lung_cycles.csv"))
    return manifest, cycles


# Class-distribution figures
def plot_class_dist_heart(manifest):
    """Heart recording class distribution (normal vs abnormal)."""
    heart = manifest[manifest["modality"] == "heart"].copy()
    heart["label_name"] = heart["label"].map(HEART_LABEL_NAMES).fillna("unknown")
    order = ["normal", "abnormal"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=heart, x="label_name", order=order, color=ACCENT, ax=ax)
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title("PhysioNet/CinC 2016 — heart recording class distribution")
    ax.set_xlabel("class")
    ax.set_ylabel("recordings")
    return _save(fig, "class_dist_heart.png")


def plot_class_dist_lung(cycles):
    """Lung cycle 4-class distribution (normal/crackle/wheeze/both)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=cycles, x="label", order=LUNG_CLASS_ORDER, color=ACCENT, ax=ax)
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title("ICBHI 2017 — lung cycle class distribution")
    ax.set_xlabel("cycle class")
    ax.set_ylabel("cycles")
    return _save(fig, "class_dist_lung.png")


# Duration histograms
def plot_duration_hist_heart(manifest):
    """Heart recording duration histogram (seconds)."""
    heart = manifest[manifest["modality"] == "heart"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(heart["duration_s"].dropna(), bins=40, color=ACCENT, ax=ax)
    ax.set_title("Heart (CinC 2016) — recording duration distribution")
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("recordings")
    return _save(fig, "duration_hist_heart.png")


def plot_duration_hist_lung(manifest):
    """Lung recording duration histogram (seconds)."""
    lung = manifest[manifest["modality"] == "lung"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(lung["duration_s"].dropna(), bins=40, color=ACCENT, ax=ax)
    ax.set_title("Lung (ICBHI 2017) — recording duration distribution")
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("recordings")
    return _save(fig, "duration_hist_lung.png")


# Per-database / per-class breakdowns
def plot_heart_per_db(manifest):
    """Heart per-database (A–E) recording counts."""
    heart = manifest[manifest["modality"] == "heart"].copy()
    heart["db_source"] = heart["db_source"].astype(str).str.lower()
    order = sorted(heart["db_source"].dropna().unique())
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=heart, x="db_source", order=order, color=ACCENT, ax=ax)
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_title("Heart (CinC 2016) — recordings per source database (A–E)")
    ax.set_xlabel("database")
    ax.set_ylabel("recordings")
    return _save(fig, "heart_per_db_counts.png")


def plot_lung_per_class(cycles):
    """Lung per-class cycle counts (alias of class distribution, explicit name)."""
    counts = cycles["label"].value_counts().reindex(LUNG_CLASS_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, color=ACCENT, ax=ax)
    for i, v in enumerate(counts.values):
        ax.annotate(int(v), (i, v), ha="center", va="bottom")
    ax.set_title("ICBHI 2017 — lung cycles per class")
    ax.set_xlabel("cycle class")
    ax.set_ylabel("cycles")
    return _save(fig, "lung_per_class_counts.png")


# Native sampling-rate heterogeneity (ICBHI)
def plot_icbhi_native_sr(manifest):
    """Distribution of ICBHI native sampling rates (header-only via sf.info)."""
    lung = manifest[manifest["modality"] == "lung"]
    rates = []
    for path in lung["filepath"]:
        try:
            rates.append(int(sf.info(path).samplerate))
        except Exception:
            continue
    sr = pd.Series(rates, name="native_sr")
    counts = sr.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=[str(i) for i in counts.index], y=counts.values,
                color=ACCENT, ax=ax)
    for i, v in enumerate(counts.values):
        ax.annotate(int(v), (i, v), ha="center", va="bottom")
    ax.set_title("ICBHI 2017 — native sampling-rate distribution")
    ax.set_xlabel("native sample rate (Hz)")
    ax.set_ylabel("recordings")
    return _save(fig, "icbhi_native_sr_hist.png")


# Example waveform + log-mel panels
def _waveform_logmel_panel(path, sr, title, out_name):
    """Render a (waveform, log-mel) 2-panel figure for one representative file."""
    y = load_resampled(path, target_sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    logmel = librosa.power_to_db(mel, ref=np.max)

    fig, axes = plt.subplots(2, 1, figsize=(7, 5))
    t = np.arange(len(y)) / sr
    axes[0].plot(t, y, linewidth=0.6, color="#333333")
    axes[0].set_title(f"{title} — waveform")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("amplitude")
    img = librosa.display.specshow(logmel, sr=sr, x_axis="time", y_axis="mel",
                                   ax=axes[1])
    axes[1].set_title(f"{title} — log-mel spectrogram")
    axes[1].grid(False)
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    fig.tight_layout()
    return _save(fig, out_name)


def plot_example_panels(manifest, cycles):
    """One example waveform+log-mel panel per class for each modality."""
    outputs = []

    # Heart: one representative recording per class.
    heart = manifest[manifest["modality"] == "heart"]
    for label_val, label_name in HEART_LABEL_NAMES.items():
        subset = heart[heart["label"] == label_val]
        if subset.empty:
            continue
        path = subset.iloc[0]["filepath"]
        outputs.append(_waveform_logmel_panel(
            path, config.SR_HEART, f"Heart {label_name}",
            f"example_panel_heart_{label_name}.png"))

    # Lung: one representative cycle per class (full recording shown for context).
    for cls in LUNG_CLASS_ORDER:
        subset = cycles[cycles["label"] == cls]
        if subset.empty:
            continue
        path = subset.iloc[0]["filepath"]
        outputs.append(_waveform_logmel_panel(
            path, config.SR_LUNG, f"Lung {cls}",
            f"example_panel_lung_{cls}.png"))

    return outputs


def main():
    """Generate the full EDA figure set under results/figures/eda/."""
    _apply_style()
    _ensure_dir()
    manifest, cycles = load_tables()

    produced = [
        plot_class_dist_heart(manifest),
        plot_class_dist_lung(cycles),
        plot_duration_hist_heart(manifest),
        plot_duration_hist_lung(manifest),
        plot_heart_per_db(manifest),
        plot_lung_per_class(cycles),
        plot_icbhi_native_sr(manifest),
    ]
    produced.extend(plot_example_panels(manifest, cycles))

    print(f"[eda] wrote {len(produced)} figures to {EDA_DIR}")
    for p in produced:
        print(f"  - {os.path.relpath(p, config.PROJECT_ROOT)}")
    return produced


if __name__ == "__main__":
    main()
