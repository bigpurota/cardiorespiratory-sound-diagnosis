// 08-annexes.typ — Annexes (Annex 5 §10; Annex 7 §1.2 — annexes lettered
// alphabetically: Annex A, B, C, D). Each annex is numbered and listed in the
// Table of Contents. SKELETON with [TODO] slots. Heading numbering switches to
// letters here so headings read "Annex A", "Annex B", ...

// Switch heading numbering to alphabetic for the annexes (Annex 7 §1.2: annexes
// lettered A, B, C, ...). The numbering prefix supplies the letter, so headings
// read e.g. "Annex A   Full metric tables".
#set heading(numbering: (..n) => {
  let nums = n.pos()
  if nums.len() == 1 {
    [Annex #numbering("A", nums.at(0))]
  } else {
    numbering("A.1", ..nums)
  }
})
#counter(heading).update(0)

= Full metric tables

This annex reproduces the complete per-model metric tables for both modalities
and both feature sets, beyond the summary in Chapter 3.

#text(fill: rgb("#b00020"), weight: "bold")[[TODO: embed the full
results/tables/metrics_heart_classical.csv, metrics_lung_classical.csv and the
CNN metric rows — all columns (Se, Sp, macro-F1, AUC, accuracy, per-class Se).]]

= Volumetric characteristics and reproducibility

Annex 5 §2.5 requires volumetric characteristics. This annex gives sample sizes,
dataset volumes, model sizes, training times and the exact software environment.

#figure(
  caption: [Dataset and experiment volumetrics (placeholder)],
  table(
    columns: 2,
    align: (left, left),
    table.header([*Item*], [*Value*]),
    [Heart recordings (CinC 2016, A–E)], [TODO],
    [Heart train / test windows], [TODO / TODO],
    [Lung recordings (ICBHI 2017)], [TODO],
    [Lung respiratory cycles], [TODO (4-class: TODO)],
    [Lung train / test cycles], [TODO / TODO],
    [Common sampling rate], [4000 Hz],
    [Heart bandpass], [20–400 Hz, Butterworth order 4],
    [Lung bandpass], [200–1800 Hz, Butterworth order 4],
    [Log-mel image size], [64 × 128],
    [EfficientNet-B0 parameters], [approx. 4,010,110],
    [SmallCNN parameters], [TODO (`count_params`)],
    [Classical training time], [TODO],
    [CNN training time], [TODO],
    [Lines of code (src/)], [TODO (`cloc src/`)],
    [Random seed], [42],
  ),
) <tab-annexB-volumetrics>

*Pinned software environment.* Python 3.11; librosa 0.11.0; scikit-learn 1.8.0;
XGBoost 3.2.0; PyTorch 2.11.0; torchaudio 2.11.0; timm ≥ 1.0; imbalanced-learn
0.14.1; numpy ≥ 1.26; scipy ≥ 1.13; pandas ≥ 2.2. The full pinned list is in the
repository `requirements.txt`.

= Spectrogram and exploratory-analysis gallery

#text(fill: rgb("#b00020"), weight: "bold")[[TODO: embed the EDA figures already
on disk under results/figures/eda/ — class distributions, duration histograms,
per-database/per-class counts, native sampling-rate histogram, and the example
spectrogram panels (heart normal/abnormal; lung normal/crackle/wheeze/both).]]

= Repository structure

The reproducible code accompanying this report is organised as follows
(top-level, abridged):

```text
config.py                  shared config; seeds RNGs at 42; canonical paths/rates
params/{heart,lung}.yaml   per-modality preprocessing parameters
src/
  preprocess.py            resample → Butterworth bandpass → peak-normalise
  segment.py               fixed 3-s heart windows (silence/ragged-tail guards)
  features.py              MFCC+Δ+ΔΔ + spectral stats (240-d / 250-d vectors)
  spectrograms.py          torchaudio log-mel 64×128 image
  split.py                 patient-level splits + assert_no_patient_leakage
  metrics.py               MAcc, ICBHI score, majority vote, confusion matrices
  cnn.py                   SmallCNN + EfficientNet-B0 builders
  train_classical.py       classical fit→predict→evaluate driver
  train_cnn.py             deep-learning train+evaluate driver
results/
  tables/unified_comparison.csv   one row per (modality, feature set, model)
  figures/                        confusion matrices, learning curves, EDA
```

The repository link is supplied as a separate `TXT` file in the submission
package, as required.
