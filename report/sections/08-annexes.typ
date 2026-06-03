
#set heading(numbering: (..n) => {
  let nums = n.pos()
  if nums.len() == 1 {
    [Annex #numbering("A", nums.at(0))]
  } else {
    numbering("A.1", ..nums)
  }
})
#counter(heading).update(0)
#state("annex-mode", false).update(true)

= Full metric tables

This annex reproduces the complete per-model metric tables for both modalities,
beyond the rounded summary in Chapter 3: the heart-sound classical results are given
in @tab-annex-heart-classical and the lung-sound classical results in
@tab-annex-lung-classical. Numbers are taken directly from
`results/tables/metrics_heart_classical.csv`, `metrics_lung_classical.csv`,
`metrics_heart_cnn.csv` and `metrics_lung_cnn.csv`.

== Heart-sound metrics (classical models)

#figure(
  caption: [Complete heart-sound classical metrics. $"MAcc" = ("Se"+"Sp") \/ 2$. AUC-ROC
  is recording-level after majority vote. Feature sets: A = MFCC+Δ+ΔΔ ($240$-d);
  B = MFCC+Δ+ΔΔ+spectral stats ($250$-d).],
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, center, center, center, center, center),
    table.header(
      [*Set*], [*Model*], [*MAcc*], [*Se*], [*Sp*], [*macro-F1*], [*AUC*]),
    [A], [LogReg],  [0.7945], [0.8692], [0.7198], [0.7062], [0.8494],
    [A], [SVM],     [0.8589], [0.9154], [0.8024], [0.7827], [0.9330],
    [A], [RF],      [0.8169], [0.6923], [0.9415], [0.8270], [0.9418],
    [A], [XGBoost], [0.8791], [0.9154], [0.8427], [0.8158], [0.9500],
    [B], [LogReg],  [0.8248], [0.9077], [0.7419], [0.7339], [0.8985],
    [B], [SVM],     [0.8694], [0.9385], [0.8004], [0.7882], [0.9352],
    [B], [RF],      [0.8313], [0.7231], [0.9395], [0.8370], [0.9450],
    [B], [XGBoost], [*0.9025*], [*0.9462*], [*0.8589*], [0.8394], [*0.9573*],
  ),
) <tab-annex-heart-classical>

== Heart-sound metrics (deep-learning models, HPO-tuned, 3-seed mean)

#figure(
  caption: [Heart-sound deep-learning metrics. HPO-tuned (val-only selection, 128
  trials); values are the mean over three seeds (1, 2, 42). Parameter counts reflect
  the HPO-selected architecture (the tuned SmallCNN uses wider 32–256 channels).],
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, center, center, center, center, center),
    table.header(
      [*Model*], [*Params*], [*MAcc*], [*Se*], [*Sp*], [*macro-F1*], [*AUC*]),
    [SmallCNN],        [389 314],   [0.871 ± 0.009], [0.910], [0.831], [0.805], [0.946],
    [EfficientNet-B0], [4 010 110], [0.898 ± 0.008], [0.936], [0.860], [0.837], [0.966],
  ),
)

== Lung-sound metrics (classical models)

#figure(
  caption: [Complete lung-sound classical metrics. $"ICBHI Score" = ("Se"_"abnormal" + "Sp") \/ 2$.
  $"Se"_"crk"$, $"Se"_"wh"$, $"Se"_"both"$ = per-class sensitivities on crackle, wheeze and both classes.],
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, center, center, center, center, center, center),
    table.header(
      [*Set*], [*Model*], [*ICBHI*], [*Se*], [*Sp*], [$bold("Se"_"crk")$], [$bold("Se"_"wh")$], [$bold("Se"_"both")$]),
    [A], [LogReg],  [0.5069], [0.6561], [0.3577], [0.357], [0.340], [0.140],
    [A], [SVM],     [0.5363], [0.6180], [0.4545], [0.404], [0.405], [0.105],
    [A], [RF],      [0.4763], [0.1013], [0.8513], [0.055], [0.094], [0.000],
    [A], [XGBoost], [0.5108], [0.4703], [0.5513], [0.292], [0.244], [0.140],
    [B], [LogReg],  [0.5329], [0.6933], [0.3724], [0.397], [0.349], [0.163],
    [B], [SVM],     [*0.5368*], [0.6152], [0.4583], [0.405], [0.394], [0.116],
    [B], [RF],      [0.4722], [0.0976], [0.8468], [0.066], [0.102], [0.023],
    [B], [XGBoost], [0.5002], [0.3690], [0.6314], [0.214], [0.231], [0.116],
  ),
) <tab-annex-lung-classical>

== Lung-sound metrics (deep-learning models, HPO-tuned, 3-seed mean)

#figure(
  caption: [Lung-sound deep-learning metrics. HPO-tuned (val-only selection, 128
  trials); values are the mean over three seeds (1, 2, 42). Per-class sensitivities
  vary across seeds; representative-run breakdowns appear in Chapter 3.],
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, auto, center, center, center, center),
    table.header(
      [*Model*], [*Params*], [*ICBHI*], [*Se*], [*Sp*], [*macro-F1*]),
    [SmallCNN],        [98 148],    [0.540 ± 0.022], [0.755], [0.325], [0.301],
    [EfficientNet-B0], [4 012 672], [*0.555 ± 0.016*], [0.509], [0.601], [0.309],
  ),
)

= Volumetric characteristics and reproducibility

Annex 5 §2.5 requires volumetric characteristics. @tab-annexB-volumetrics provides
sample sizes, model sizes, training times and dataset volumes, derived from
`results/tables/volumetrics_classical.csv` and `results/tables/volumetrics_cnn.csv`.

#figure(
  caption: [Dataset and experiment volumetrics. Heart patient counts are omitted:
  CinC 2016 provides no recording-to-subject map, so the heart split is grouped at
  the recording level (Section 2.1).],
  table(
    columns: (3fr, 1fr, 1fr),
    align: (left, center, center),
    table.header([*Item*], [*Heart*], [*Lung*]),
    [Dataset],                   [CinC 2016 (A–E)], [ICBHI 2017],
    [Train recordings / patients], [2 500 / —], [551 / 79],
    [Test recordings / patients],  [626 / —],   [369 / 47],
    [Train segments/cycles],     [33 246], [4 262],
    [Test segments/cycles],      [4 167],  [2 636],
    [Segment/cycle length],      [3.0 s (windows)], [3.0 s (padded cycles)],
    [Common sampling rate],      [4 000 Hz], [4 000 Hz],
    [Bandpass],                  [20–400 Hz], [200–1800 Hz],
    [Filter order],              [Butterworth 4], [Butterworth 4],
    [Feature set A dimension],   [240-d], [240-d],
    [Feature set B dimension],   [250-d], [250-d],
    [Log-mel image size],        [64 × 128], [64 × 128],
    [Classical feature data (MB)], [74.2], [13.7],
    [Log-mel data (MB)],         [1 226.8], [226.2],
    [SVM-B train time (s)],      [659], [31],
    [XGBoost-B train time (s)],  [21],  [26],
    [SmallCNN parameters],       [389 314], [98 148],
    [SmallCNN train time (s)],   [262], [26],
    [EfficientNet-B0 parameters], [4 010 110], [4 012 672],
    [EfficientNet-B0 train time (s)], [1 345], [204],
    [Random seed],               [42], [42],
  ),
) <tab-annexB-volumetrics>

_Pinned software environment._ Python 3.11; librosa 0.11.0; scikit-learn 1.8.0;
XGBoost 3.2.0; PyTorch 2.11.0; torchaudio 2.11.0; timm $gt.eq 1.0$;
imbalanced-learn 0.14.1; numpy $gt.eq 1.26$; scipy $gt.eq 1.13$; pandas $gt.eq 2.2$.
The full pinned dependency list is committed to the repository as `requirements.txt`.

= Spectrogram and exploratory-analysis gallery

The following figures were generated by the exploratory data analysis stage of
the pipeline (scripts that read the raw datasets and produce class-distribution,
duration-histogram and example-spectrogram plots).

_Heart-sound class distribution_ (@fig-eda-class-heart): across the $3126$
recordings of databases A–E, normal recordings dominate ($2495$, $79.8%$) over
abnormal ones ($631$, $20.2%$); the A–E label files carry only the normal/abnormal
verdict, so no recording falls in the organisers' residual "unsure" tier. This
roughly four-to-one class imbalance motivates class-weighted training.

#figure(
  image("../../results/figures/eda/class_dist_heart.png", width: 70%),
  caption: [Class distribution in the CinC 2016 heart-sound training set (databases A–E
  combined): $2495$ normal versus $631$ abnormal recordings.],
) <fig-eda-class-heart>

_Heart recording duration_ (@fig-eda-dur-heart): recording lengths range from
approximately $5$ to $120$ seconds, with a mode around $20"–"30$ seconds. The $3$-second
windowing strategy in Chapter 2 generates between $1$ and $40$ windows per recording.

#figure(
  image("../../results/figures/eda/duration_hist_heart.png", width: 70%),
  caption: [Duration distribution of CinC 2016 heart recordings.],
) <fig-eda-dur-heart>

_Heart recordings per database_ (@fig-eda-db): database E dominates the pool with
$2141$ of the $3126$ recordings (about $68%$), followed by B ($490$) and A ($409$),
while C ($31$) and D ($55$) contribute only a handful each. This heavy skew toward a
single source database is a caveat for the heart results, and the bandpass filter
must operate across this heterogeneous source mix.

#figure(
  image("../../results/figures/eda/heart_per_db_counts.png", width: 70%),
  caption: [Recording counts per CinC 2016 source database (A–E).],
) <fig-eda-db>

_Lung-sound class distribution_ (@fig-eda-class-lung): the ICBHI 2017 cycle-level
distribution shows that "normal" ($52.8%$) greatly outnumbers the three abnormal
classes, with "both" (crackle + wheeze) being the rarest at $7.3%$.

#figure(
  image("../../results/figures/eda/class_dist_lung.png", width: 70%),
  caption: [Cycle-level class distribution in ICBHI 2017 (6 898 annotated cycles).],
) <fig-eda-class-lung>

_Native sampling-rate histogram_ (@fig-eda-sr): the four ICBHI recording devices
contribute at $4000$ Hz, $10 thin 000$ Hz, $22 thin 050$ Hz and $44 thin 100$ Hz. Resampling to $4000$ Hz
in preprocessing eliminates device-specific spectral artefacts.

#figure(
  image("../../results/figures/eda/icbhi_native_sr_hist.png", width: 70%),
  caption: [Native sampling-rate distribution across ICBHI 2017 recordings. All
  recordings are resampled to $4000$ Hz before feature extraction.],
) <fig-eda-sr>

_Example spectrograms_ (heart, @fig-eda-spec-heart-norm and @fig-eda-spec-heart-abn):
the normal spectrogram shows clear S1–S2 periodicity concentrated below $150$ Hz.
The abnormal spectrogram shows a sustained mid-systolic murmur occupying a wider
frequency band (the recording is labelled abnormal; CinC 2016 does not provide a
specific murmur type).

#figure(
  image("../../results/figures/eda/example_panel_heart_normal.png", width: 80%),
  caption: [Example log-mel spectrogram panel: normal heart recording (CinC 2016).
  S1 and S2 events are visible as periodic vertical intensifications.],
) <fig-eda-spec-heart-norm>

#figure(
  image("../../results/figures/eda/example_panel_heart_abnormal.png", width: 80%),
  caption: [Example log-mel spectrogram panel: abnormal heart recording (CinC 2016)
  with a mid-systolic murmur. The murmur occupies a broad frequency band between
  the S1 and S2 events.],
) <fig-eda-spec-heart-abn>

_Example lung-cycle panels_ (@fig-eda-spec-lung): representative log-mel spectrograms
for each of the four ICBHI 2017 cycle classes, illustrating the acoustic diversity the
respiratory classifier must separate within a single short cycle.

#figure(
  grid(
    columns: (1fr, 1fr), gutter: 5pt,
    image("../../results/figures/eda/example_panel_lung_normal.png", width: 100%),
    image("../../results/figures/eda/example_panel_lung_crackle.png", width: 100%),
    image("../../results/figures/eda/example_panel_lung_wheeze.png", width: 100%),
    image("../../results/figures/eda/example_panel_lung_both.png", width: 100%),
  ),
  caption: [Example log-mel spectrogram panels for the four ICBHI 2017 cycle classes
  (top row: normal, crackle; bottom row: wheeze, and crackle-plus-wheeze "both").],
) <fig-eda-spec-lung>

= Repository structure

The reproducible code accompanying this report is organised as follows
(top-level, abridged):

```text
params/{heart,lung}.yaml   per-modality preprocessing parameters
src/
  config.py                shared config; seeds RNGs at 42; canonical paths/rates
  preprocess.py            resample → Butterworth bandpass → peak-normalise
  segment.py               fixed 3-s heart windows (silence/ragged-tail guards)
  features.py              MFCC+Δ+ΔΔ + spectral stats (240-d / 250-d vectors)
  spectrograms.py          torchaudio log-mel 64×128 image
  split.py                 patient-level splits + assert_no_patient_leakage
  metrics.py               MAcc, ICBHI score, majority vote, confusion matrices
  cnn.py                   SmallCNN + EfficientNet-B0 builders
  train_classical.py       classical fit→predict→evaluate driver
  train_cnn.py             deep-learning train+evaluate driver
  cross_modal.py           ranking transfer + joint multi-task model
scripts/                   CLI drivers (download → build → run_*)
tests/                     leakage / determinism / metric / pipeline tests
results/
  tables/unified_comparison.csv   one row per (modality, feature set, model)
  tables/metrics_{heart,lung}_{classical,cnn}.csv  full per-model metrics
  tables/volumetrics_{classical,cnn}.csv  dataset sizes, model params, times
  figures/                        confusion matrices, learning curves, EDA
report/
  main.typ                 master Typst document
  sections/                section files (abstract, introduction, ch1–ch4,
                           conclusion, bibliography, annexes)
  refs.bib                 bibliography
  helpers.typ              shared formatting helpers
docs/pdf/                  compiled report PDF (the submission deliverable)
```

The repository link is supplied as a separate `TXT` file in the submission
package, as required by the department.
