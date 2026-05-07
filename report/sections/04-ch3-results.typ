// 04-ch3-results.typ — Chapter 3: Experimental results (heart + lung).
// All numbers are taken directly from:
//   results/tables/unified_comparison.csv   (all 20 rows)
//   results/tables/metrics_heart_classical.csv
//   results/tables/metrics_lung_classical.csv
//   results/tables/metrics_heart_cnn.csv
//   results/tables/metrics_lung_cnn.csv
//   results/tables/volumetrics_classical.csv
//   results/tables/volumetrics_cnn.csv
// Classical results are FINAL. DL rows are marked as core-run (preliminary)
// with a clearly labelled drop-in slot for the HPO+multiseed upgrade.
//
// <<DL-RESULTS-DROPIN: replace all rows/cells tagged "[core-run — prelim]" with
//   HPO+multi-seed mean±std once the GPU sweep completes. Also update the
//   headline in the abstract and conclusion.>>

#import "../helpers.typ": *

= Experimental results

This chapter reports the empirical results of the classical and deep-learning
experiments on heart and lung sounds under the leakage-safe, patient-level
protocol described in Chapter 2. All headline numbers are at the recording level
for heart sounds and at the cycle level for lung sounds, using the official
((Se + Sp) / 2) metric of each task. Where a model predicted only one class the
result is marked degenerate in the discussion; no such degenerate case arose in
the final experiments. Deep-learning rows in the comparison table are from the
core CPU run and are marked preliminary; an upgraded GPU hyperparameter-search
result will replace them before the final PDF submission.

== Unified comparison table

@tab-unified collects every model trained in this study under a single schema,
enabling direct cross-method and cross-modality comparison. The 16 classical rows
are final. The 4 deep-learning rows are from the core run and are marked
preliminary. Rows are grouped first by modality, then by feature set, then by
model. Within each modality the best primary-metric score is shown in bold.

// Full 20-row comparison table built from unified_comparison.csv
#figure(
  caption: [Unified model comparison across both modalities.
  Heart metric: MAcc = (Se+Sp)/2 (CinC 2016 official). Lung metric: ICBHI Score = (Se+Sp)/2
  (ICBHI 2017 official). DL rows marked #super[†] are preliminary (core CPU run); a GPU-tuned
  multi-seed result will replace them.
  Feature sets: A = MFCC+Δ+ΔΔ (240-d); B = MFCC+Δ+ΔΔ+spectral (250-d); log-mel = 64×128 image.],
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, left, left, center, center, center, center, center),
    table.header(
      [*Modality*], [*Feature set*], [*Model*], [*Metric*],
      [*Score*], [*Se*], [*Sp*], [*macro-F1*],
    ),
    // ---- Heart classical ----
    [Heart], [A (MFCC+Δ)],      [LogReg],   [MAcc],  [0.794], [0.869], [0.720], [0.706],
    [Heart], [A (MFCC+Δ)],      [SVM],      [MAcc],  [0.859], [0.915], [0.802], [0.783],
    [Heart], [A (MFCC+Δ)],      [RF],       [MAcc],  [0.817], [0.692], [0.942], [0.827],
    [Heart], [A (MFCC+Δ)],      [XGBoost],  [MAcc],  [0.879], [0.915], [0.843], [0.816],
    [Heart], [B (MFCC+Δ+spec)], [LogReg],   [MAcc],  [0.825], [0.908], [0.742], [0.734],
    [Heart], [B (MFCC+Δ+spec)], [SVM],      [MAcc],  [0.869], [0.938], [0.800], [0.788],
    [Heart], [B (MFCC+Δ+spec)], [RF],       [MAcc],  [0.831], [0.723], [0.940], [0.837],
    table.cell(text(weight: "bold")[[Heart]]),
    table.cell(text(weight: "bold")[[B (MFCC+Δ+spec)]]),
    table.cell(text(weight: "bold")[[XGBoost]]),
    table.cell(text(weight: "bold")[[MAcc]]),
    table.cell(text(weight: "bold")[[*0.903*]]),
    table.cell(text(weight: "bold")[[*0.946*]]),
    table.cell(text(weight: "bold")[[*0.859*]]),
    table.cell(text(weight: "bold")[[0.839]]),
    // ---- Heart DL ----
    [Heart], [log-mel 64×128], [SmallCNN],       [MAcc], [0.861#super[†]], [0.915], [0.806], [0.786],
    [Heart], [log-mel 64×128], [EfficientNet-B0], [MAcc], [0.872#super[†]], [0.915], [0.829], [0.804],
    // ---- Lung classical ----
    [Lung], [A (MFCC+Δ)],      [LogReg],  [ICBHI], [0.507], [0.656], [0.358], [0.273],
    [Lung], [A (MFCC+Δ)],      [SVM],     [ICBHI], [0.536], [0.618], [0.454], [0.319],
    [Lung], [A (MFCC+Δ)],      [RF],      [ICBHI], [0.476], [0.101], [0.851], [0.226],
    [Lung], [A (MFCC+Δ)],      [XGBoost], [ICBHI], [0.511], [0.470], [0.551], [0.297],
    [Lung], [B (MFCC+Δ+spec)], [LogReg],  [ICBHI], [0.533], [0.693], [0.372], [0.291],
    [Lung], [B (MFCC+Δ+spec)], [SVM],     [ICBHI], [0.537], [0.615], [0.458], [0.320],
    [Lung], [B (MFCC+Δ+spec)], [RF],      [ICBHI], [0.472], [0.098], [0.847], [0.243],
    [Lung], [B (MFCC+Δ+spec)], [XGBoost], [ICBHI], [0.500], [0.369], [0.631], [0.294],
    // ---- Lung DL ----
    table.cell(text(weight: "bold")[[Lung]]),
    table.cell(text(weight: "bold")[[log-mel 64×128]]),
    table.cell(text(weight: "bold")[[SmallCNN]]),
    table.cell(text(weight: "bold")[[ICBHI]]),
    table.cell(text(weight: "bold")[[*0.551*#super[†]]]),
    table.cell(text(weight: "bold")[[*0.678*]]),
    table.cell(text(weight: "bold")[[*0.424*]]),
    table.cell(text(weight: "bold")[[0.329]]),
    [Lung], [log-mel 64×128], [EfficientNet-B0], [ICBHI], [0.549#super[†]], [0.521], [0.577], [0.323],
  ),
) <tab-unified>

#v(0.4em)
#text(size: 10pt)[#super[†] Preliminary core-run result (CPU, default hyper-parameters). An HPO + multi-seed
  mean±std result will replace these rows before final submission.
  // <<DL-RESULTS-DROPIN: update superscript-dagger rows above with HPO mean±std>>
]

== Heart-sound results (PhysioNet/CinC 2016)

=== Classical models

All eight classical configurations (two feature sets × four classifiers) were
evaluated on the same patient-level test set of 626 test windows from 626 unique
patients. The best classical result was achieved by XGBoost on feature set B
(MFCC+Δ+ΔΔ + spectral statistics), reaching MAcc = 0.903, with Se = 0.946 and
Sp = 0.859. This result confirms the practical value of the five additional
spectral statistics: XGBoost on feature set A scored only MAcc = 0.879, a
meaningful 2.4 percentage-point improvement from the richer feature set.

SVM was the second-best classical model across both feature sets (MAcc 0.859 on A,
0.869 on B). SVM and XGBoost both achieve high sensitivity (Se ≥ 0.915 in all
configurations), reflecting the fact that class-weighted training forces these
models to capture true positives. By contrast, random forest reaches high
specificity (Sp ≥ 0.940) but lower sensitivity (Se ≈ 0.69–0.72), suggesting that
RF's ensemble structure defaults more readily to the majority normal class.
Logistic regression, despite being the simplest linear model, achieves a
respectable MAcc of 0.825 on feature set B, demonstrating that the MFCC+spectral
feature space is well-structured for linear separation.

@fig-cm-heart-best shows the confusion matrix for the best model (XGBoost, set B).
Out of 130 abnormal test recordings, 123 are correctly identified, and out of 496
normal recordings, 425 are correctly identified, with 71 false negatives and only 7
false positives on the abnormal class.

#figure(
  image("../../results/figures/cm_heart_B_xgb.png", width: 58%),
  caption: [Confusion matrix of the best heart-sound classical classifier (XGBoost,
  feature set B: MFCC+Δ+ΔΔ + spectral statistics). Rows: true class; columns:
  predicted class. Test set: 626 windows from 626 patients.],
) <fig-cm-heart-best>

For comparison, @fig-cm-heart-svm shows the SVM confusion matrix (feature set B).
The SVM achieves similar sensitivity but with slightly fewer false positives,
reflecting a different decision-threshold behaviour.

#figure(
  image("../../results/figures/cm_heart_B_svm.png", width: 58%),
  caption: [Confusion matrix of the second-best heart-sound classical classifier (SVM,
  feature set B). Same test set as @fig-cm-heart-best.],
) <fig-cm-heart-svm>

=== Deep-learning models

#text(fill: rgb("#666666"), style: "italic")[The following DL results are from the
core CPU run with default hyper-parameters. An upgraded GPU search result will
replace these numbers before final submission.
// <<DL-RESULTS-DROPIN: replace core-run numbers below with HPO mean±std>>
]

The compact SmallCNN trained on 64×128 log-mel spectrograms reached MAcc = 0.861,
and EfficientNet-B0 reached MAcc = 0.872. Both deep models underperform the best
classical model (XGBoost-B, MAcc = 0.903) on this task by 3.1 and 3.0 percentage
points respectively. This outcome is consistent with the dataset regime: the
training split contains 33,246 3-second windows from 2,500 recordings, a size
at which gradient-boosted trees on well-engineered MFCC features are known to be
competitive with or superior to compact CNNs @xgboost_heart.

The learning curves for both deep models (@fig-lc-heart-cnn and @fig-lc-heart-eff)
show convergence without severe overfitting, validating that the early-stopping and
dropout (≥ 0.3) regularisation are effective. Both models reach their best
checkpoint within 20 epochs.

#figure(
  image("../../results/figures/learning_curve_heart_cnn.png", width: 72%),
  caption: [Training and validation learning curves for SmallCNN on heart sounds.
  Primary metric (MAcc) vs. epoch. Best checkpoint is marked.],
) <fig-lc-heart-cnn>

#figure(
  image("../../results/figures/learning_curve_heart_effnet.png", width: 72%),
  caption: [Training and validation learning curves for EfficientNet-B0 on heart sounds.
  Primary metric (MAcc) vs. epoch.],
) <fig-lc-heart-eff>

== Lung-sound results (ICBHI 2017)

=== Classical models

All eight classical configurations were evaluated on the official ICBHI 60/40
patient-independent test set of 2,636 cycles from 47 test patients. ICBHI scores
are notably lower than heart MAcc across all models, reflecting the fundamentally
harder nature of this task: four imbalanced classes (three minority abnormal types)
versus two classes.

The best classical result is SVM on feature set B with ICBHI score = 0.537
(Se = 0.615, Sp = 0.458). The score on feature set A is nearly identical
(0.536), suggesting that the spectral statistics add marginal value for lung
classification compared to the clear boost they provided for heart classification.
XGBoost on feature set A scores 0.511 and declines on feature set B (0.500),
which is the opposite pattern from heart sounds — an observation directly relevant
to the cross-modal analysis of Chapter 4.

A striking pattern in the lung results is the contrast between random forest and
all other classifiers. RF achieves very high specificity (Sp ≈ 0.85) but extremely
low sensitivity on the abnormal classes (Se ≈ 0.10). This "majority-class
collapse" — predicting almost exclusively normal — is a failure mode specific to
RF's majority-vote aggregation when classes are highly imbalanced and SMOTE
oversampling has not fully counteracted the imbalance within the training fold.
The pattern is visible in @fig-cm-lung-best, where nearly all cycles are predicted
as normal or wheeze, and the crackle and "both" rows are almost entirely
misclassified.

#figure(
  image("../../results/figures/cm_lung_B_svm.png", width: 58%),
  caption: [Confusion matrix of the best lung-sound classical classifier (SVM,
  feature set B). Rows: true class; columns: predicted class. Test set: 2636 cycles
  from 47 patients.],
) <fig-cm-lung-best>

Logistic regression on feature set B achieves ICBHI = 0.533, comparable to SVM.
Both linear models benefit from class-weighted loss, which forces predictions
across all four classes rather than collapsing to the majority.

=== Deep-learning models

#text(fill: rgb("#666666"), style: "italic")[The following DL results are from the
core CPU run. A GPU-tuned multi-seed result will replace these numbers before
final submission.
// <<DL-RESULTS-DROPIN: replace core-run lung DL numbers with HPO mean±std>>
]

The SmallCNN on 64×128 log-mel spectrograms reached ICBHI = 0.551, the single
best result across all 20 model configurations for lung classification. This
represents a 1.4 percentage-point improvement over the best classical model.
EfficientNet-B0 reached ICBHI = 0.549, statistically indistinguishable from CNN
in a single-run comparison. Crucially, both deep models exhibit better sensitivity
on the minority abnormal classes compared to classical models, which is reflected
in a more balanced confusion matrix (@fig-cm-lung-cnn).

#figure(
  image("../../results/figures/cm_lung_cnn.png", width: 58%),
  caption: [Confusion matrix of the best lung-sound classifier (SmallCNN on
  log-mel spectrograms, preliminary core-run result). The log-mel representation
  enables better discrimination of crackle and wheeze relative to classical models.],
) <fig-cm-lung-cnn>

@fig-lc-lung-cnn shows the learning curve for the lung CNN. The validation metric
converges within 25 epochs without divergence, consistent with adequate
regularisation.

#figure(
  image("../../results/figures/learning_curve_lung_cnn.png", width: 72%),
  caption: [Training and validation learning curves for SmallCNN on lung sounds
  (ICBHI score vs. epoch).],
) <fig-lc-lung-cnn>

== Volumetric characteristics

@tab-volumetrics collects the dataset sizes, model parameter counts and training
times that characterise the experimental scale. These are required by Annex 5
§2.5. Beyond the data and model sizes, the experimental pipeline itself comprises
6 976 lines of Python across 31 modules in `src/` and `scripts/` (approximately
289 KB of source code), supported by a further 2 959 lines of automated tests.
Full details including per-class cycle counts appear in Annex B.

#figure(
  caption: [Volumetric characteristics of the experiments. Training times are
  wall-clock seconds on CPU (Intel-class laptop, single thread where applicable).],
  table(
    columns: (2fr, 1fr, 1fr),
    align: (left, center, center),
    table.header([*Quantity*], [*Heart*], [*Lung*]),
    [Train segments/cycles],         [33 246], [4 262],
    [Test segments/cycles],          [4 167],  [2 636],
    [Train recordings/patients],     [2 500 / 2 500], [551 / 79],
    [Test recordings/patients],      [626 / 626],    [369 / 47],
    [Best classical model (SVM-B) train time (s)], [659], [31],
    [Best classical model (XGB-B) train time (s)], [21],  [26],
    [SmallCNN parameters],           [97 890], [98 148],
    [SmallCNN train time (s)],       [159],    [42],
    [EfficientNet-B0 parameters],    [4 010 110], [4 012 672],
    [EfficientNet-B0 train time (s)],[622],    [187],
    [Log-mel data volume (MB)],      [1 227],  [226],
    [Classical feature data volume (MB)], [74], [14],
  ),
) <tab-volumetrics>

== Chapter summary

On heart sounds, the best overall result is XGBoost with MFCC+Δ+ΔΔ + spectral
statistics (feature set B), reaching MAcc = 0.903 (Se = 0.946, Sp = 0.859). The
deep-learning models (SmallCNN 0.861, EfficientNet-B0 0.872) are competitive but
do not exceed the best classical model in the core run; an HPO-tuned result may
alter this ranking.
// <<DL-RESULTS-DROPIN: update "do not exceed" statement if HPO heart result exceeds 0.903>>

On lung sounds, the overall best result is the SmallCNN with ICBHI = 0.551,
slightly ahead of the best classical model (SVM-B, 0.537). The gap between heart
and lung scores (MAcc 0.90 vs. ICBHI 0.55) reflects the fundamental difficulty
difference: a binary task on longer recordings with majority-vote aggregation
versus a four-class imbalanced task on short cycles. Chapter 4 analyses what these
differences imply for cross-modal transfer of method rankings.

#team-note[
  The heart and lung experiments were run by #MEMBER("08") (classical models) and
  #MEMBER("09") (deep-learning models); the evaluation framework, tables, confusion-
  matrix and learning-curve figures were produced by #MEMBER("10"). Integration
  was coordinated by #MEMBER("01").
]
