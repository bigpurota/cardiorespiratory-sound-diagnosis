// 04-ch3-results.typ — Chapter 3: Experimental results (heart + lung). SKELETON.
// The comparison table mirrors the column schema of
// results/tables/unified_comparison.csv:
//   modality, feature_set, model, primary_metric_name, primary_metric, Se, Sp,
//   macro_f1, auc_roc, accuracy, n_train, n_test
// A placeholder row carries [TODO] where final numbers are slotted in. Confusion-
// matrix and learning-curve figure slots reference the real PNG paths under
// results/figures/ (commented so the file compiles before figures are embedded).

#import "../helpers.typ": *

= Experimental results

This chapter reports the empirical results of the classical and deep-learning
experiments on heart and lung sounds, under the leakage-safe protocol of
Chapter 2. All headline numbers are stated at the recording level for heart and
at the cycle level for lung, using the official ((Se + Sp) / 2) metric of each
task. #text(fill: rgb("#b00020"), weight: "bold")[[TODO: numbers from experiments —
fill the comparison table and the per-task discussion from
`results/tables/unified_comparison.csv` and the `metrics_*.csv` files]]

== Unified comparison table

@tab-unified summarises every model under one schema. The placeholder rows
reproduce the column layout of the results file and are to be replaced by the
final values.

#figure(
  caption: [Unified comparison of all models across both modalities (placeholder — values from `results/tables/unified_comparison.csv`)],
  table(
    columns: 9,
    align: (left, left, left, center, center, center, center, center, center),
    table.header(
      [*Modality*], [*Feature set*], [*Model*], [*Metric*],
      [*Score*], [*Se*], [*Sp*], [*macro-F1*], [*n#sub[test]*],
    ),
    [heart], [MFCC+Δ], [LogReg], [MAcc], [TODO], [TODO], [TODO], [TODO], [TODO],
    [heart], [MFCC+Δ], [SVM], [MAcc], [TODO], [TODO], [TODO], [TODO], [TODO],
    [heart], [MFCC+Δ], [RF], [MAcc], [TODO], [TODO], [TODO], [TODO], [TODO],
    [heart], [MFCC+Δ+spec], [XGBoost], [MAcc], [TODO], [TODO], [TODO], [TODO], [TODO],
    [lung], [MFCC+Δ], [SVM], [ICBHI], [TODO], [TODO], [TODO], [TODO], [TODO],
    [lung], [MFCC+Δ], [RF], [ICBHI], [TODO], [TODO], [TODO], [TODO], [TODO],
    [lung], [log-mel], [CNN], [ICBHI], [TODO], [TODO], [TODO], [TODO], [TODO],
  ),
) <tab-unified>

#text(fill: rgb("#0050b0"), weight: "bold")[
  [Implementation note: at final write-up this hand-written placeholder table will
  be replaced by a programmatic embed of results/tables/unified_comparison.csv via
  `#csv("../../results/tables/unified_comparison.csv")` mapped to `table(..)`.]
]

== Heart-sound results (PhysioNet/CinC 2016)

#text(fill: rgb("#b00020"), weight: "bold")[[TODO: discuss the heart models —
best classical model and its MAcc, sensitivity/specificity trade-off, the effect
of the spectral feature set, and the CNN result. Reference @tab-unified.]]

// Confusion-matrix figure slot (real paths exist under results/figures/):
//   cm_heart_B_xgb.png (best classical), cm_heart_A_svm.png, etc.
#figure(
  kind: image,
  rect(width: 60%, height: 4cm, stroke: 0.5pt + gray)[
    #align(center + horizon)[#text(fill: gray)[Figure slot — `results/figures/cm_heart_B_xgb.png`]]
  ],
  caption: [Confusion matrix of the best heart-sound classifier (placeholder)],
) <fig-cm-heart>

== Lung-sound results (ICBHI 2017)

#text(fill: rgb("#b00020"), weight: "bold")[[TODO: discuss the lung models —
ICBHI score per model, the normal-vs-abnormal sensitivity/specificity balance,
the difficulty of the minority "both" class, and the CNN result. Note honestly
that the lung scores are lower than heart, consistent with the harder 4-class
imbalanced task.]]

#figure(
  kind: image,
  rect(width: 60%, height: 4cm, stroke: 0.5pt + gray)[
    #align(center + horizon)[#text(fill: gray)[Figure slot — `results/figures/cm_lung_cnn.png`]]
  ],
  caption: [Confusion matrix of the best lung-sound classifier (placeholder)],
) <fig-cm-lung>

// Learning-curve figure slot for the CNN (overfitting check):
#figure(
  kind: image,
  rect(width: 60%, height: 4cm, stroke: 0.5pt + gray)[
    #align(center + horizon)[#text(fill: gray)[Figure slot — `results/figures/learning_curve_lung_cnn.png`]]
  ],
  caption: [Training and validation learning curves for the lung CNN (placeholder)],
) <fig-lc-lung>

== Volumetric characteristics

A summary of sample counts, model sizes and training times is given here and in
full in Annex B (Annex 5 §2.5 requires volumetric characteristics).

#figure(
  caption: [Volumetric characteristics of the experiments (placeholder — see Annex B)],
  table(
    columns: 4,
    align: (left, center, center, center),
    table.header([*Quantity*], [*Heart*], [*Lung*], [*Source*]),
    [Training samples (windows / cycles)], [TODO], [TODO], [unified\_comparison.csv `n_train`],
    [Test samples], [TODO], [TODO], [unified\_comparison.csv `n_test`],
    [Best classical model parameters], [TODO], [TODO], [model size log],
    [CNN parameters], [TODO], [TODO], [`count_params`],
    [Classical training time], [TODO], [TODO], [training log],
    [CNN training time], [TODO], [TODO], [training log],
  ),
) <tab-volumetrics>

=== Chapter summary

#text(fill: rgb("#b00020"), weight: "bold")[[TODO: one paragraph — which method
family performed best on each modality, and the headline scores, leading into the
cross-modal analysis of Chapter 4.]]

#team-note[
  The heart and lung experiments were run by #MEMBER("08") (classical) and
  #MEMBER("09") (deep learning); the metric framework, tables and confusion-matrix
  figures were produced by #MEMBER("10").
]
