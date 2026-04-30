// 02-ch1-litreview.typ — Chapter 1: Analytical literature review (Annex 5 §6).
// SKELETON with structured headings + [CITE]/[TODO] markers. Citations reference
// the REAL sources seeded in report/refs.bib (from SUMMARY.md §Sources). This
// chapter must be an ANALYSIS that positions our work, not a list of papers
// (Annex 5 §6). The number-independent framing is drafted; the per-paragraph
// synthesis prose is to be expanded by the literature-review members.

#import "../helpers.typ": *

= Analytical review of existing approaches

This chapter positions the present study within the literature on auscultation-
sound classification. It proceeds from the physiological basis of the signals,
through the public datasets and the two dominant method families (classical
feature engineering and deep learning), to the cross-modality question that
motivates our contribution, closing with an explicit statement of the research
gap and our novelty claim.

== Physiological basis of auscultation sounds

#text(fill: rgb("#b00020"), weight: "bold")[[TODO: 1–2 paragraphs]]
Heart sounds arise from valve closure and blood flow (S1, S2, murmurs);
respiratory sounds (crackles, wheezes) arise from airway dynamics; arterial
bruits arise from turbulent flow across a stenosis. Their distinct frequency
bands motivate the per-modality bandpass choices used later in Chapter 2.
#text(fill: rgb("#0050b0"), weight: "bold")[[CITE: physiology reference — TODO: add ref]]

== Public datasets

The two heart and lung datasets used here are the de-facto benchmarks for their
tasks. The PhysioNet/CinC 2016 challenge released a large, openly licensed
collection of phonocardiograms with a binary normal/abnormal label
@cinc2016 @liu2016, while the ICBHI 2017 database provides cycle-level
respiratory annotations across four classes @rocha2019. The supplementary CirCor
DigiScope database @circor2022 offers richer murmur metadata and is noted as a
future extension. A recurring methodological point in this literature is the
distinction between the *official* patient-independent splits and ad-hoc random
splits; only the former yield numbers comparable across studies.

#text(fill: rgb("#b00020"), weight: "bold")[[TODO: analytical comparison of the
datasets — size, labels, imbalance, official splits — and why CinC 2016 and
ICBHI 2017 are the right primary choices for this study.]]

== Classical feature-based approaches

A large body of work classifies auscultation sounds from engineered acoustic
features — most prominently mel-frequency cepstral coefficients (MFCC) and their
temporal derivatives, often augmented with spectral statistics — fed to classical
classifiers such as support-vector machines, random forests and gradient boosting
@mfcc_svm @xgboost_heart. These pipelines are computationally cheap, interpretable
and strong baselines.

#text(fill: rgb("#b00020"), weight: "bold")[[TODO: synthesise the reported
accuracies and the metrics used; flag papers that report only raw accuracy on
imbalanced data, and contrast with the (Se+Sp)/2 official metrics.]]

== Deep-learning approaches

Deep models operate on time–frequency images (typically log-mel spectrograms),
ranging from compact convolutional networks to transfer learning from ImageNet
backbones and, more recently, audio transformers @cnn_lstm @ast_patchmix. These
approaches report the strongest numbers but are more data- and compute-hungry,
and several published results are inflated by augmentation applied before the
train/test split.

#text(fill: rgb("#b00020"), weight: "bold")[[TODO: analytical contrast of CNN /
transfer / transformer results; note the leakage and augmentation pitfalls that
make some headline numbers non-replicable.]]

== The cross-modality gap and research novelty

Almost all of the above treats a single modality. Unified models spanning heart
and lung sounds are only beginning to appear @unified_heartlung @auscultabase,
and arterial bruits lack any open dataset suitable for supervised learning
@stroke2024 @carotid_pmc. Methodological surveys of machine-learning pitfalls
@lones2021 stress reproducibility and leakage-safe evaluation as prerequisites
for trustworthy comparison.

The research gap is therefore twofold: (i) a *single, leakage-safe pipeline
evaluated identically across modalities* is rare, and (ii) the *cross-modal
transfer of method rankings* is largely unstudied. Our contribution targets
exactly this gap — a unified comparative pipeline applied to heart and lung
sounds under one patient-level protocol, with an analytical treatment of arterial
sound. This positions the present work as complementary to single-modality
studies: where they optimise one task, we ask whether their conclusions
generalise across auscultation domains.

=== Chapter summary

The literature offers mature single-modality methods but few leakage-safe,
cross-modal comparisons. This motivates the unified pipeline specified in
Chapter 2.

#team-note[
  This chapter was prepared by #MEMBER("02") (heart-sound literature) and
  #MEMBER("03") (lung-sound and arterial literature), with the cross-modality
  framing and novelty statement integrated by #MEMBER("10").
]
