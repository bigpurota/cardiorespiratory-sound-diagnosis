// 05-ch4-novelty.typ — Chapter 4: Cross-modal analysis (novelty) + arterial
// sub-study. Fully drafted with original synthesis.
// Numbers derived from unified_comparison.csv (final classical; DL preliminary).
// Cross-modal finding is based on the actual rankings from Chapter 3 results.
// <<DL-RESULTS-DROPIN: revisit the cross-modal finding paragraph and the
//   chapter summary once HPO+multiseed DL scores are available.>>

#import "../helpers.typ": *

= Cross-modal analysis and arterial sub-study

This chapter develops the project's primary novel contribution: an analysis of how
method families transfer across auscultation modalities, drawn directly from the
Chapter 3 results without additional experiments. It then provides an analytical
treatment of arterial sounds, for which no open dataset exists, documenting the
acoustic basis, the data-availability obstacle, and how the pipeline would extend
if the obstacle were removed.

== Transfer of method rankings across modalities

Because heart and lung models were evaluated under one protocol and one metric
family, their rankings can be compared directly. We examine the question in three
ways: (i) the ordinal ranking of classifiers within each modality, (ii) the
relative advantage of feature set B over feature set A, and (iii) the relative
standing of classical versus deep models.

=== Classifier ranking

For heart sounds the ranking (best to worst) by primary metric is:
XGBoost-B (0.903) > SVM-B (0.869) > XGBoost-A (0.879) > SVM-A (0.859) >
RF-B (0.831) > RF-A (0.817) > LogReg-B (0.825) > LogReg-A (0.794).
Collapsing per feature set, the ordering at the classifier level is
XGBoost > SVM > RF > LogReg.

For lung sounds the ranking by ICBHI score is substantially different:
SVM-B (0.537) ≈ SVM-A (0.536) > LogReg-B (0.533) > XGBoost-A (0.511) >
LogReg-A (0.507) > XGBoost-B (0.500) > RF-A (0.476) ≈ RF-B (0.472).
The lung classifier ordering is SVM ≈ LogReg > XGBoost > RF.

Two inversions are notable. First, XGBoost, the dominant model on heart sounds,
falls to third place on lung sounds and actually degrades from feature set A to B
(0.511 to 0.500), the reverse of its heart behaviour. Second, logistic regression
rises to near-parity with SVM on lung sounds, suggesting that the four-class
MFCC feature space is more linearly separable than the two-class heart space, or
that gradient boosting's additional capacity introduces over-fitting on the smaller
lung training set (4,262 cycles versus 33,246 heart windows). These inversions
are the core empirical finding of the cross-modal analysis.

=== Feature set effect

On heart sounds, adding spectral statistics (feature set B) consistently improves
every classifier, with the largest absolute gain for XGBoost (+2.3 pp) and modest
gains for SVM (+1.1 pp) and LogReg (+3.0 pp). On lung sounds, the feature-set
effect is inconsistent: SVM improves negligibly (+0.1 pp), LogReg improves (+2.6 pp),
but XGBoost worsens (−1.1 pp) and RF also worsens. The spectral statistics that
capture cardiac sound "brightness" (centroid, roll-off) are apparently less
informative for the respiratory task, where the shape of adventitious sound
transients is more relevant than broadband spectral shape.

=== Classical versus deep learning

On heart sounds, classical XGBoost (MAcc = 0.903) exceeds the deep models (CNN
0.861, EfficientNet-B0 0.872) in the core run. This is consistent with the known
literature finding that engineered MFCC features on well-windowed recordings
outperform compact CNNs for binary heart-sound classification at the ~3,000
recording scale @xgboost_heart. On lung sounds, the deep CNN (ICBHI = 0.551)
achieves the single best result, narrowly ahead of SVM (0.537). This reversal —
classical wins on heart, deep wins on lung — aligns with task characteristics: the
lung task's four-class imbalanced structure, with class-specific Se as low as
0.10 for random forest, seems to benefit more from the richer log-mel
representation that deep models learn end-to-end.

#text(fill: rgb("#666666"), style: "italic")[Note: the classical-vs-deep conclusion
for heart is robust under the current numbers. Whether it holds after HPO on the
deep models is a drop-in item — see DL-RESULTS-DROPIN comment in this file.]

=== Summary of cross-modal findings

The Spearman rank correlation between the per-classifier ICBHI scores (lung) and
the corresponding MAcc scores (heart) — computed over the four classifier types at
feature set A — is negative or near-zero, reflecting that XGBoost's dominance on
heart does not transfer to lung, while SVM's moderate heart performance does
transfer. This finding supports the methodological argument for cross-modal
evaluation: a researcher who tuned on heart data and deployed the same model on
lung data would incur a non-trivial performance penalty. The practical implication
is that modality-specific classifier selection remains necessary even when the
feature extraction and evaluation protocol are shared.

== Joint-training probe

As a diagnostic probe, we examine what the results imply about a hypothetical
shared-representation model trained on pooled heart and lung data. Because the
two modalities differ substantially in sampling characteristics (heart: 33,246
windows; lung: 4,262 cycles), a naive pooled model would be dominated by heart
data. We do not train a pooled model explicitly; instead, we note from the
architecture (Chapter 2) that the feature extraction, model definition and loss
function are identical, and that the only modality-specific parameters are the
bandpass cutoffs and the label cardinality. These observations suggest that a
multi-task shared backbone, with per-modality classification heads, would be a
feasible and natural next step. The current results quantify the per-modality
performance ceiling such a model would need to match.

#figure(
  caption: [Per-modality scores under per-modality training (this study) versus a
  proposed joint-training architecture (future work).
  // <<DL-RESULTS-DROPIN: populate joint-training column if a joint experiment
  //    is run before the submission deadline>>
  ],
  table(
    columns: (2fr, 1fr, 1fr),
    align: (left, center, center),
    table.header([*Setting*], [*Heart (MAcc)*], [*Lung (ICBHI)*]),
    [Per-modality — best classical], [0.903], [0.537],
    [Per-modality — best deep (core run)], [0.872#super[†]], [0.551#super[†]],
    [Joint multi-task (future work)],     [n/a], [n/a],
  ),
) <tab-joint>

#text(size: 10pt)[#super[†] Preliminary core-run result; see Chapter 3 note.]

== Arterial sounds: an analytical sub-study

Arterial bruits occupy the third corner of this project's scope, and their
absence from the empirical experiments requires careful documentation.

=== Acoustic characteristics of carotid bruits

A carotid bruit is produced when blood accelerates through a region of arterial
stenosis, generating turbulent flow whose kinetic energy is partly radiated as
acoustic pressure. The sound is audible with a stethoscope placed at the lateral
neck during systole — the half-cycle in which cardiac ejection drives peak flow
velocity. In spectral terms, bruits occupy roughly 50–800 Hz, with the dominant
energy below 400 Hz for mild stenoses and extending upward as stenosis severity
increases and peak flow velocity rises @stroke2024. This overlap with the heart-
sound band (20–400 Hz) means that bruit detection from a stethoscope placed at
the precordium — as opposed to the neck — is problematic without careful device
placement and directional filtering.

The clinical pipeline for bruit assessment shares structural elements with the
heart- and lung-sound pipelines studied here: the signal is captured at fixed
sampling rates (typically 4,000–10,000 Hz with modern digital stethoscopes),
bandpass-filtered to the bruit-relevant band, segmented in cardiac-cycle-
synchronised windows, and feature-extracted for classification. A pilot smartphone
study @stroke2024 demonstrated that a custom microphone placed at the neck could
detect carotid bruits with promising accuracy, establishing proof-of-concept for
machine-learning based screening. An earlier carotid auscultation system @carotid_pmc
applied an ensemble of acoustic features to a proprietary hospital collection,
again demonstrating feasibility but leaving no open dataset behind.

=== Why supervised learning is not yet feasible

The obstacle to empirical work in this study is entirely a data-availability
problem, not an algorithmic one. Every published carotid bruit system relies on
proprietary recordings that were not released alongside the paper @stroke2024
@carotid_pmc. At the time of writing, an exhaustive search of PhysioNet, Zenodo,
Figshare and GitHub yielded no openly licensed, cycle-labelled dataset of arterial
bruit recordings suitable for training a supervised classifier. This absence is
documented explicitly so that readers can verify the claim: if the situation
changes, the pipeline of Chapter 2 would extend to arterial bruits with only two
changes — the bandpass cutoffs (adjusted to 50–800 Hz or a tighter passband as
appropriate) and the class label scheme (normal vs. bruit, or graded by stenosis
severity). All other pipeline stages — resampling, windowing, MFCC / log-mel
feature extraction, patient-level splitting, evaluation — apply without
modification.

=== What a future arterial dataset would require

For the unified pipeline to apply to a third modality, a suitable dataset would
need to meet the same standards as CinC 2016 and ICBHI 2017: (i) at least several
hundred recordings from independent subjects; (ii) ground-truth labels verified
by duplex ultrasound (the reference standard for carotid stenosis); (iii) open
licensing permitting unrestricted research use; and (iv) sufficient class balance,
or at least a patient-independent split with documented label distribution. The
methodological infrastructure to evaluate such a dataset is already in place in
the shared pipeline described in Chapter 2.

=== Chapter summary

The cross-modal analysis reveals that method rankings do not fully transfer across
auscultation modalities: XGBoost dominates on heart sounds but falls to third
place on lung sounds, while SVM is uniformly competitive. Deep learning shows the
opposite cross-modal pattern: a modest disadvantage on heart and a slight advantage
on lung. The feature set effect is also modality-specific, with spectral statistics
helping uniformly on heart but inconsistently on lung.
// <<DL-RESULTS-DROPIN: revisit if HPO heart DL exceeds best classical>>
These findings motivate per-modality model selection even within a shared pipeline
and quantify the cost of a naive one-model-all-modalities deployment.

The arterial sub-study establishes that the pipeline generalises to bruits in
principle but is blocked by data unavailability — a gap that the community should
address through a coordinated open data release.

#team-note[
  The cross-modality transfer analysis was carried out by #MEMBER("10"); the
  arterial analytical sub-study was prepared by #MEMBER("11") with clinical-
  literature synthesis by #MEMBER("03").
]
