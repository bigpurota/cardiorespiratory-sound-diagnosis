// 05-ch4-novelty.typ — Chapter 4: Cross-modal analysis (novelty) + arterial
// sub-study. All numbers final (HPO+multi-seed DL, cross-modal transfer, joint).

#import "../helpers.typ": *

= Cross-modal analysis and arterial sub-study

This chapter develops the project's primary novel contribution: an analysis of how
method families transfer across auscultation modalities, combining the per-modality
Chapter 3 results with dedicated cross-modal transfer experiments (pretrained
encoder applied to the opposite modality) and a joint multi-task probe. It then
provides an analytical treatment of arterial sounds, for which no open dataset
exists, documenting the acoustic basis, the data-availability obstacle, and how
the pipeline would extend if the obstacle were removed.

== Transfer of method rankings across modalities

Because heart and lung models were evaluated under one protocol and one metric
family, their rankings can be compared directly. We examine the question in three
ways: (i) the ordinal ranking of classifiers within each modality, (ii) the
relative advantage of feature set B over feature set A, and (iii) the relative
standing of classical versus deep models.

=== Classifier ranking

For heart sounds the ranking (best to worst) by primary metric is:
XGBoost-B (0.903) > XGBoost-A (0.879) > SVM-B (0.869) > SVM-A (0.859) >
RF-B (0.831) > LogReg-B (0.825) > RF-A (0.817) > LogReg-A (0.794).
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
lung training set (4,262 cycles versus 33,246 heart windows). These rank inversions
are the central qualitative observation of the cross-modal analysis; with only four
classifier types available they are reported as a descriptive pattern, not a
statistically established law.

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

With HPO-tuned hyperparameters and three-seed averaging, the picture on heart
sounds is more nuanced than the core run suggested. EfficientNet-B0 reaches
MAcc = 0.898 ± 0.008 (Se = 0.936, Sp = 0.860), comparable with the
best classical model (XGBoost-B, MAcc = 0.903): the 0.005 difference is below
EfficientNet-B0's ±0.008 seed standard deviation, though the classical model remains
numerically best. SmallCNN reaches MAcc = 0.871 ± 0.009. The clear classical
advantage seen in the untuned core run thus shrinks to a margin within seed noise
after equivalent optimisation effort, without being reversed.

On lung sounds, the deep models achieve ICBHI = 0.540 ± 0.022 (CNN) and
0.555 ± 0.016 (EfficientNet-B0), placing them marginally ahead of but practically
indistinguishable from the best classical model (SVM-B, 0.537). Both modalities
therefore converge on the same finding: with proper HPO, classical and deep methods
occupy the same performance tier on these tasks and dataset scales.

=== Summary of cross-modal findings (classical)

The Spearman rank correlation @spearman1904 between the per-classifier ICBHI scores
(lung) and the corresponding MAcc scores (heart) — computed over the four classical
classifier types at feature set A — is $rho = 0.60$ ($p = 0.40$, $n = 4$). This
represents a moderate positive correlation: classifiers that rank higher on heart
tend to rank higher on lung, but the relationship is not significant at conventional
thresholds given the small sample size. Concretely, XGBoost's dominance on heart
does not fully transfer to lung (it falls to third), while SVM is consistently
competitive on both modalities. This finding supports the methodological argument
for cross-modal evaluation: a researcher who tuned exclusively on heart data and
deployed the same model on lung data would incur a non-trivial performance penalty.

== Deep cross-modal transfer and joint multi-task experiments

To probe whether the shared log-mel pipeline enables feature transfer between
modalities, we ran three additional deep-learning experiments beyond the per-modality
training of Chapter 3: (i) direct transfer, where an encoder pre-trained on one
modality is applied to the other without re-training; (ii) joint multi-task
training, where a single shared encoder with two task-specific classification
heads is trained simultaneously on both modalities; and (iii) in-domain baselines
for reference. The cross-modal heat-map in @fig-cross-modal-heatmap summarises the
six source→target combinations.

#figure(
  image("../../results/figures/cross_modal_heatmap.png", width: 78%),
  caption: [Cross-modal transfer heat-map for the EfficientNet-B0 model. Each cell
  shows the primary metric (MAcc for heart; ICBHI Score for lung) when a model
  trained on the source modality (rows) is evaluated on the target modality
  (columns): the diagonal is in-domain, off-diagonal cells are direct transfer, and
  the bottom row is the joint multi-task model. The corresponding SmallCNN values
  appear in @tab-joint.],
) <fig-cross-modal-heatmap>

The headline finding is that cross-modal transfer is strongly asymmetric.
Heart→lung transfer yields only ICBHI = 0.524 (CNN) and 0.526 (EfficientNet-B0),
which is below the lung in-domain baseline (0.559 / 0.578) — a negative or
near-neutral transfer. By contrast, lung→heart transfer yields MAcc = 0.854
(CNN) and 0.876 (EfficientNet-B0), close to the heart in-domain baseline
(0.861 / 0.885) — a strong positive transfer. These in-domain baselines are
single-seed reference runs from the transfer experiment, slightly below the
multi-seed HPO means of Chapter 3 (heart CNN 0.871, EfficientNet 0.898); transfer
and in-domain were run under the identical single-seed protocol, so the comparison
is internally consistent and is not sensitive to which baseline is chosen — the
direction of the effect (near-in-domain one way, a drop the other) holds against the
multi-seed means as well. This asymmetry suggests that
lung-pretrained spectral features capture patterns broadly useful for binary
heart classification, whereas heart-pretrained features (tuned for the binary
normal/abnormal boundary in phonocardiograms) do not generalise to the
four-class respiratory task.

The joint multi-task model (shared encoder, two classification heads) achieves
MAcc = 0.844 / 0.840 for heart (CNN / EfficientNet-B0) and ICBHI = 0.565 / 0.567
for lung. The joint model roughly preserves both modalities: it holds lung
performance essentially level with per-modality training (0.565 vs. 0.559 for CNN —
a difference well inside the ±0.022 lung seed spread, so not a genuine improvement)
and shows a slight heart drop (0.844 vs. 0.861 for CNN), consistent with the known
multi-task trade-off where the shared encoder cannot simultaneously optimise for
both objectives at a fixed capacity.

@tab-joint collects all per-setting scores for direct reference.

#figure(
  caption: [Per-modality scores under in-domain, transfer and joint-training
  settings. In-domain scores are single-seed reference runs (comparable to the
  Chapter 3 mean); see Chapter 3 for multi-seed HPO values.
  ],
  table(
    columns: (2.2fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    table.header(
      [*Setting*], [*Heart CNN*], [*Heart EffNet*], [*Lung CNN*], [*Lung EffNet*]),
    [In-domain],                  [0.861], [0.885], [0.559], [0.578],
    [Transfer heart→lung],        [0.524], [0.526], [—],     [—],
    [Transfer lung→heart],        [—],     [—],     [0.854], [0.876],
    [Joint multi-task],           [0.844], [0.840], [0.565], [0.567],
    [Per-modality classical best],[0.903], [—],     [0.537], [—],
  ),
) <tab-joint>

== Audio Spectrogram Transformer: methodological extension

As a methodological extension of the deep-learning pipeline, a pretrained Audio
Spectrogram Transformer (AST) @gong2021ast — originally trained on AudioSet — was
integrated into the shared pipeline. The AST's forward pass and training loop were
verified to run correctly within the project's data-loading and evaluation framework.
However, a complete fine-tune-and-evaluation run was not finalised within the
project's time budget; producing reliable multi-seed results would have required
additional GPU compute that was not available before the submission deadline. AST
is therefore presented as a methodological extension rather than a headline result:
the integration work is complete, the code is available in the repository, and the
experiment can be run to completion given additional compute resources.

== Arterial sounds: an analytical sub-study

Arterial bruits occupy the third corner of this project's scope, and their
absence from the empirical experiments requires careful documentation.

=== Acoustic characteristics of carotid bruits

A carotid bruit is produced when blood accelerates through a region of arterial
stenosis, generating turbulent flow whose kinetic energy is partly radiated as
acoustic pressure. The acoustic basis of this phenomenon was established by the
foundational phonoangiography work of Lees and Dewey @lees1970, who showed that
the spectrum of a bruit recorded at the skin surface matches that of laboratory
turbulent pipe flow and can be related quantitatively to arterial diameter and
flow velocity. The sound is audible with a stethoscope placed at the lateral
neck during systole — the half-cycle in which cardiac ejection drives peak flow
velocity. In spectral terms, bruits occupy roughly 50–800 Hz, with the dominant
energy below 400 Hz for mild stenoses and extending upward as stenosis severity
increases and peak flow velocity rises. Duncan et al. @duncan1975
demonstrated that this upward spectral shift is systematic enough to estimate the
residual lumen diameter directly from the bruit spectrum — a quantitative,
feature-based reading of an auscultation signal that closely anticipates the
classification approach pursued in this project for heart and lung sounds. This
overlap with the heart-sound band (20–400 Hz) means that bruit detection from a
stethoscope placed at the precordium — as opposed to the neck — is problematic
without careful device placement and directional filtering.

The clinical motivation for automating this assessment is strong. Manual
auscultation for carotid bruits has only modest, operator-dependent diagnostic
accuracy: the Rational Clinical Examination review by Sauvé et al. @sauve1993
found the presence of a bruit to be at best a weak-to-moderate predictor of
high-grade stenosis, with sensitivity and specificity that vary widely between
examiners. Yet an audible bruit carries genuine prognostic weight — a meta-analysis
pooling 17,295 patients reported that carotid bruits roughly double the rate of
cardiovascular death and myocardial infarction @pickett2008. An objective,
reproducible acoustic classifier could therefore both reduce inter-observer
variability and surface a clinically actionable risk marker, which is precisely
the contribution that the feature-engineering and deep-learning pipelines compared
in this report are designed to make for the heart and lung modalities.

The clinical pipeline for bruit assessment shares structural elements with the
heart- and lung-sound pipelines studied here: the signal is captured at fixed
sampling rates (typically 4,000–10,000 Hz with modern digital stethoscopes),
bandpass-filtered to the bruit-relevant band, segmented in cardiac-cycle-
synchronised windows, and feature-extracted for classification. A recent smartphone
study @stroke2024 demonstrated on-device deep-learning detection of carotid stenosis
from neck auscultation, establishing proof-of-concept for machine-learning-based
screening. An earlier computer-assisted auscultation system @carotid_pmc was built
specifically to acquire carotid vascular sounds reproducibly — a prerequisite for
assembling a database of pathological bruits — yet, like the smartphone work, it
released no open dataset.

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
feature extraction, grouped (subject-level) splitting, evaluation — apply without
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

The cross-modal analysis delivers three findings. First, classical method rankings
do not fully transfer: XGBoost dominates on heart sounds but falls to third place
on lung, while SVM is uniformly competitive. The Spearman rank correlation across
the four classical methods is $rho = 0.60$ ($p = 0.40$), a moderate positive association
that falls short of significance at the small sample size. Second, deep cross-modal
transfer is asymmetric: lung-pretrained features transfer well to heart (near
in-domain performance), but heart-pretrained features do not transfer to lung.
Joint multi-task training preserves both modalities at the cost of a slight heart
drop, consistent with shared-capacity constraints. Third, with HPO-tuned deep
models, the deep-vs-classical gap on heart closes to within noise, while lung
remains equally hard for all method families. These findings motivate per-modality
model selection even within a shared pipeline and quantify the cost of a naive
one-model-all-modalities deployment.

The AST extension verifies that transformer-based audio models integrate
cleanly into the pipeline; finalising the fine-tune is reserved for future work.

The arterial sub-study establishes that the pipeline generalises to bruits in
principle but is blocked by data unavailability — a gap that the community should
address through a coordinated open data release.
