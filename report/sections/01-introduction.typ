
#import "../helpers.typ": *

#heading(numbering: none, outlined: true)[Introduction]

Auscultation (listening to the internal sounds of the body with a stethoscope)
remains one of the oldest and most accessible diagnostic techniques in medicine.
Despite the spread of imaging, the acoustic signatures of the heart, lungs and
large arteries still carry clinically decisive information: a murmur can reveal a
valvular defect, a crackle or wheeze can localise pulmonary pathology, and a
bruit can betray a narrowed carotid artery. Yet the skill of interpreting these
sounds is unevenly distributed, hard to acquire and subjective. Automated
analysis of auscultation recordings therefore promises a low-cost screening and
triage aid, particularly where specialist expertise is scarce.

This project studies, empirically and comparatively, how well machine learning
can support such a screening task across more than one auscultation modality at
once. We deliberately frame the contribution as a screening and triage aid rather
than a diagnostic instrument: the goal is to flag recordings that warrant expert
review, not to replace the clinician.

#heading(level: 3, numbering: none, outlined: false)[Relevance]

Three factors make the problem timely. First, large, openly licensed datasets of
heart and lung sounds now exist, removing the historical barrier of data
scarcity for these two modalities. Second, the methodological pitfalls of audio
classification (chiefly data leakage from splitting recordings rather than
patients) are now well documented, so a study that avoids them produces results
that are genuinely comparable across methods. Third, most publicly available work
treats each modality in isolation; a single pipeline evaluated identically on
heart and lung sounds is comparatively rare, and arterial sounds are almost
entirely unaddressed for want of open data.

#heading(level: 3, numbering: none, outlined: false)[Practical significance]

A reliable acoustic screening tool would let non-specialists capture a short
recording and obtain an objective "normal / needs review" indication. In primary
care, remote regions and tele-medicine this could shorten the path to a
specialist for patients who need one while reducing unnecessary referrals.
Because the same software backbone serves several modalities, the practical cost
of deploying it across heart and lung screening is shared rather than duplicated.

#heading(level: 3, numbering: none, outlined: false)[Innovation]

The novelty of this study is a _unified, leakage-safe, cross-modal comparison
pipeline_. A single codebase, governed by per-modality configuration files,
applies the same preprocessing, the same leakage-safe grouped partitioning with an
explicit zero-leakage assertion, the same metric family and the same set of
classical and deep models to both heart and lung sounds. This lets us ask a
question that single-modality studies cannot: do the method families that win on
one auscultation modality also win on another? We further contribute an honest
analytical treatment of arterial bruits, documenting precisely why supervised
learning is not yet feasible there and how the same pipeline would extend if data
became available. The emphasis throughout is on reliability and reproducibility (honest baselines under a
leakage-free protocol) rather than on chasing
state-of-the-art accuracy through aggressive tuning.

Concretely, the work makes four contributions:

+ a single leakage-safe pipeline that applies identical preprocessing,
  grouped partitioning (with an explicit zero-leakage assertion), feature
  extraction and modelling to two auscultation modalities, enabling a
  like-for-like comparison that isolated single-modality studies cannot offer;
+ a head-to-head comparison of the classical family (MFCC features with logistic
  regression, support-vector machines, random forests and gradient boosting)
  against the deep family (a compact convolutional network and a hyperparameter-tuned,
  three-seed EfficientNet-B0) under one class-imbalance-robust metric per task;
+ the empirical finding that method and feature-set choices transfer
  _asymmetrically_ between modalities (choices learned on lung sounds carry over
  to heart sounds, whereas the reverse transfer is near-neutral), a result
  invisible to studies that consider one modality at a time; and
+ an analytical arterial sub-study that states plainly why supervised learning
  is not yet possible for carotid bruits and specifies what an open dataset would
  need to provide.

#heading(level: 3, numbering: none, outlined: false)[Research objectives]

The work pursues the following objectives:

+ to assemble a reproducible, leakage-safe experimental pipeline shared by the
  heart and lung modalities, with fixed seeds and pinned dependencies;
+ to extract classical acoustic features (MFCC with deltas and spectral
  statistics) and to train and evaluate logistic regression, support-vector
  machines, random forests and gradient boosting on both modalities;
+ to train deep-learning models (a compact convolutional network and an
  EfficientNet-B0 transfer model) on log-mel spectrograms of the same data;
+ to evaluate every model with the official, class-imbalance-robust metric of
  each task and with confusion matrices, aggregating to the clinically
  meaningful recording level for heart sounds;
+ to analyse how method rankings transfer across the two modalities, and to
  situate arterial-sound diagnosis analytically; and
+ to deliver an original, fully cited research report conforming to the
  department's structural and formatting requirements.

The remainder of the report is organised as follows. Chapter 1 reviews the
physiological basis, the datasets and the prior classical and deep-learning
approaches, and identifies the research gap. Chapter 2 specifies the methodology
in full. Chapter 3 reports the experimental results for heart and lung sounds.
Chapter 4 presents the cross-modal analysis and the arterial sub-study. The
Conclusion summarises the findings against each objective and states the
limitations and directions for future work.

#heading(level: 3, numbering: none, outlined: false)[Project group and contribution]

This research project was carried out within a project group whose sole member is
the author. Accordingly, all stages of the work were performed by the author,
Tsember Andrei Alekseevich (group БПАД244): problem formulation and the analytical
literature review; acquisition and exploratory analysis of the PhysioNet/CinC 2016
(heart) and ICBHI 2017 (lung) datasets; preprocessing, segmentation and the
construction of leakage-safe grouped splits; classical feature engineering
and the logistic-regression, support-vector-machine, random-forest and
gradient-boosting experiments; the convolutional-network and EfficientNet-B0
deep-learning experiments, including the hyperparameter search and multi-seed
evaluation, together with the code-level integration (but not the full
fine-tuning, which was deferred for want of compute) of an Audio Spectrogram
Transformer; the cross-modal transfer analysis and the arterial analytical
sub-study; and the preparation of this report under the department's structural
and formatting requirements.
