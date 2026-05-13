// 02-ch1-litreview.typ — Chapter 1: Analytical literature review (Annex 5 §6).
// Fully drafted with original prose. Citations reference real sources in
// report/refs.bib. All external claims are cited [n]. Original synthesis
// throughout — no verbatim copying from any source.

#import "../helpers.typ": *

= Analytical review of existing approaches

This chapter positions the present study within the literature on auscultation-
sound classification. It proceeds from the physiological basis of the signals,
through the public datasets that enable reproducible benchmarking, through the two
dominant method families (classical feature engineering and deep learning), to the
cross-modality question that motivates our contribution. The chapter closes with an
explicit statement of the research gap and the novelty claim.

== Physiological basis of auscultation sounds

Auscultation has been a cornerstone of cardiovascular and pulmonary diagnosis since
Laënnec introduced the stethoscope in the early nineteenth century. The sounds
produced by the body carry information about the mechanical state of organs and
vessels in a form that is non-invasive, low-cost and immediately available at the
point of care. Understanding the acoustic origins of each modality is prerequisite
to designing appropriate signal-processing chains.

=== Heart sounds

Normal cardiac auscultation yields two dominant events per cycle. The first heart
sound (S1) coincides with closure of the mitral and tricuspid valves at the onset
of systole; the second heart sound (S2) arises from aortic and pulmonic valve
closure at the onset of diastole. Both events generate transient, broadband
pressure waves in the 20–400 Hz range @cinc2016. Pathological states alter this
pattern in characteristic ways: valve stenosis or regurgitation introduces
turbulent flow and produces sustained murmurs occupying the systolic or diastolic
interval; cardiomyopathies may produce third (S3) or fourth (S4) sounds; congenital
defects can produce fixed splitting of S2. Because the clinically diagnostic
content lies largely in the 20–400 Hz band, a bandpass filter with those cutoffs
suffices for most classification tasks while effectively removing low-frequency
movement artefact and ambient noise above the relevant range.

=== Respiratory sounds

Breath sounds arise from turbulent airflow in the tracheobronchial tree. Normal
turbulent flow produces the vesicular sound (soft, low-pitched inspiration); when
airways become abnormal, superimposed adventitious sounds appear. Crackles
(formerly "rales") are short, explosive, discontinuous sounds attributed to the
sudden reopening of collapsed alveoli or small airways; they are characteristic of
pneumonia, pulmonary fibrosis and heart failure. Wheezes are continuous, musical,
high-pitched sounds caused by oscillation of airway walls during partial
obstruction; they are the hallmark of obstructive diseases such as asthma and
chronic obstructive pulmonary disease (COPD). The ICBHI annotation scheme
distinguishes four cycle-level categories — normal, crackle, wheeze, and "both"
(simultaneous crackle and wheeze) — exactly reflecting this clinical taxonomy
@rocha2019. The diagnostically relevant frequency content spans roughly 200–1800 Hz,
with crackles peaking toward the upper end of that range and wheezes at 200–600 Hz.

=== Arterial bruits

A carotid bruit is a systolic murmur-like sound produced when blood accelerates
through a stenosed region of the carotid artery, generating turbulent flow audible
with a stethoscope placed over the neck. The acoustic signature overlaps with
cardiac murmurs in bandwidth but is shorter in duration and strongly modulated by
the stenosis severity. The clinical value of bruit detection as a screening marker
for carotid artery disease is well established; pilot systems have demonstrated
that even smartphone microphones can capture diagnostically relevant features
@stroke2024. However, no openly licensed, cycle-labelled dataset of carotid bruit
recordings has been released to date, which precludes a supervised-learning study
for this modality @carotid_pmc. We therefore treat arterial sounds analytically in
Chapter 4.

== Public datasets for auscultation research

The availability of large, openly licensed auscultation datasets is a prerequisite
for reproducible benchmarking. Two collections have emerged as the community
standards for heart and lung sounds respectively, and they are the primary data
sources for this study.

=== PhysioNet/CinC 2016 — heart sounds

The PhysioNet/Computing in Cardiology (CinC) Challenge 2016 released a collection
of 3,240 phonocardiogram recordings drawn from five independent sources (databases
A through E), totalling 764 unique subjects in the publicly released training split
@cinc2016 @liu2016. Recordings vary in length from approximately 5 to 120 seconds
and were acquired with consumer-grade electronic stethoscopes in real clinical and
home settings — a deliberate choice that introduces the ambient noise and recording
variability that any deployed system would face. Labels are binary: normal or
abnormal (with a small "unsure" category that the challenge recommended discarding
or treating as negative). The class distribution is markedly imbalanced: 71 % of
the training set is labelled normal and approximately 18 % abnormal. The official
evaluation metric is mean accuracy MAcc = (Se + Sp) / 2, which is robust to this
imbalance because it equally weights sensitivity (correct abnormal identification)
and specificity (correct normal identification). Top-3 challenge entries reached
MAcc around 0.86 @potes2016, setting a public benchmark for this task.

A key methodological point is that the challenge provided only a training set with
public labels; the private test set was withheld. Any researcher using this corpus
must therefore construct their own train/test split from the 764-subject training
pool, using strictly patient-level (not recording-level) partitioning to avoid
leakage @lones2021. This requirement distinguishes rigorous studies from a body
of literature that reports inflated numbers because multiple recordings from the
same patient appear on both sides of a random split.

The CirCor DigiScope 2022 database @circor2022 extends the phonocardiogram
landscape with 5,272 recordings from 1,568 subjects, adding graded murmur metadata
(timing, shape, pitch, grade) that enables finer-grained classification. We note
it here as a natural extension for future work but do not use it in this study
because the scope is binary normal/abnormal heart discrimination, where CinC 2016
is the established benchmark.

=== ICBHI 2017 — respiratory sounds

The International Conference on Biomedical and Health Informatics (ICBHI) 2017
Respiratory Sound Database @rocha2019 contains 920 recordings from 126 patients,
annotated at the cycle level to yield 6,898 labelled respiratory cycles. Recordings
were acquired with four device types at four different sampling rates (4,000 Hz;
10,000 Hz; 22,050 Hz; 44,100 Hz), introducing device-dependent acoustic
characteristics that a pre-processing resampling step must equalise. The cycle-
level class distribution is imbalanced: normal 52.8 %, crackle 27.0 %, wheeze
12.8 %, both 7.3 %. Unlike CinC 2016, ICBHI provides an official 60/40 patient-
independent split that partitions subjects into training and test sets. Adhering
to this published split is mandatory for results to be comparable with the
literature; using a random recording-level split instead can inflate the ICBHI
score by five to ten percentage points @cnn_lstm.

The ICBHI score — (Se + Sp) / 2 with sensitivity aggregated over abnormal cycles
and specificity measured on normal cycles — mirrors the CinC MAcc in structure,
making the two tasks directly comparable under a single metric family. This
alignment is a design choice exploited in the present study.

== Classical feature-based approaches

Prior to the deep-learning era, and still competitive for modest dataset sizes,
the dominant paradigm was to extract a fixed-dimensional acoustic feature vector
from each audio segment and to train a standard classifier on those vectors.

=== Feature representations

Mel-frequency cepstral coefficients (MFCC) have become the de-facto standard
feature for auscultation classification @mfcc_svm. MFCCs compress the spectral
envelope into a compact set of coefficients (typically 13–40) by applying a
mel-scale filterbank and a discrete cosine transform; appending first- and second-
order temporal derivatives (Δ and ΔΔ) captures the rate of change of spectral
content across frames, which is informative about the temporal dynamics of murmurs
and adventitious sounds. Summarising each coefficient's time series by its mean and
standard deviation across frames yields a fixed-length vector regardless of segment
duration, which is important because heart and lung recordings vary widely in length.

Additional spectral statistics — centroid (the "brightness" of the spectrum),
roll-off frequency (the frequency below which 85 % of energy lies), bandwidth,
zero-crossing rate and RMS energy — characterise the overall spectral shape and
energy and have been shown to improve discrimination between normal and abnormal
heart sounds when appended to the MFCC vector @xgboost_heart.

=== Classifiers

Support-vector machines (SVM) with an RBF kernel are the most cited classifier for
heart- and lung-sound classification using engineered features. The implicit kernel
mapping effectively models non-linear spectral boundaries, and SVMs perform well in
the regime of tens of thousands of training samples with hundreds of features. A
multi-domain MFCC study reported MFCC coefficients carrying 72 % discriminative
weight and an SVM reaching 91.67 % accuracy on a balanced evaluation set @mfcc_svm.
Random forests and gradient-boosted trees (XGBoost) are competitive alternatives:
XGBoost consistently matches or outperforms SVM at lower computational cost on
similar tasks @xgboost_heart, and its built-in feature-importance scores provide a
degree of interpretability. Logistic regression, though limited by its linearity,
serves as a transparent baseline and often achieves surprisingly strong results
because the MFCC feature space is already well-structured for linear separation.

A practical concern common to all classifiers is class imbalance. Applying SMOTE
(Synthetic Minority Oversampling Technique) inside the training fold — never
globally before the train/test split — counteracts the majority-class bias without
causing leakage @lones2021. This is a recurring correctness requirement in the
literature that many published studies violate by applying resampling globally.

== Deep-learning approaches

Deep learning on time–frequency image representations has become the dominant
paradigm for audio classification tasks. For auscultation sounds the input is
typically a log-mel spectrogram: a two-dimensional image whose axes are time and
mel-scaled frequency, and whose pixel values represent log-power. This
representation is rich enough to capture both the spectral shape of murmurs and
the temporal periodicity of adventitious sounds, and it enables transfer from
convolutional backbones pre-trained on natural images.

=== Convolutional networks

Compact convolutional networks (CNNs) with three to five convolutional layers
are a popular first deep-learning baseline because they run in minutes on a GPU
and in tens of minutes on a CPU. A CNN–LSTM hybrid trained on ICBHI log-mel
spectrograms reached an ICBHI score of 64.92 % on the official 60/40 split,
outperforming the classical SVM baselines of that study @cnn_lstm. The attention
mechanism of the LSTM over the CNN's temporal feature maps helps capture the
episodic structure of adventitious sounds. These results also demonstrate that
sequence models can further improve over purely spatial CNNs for sounds with
strong temporal dynamics.

=== Transfer learning

EfficientNet-B0, pre-trained on ImageNet, is a popular transfer backbone for
audio spectrograms because its approximately 4 million parameters provide strong
spatial feature extraction without requiring the large datasets needed to train from
scratch @ast_patchmix. The single-channel mel image is replicated across three
channels (or a projection layer is added) to match the backbone's expected input.
The typical fine-tuning protocol freezes the backbone for a few warm-up epochs,
then unfreezes it for end-to-end training; this is especially valuable when the
audio dataset is small (fewer than a few thousand recordings) and the backbone's
ImageNet priors provide beneficial inductive biases.

=== Audio transformers and recent advances

The Audio Spectrogram Transformer (AST) and its variants apply multi-head self-
attention to mel spectrogram patches, achieving state-of-the-art performance on
general audio benchmarks. Adapted to respiratory sound classification, patch-mix
contrastive learning with AST has reached ICBHI scores in the 68–70 % range
@ast_patchmix, representing the current frontier. However, these models require
large GPU budgets and often rely on data augmentation applied before the train/test
split, which can inflate published numbers. A careful reading of the experimental
setups reveals that many high-scoring papers use random or recording-level splits
rather than the official patient-independent partition, making their headline numbers
incomparable with studies that enforce the official protocol @lones2021.

== The cross-modality gap and research novelty

The overwhelming majority of auscultation machine-learning research is unimodal:
it addresses either heart sounds or lung sounds, but not both simultaneously. Yet
a clinician uses the same instrument and a closely related interpretive framework
across modalities; a computational tool that generalises across modalities is both
more practical and more theoretically interesting. Unified models that operate on
heart and lung sounds within a single pipeline are only beginning to be explored
@unified_heartlung @auscultabase. The emerging AuscultaBase foundation model
@auscultabase represents an ambitious recent attempt at unification, but it relies
on large proprietary data not available for the community benchmark tasks.

Arterial bruits occupy a still-harder position: despite growing clinical interest in
non-invasive carotid stenosis screening @stroke2024, no openly licensed dataset of
labelled bruit recordings has been released, and the systems that have been evaluated
rely entirely on proprietary collections @carotid_pmc. This data gap prevents
empirical supervised learning for arterial sounds and is likely to persist until a
major institution commits to an open release comparable to PhysioNet or ICBHI.

Methodological reviews of machine learning for biomedical audio @lones2021 identify
three recurring pitfalls that prevent results from being trustworthy or
reproducible: (i) recording-level rather than patient-level splitting; (ii) global
data augmentation applied before the split, leaking augmented test-set neighbours
into training; and (iii) inconsistent metric choices (raw accuracy on an imbalanced
dataset vs. the balanced (Se + Sp) / 2). All three pitfalls are present in a
significant fraction of the published auscultation literature.

=== Research gap and novelty claim

The research gap addressed by this study is therefore twofold. First, a single,
leakage-safe pipeline evaluated *identically* on both heart and lung sounds — same
protocol, same metric family, same model set — is comparatively rare. Second, the
question of whether method rankings transfer across auscultation modalities (does the
best heart classifier also win on lung sounds?) is largely unstudied, even though
the answer has practical implications for how much can be shared in a multi-modality
system.

This study's novelty is a *unified, reproducible, cross-modal comparison* under a
zero-leakage patient-level protocol: the same codebase, preprocessing chain, model
set and evaluation metric are applied to both CinC 2016 heart data and ICBHI 2017
lung data, enabling a direct, fair comparison of method rankings across modalities.
The contribution is intentionally framed as honest leakage-free baselines rather
than state-of-the-art scores, because the latter without the former is unreliable.
An analytical treatment of arterial bruits documents precisely why supervised
learning is not yet feasible and what would be required to extend the pipeline to
a third modality.

=== Chapter summary

The literature offers mature single-modality methods — MFCC-based classical
classifiers and CNN/transfer deep models — but few studies enforce leakage-safe
patient-level splits or compare methods across modalities under a single protocol.
The cross-modal ranking question is largely open. This motivates the unified
pipeline specified in Chapter 2, and the cross-modal analysis of Chapter 4.
