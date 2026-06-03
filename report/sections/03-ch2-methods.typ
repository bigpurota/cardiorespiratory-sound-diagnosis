
#import "../helpers.typ": *

= Methodology

This chapter specifies the experimental pipeline in enough detail to reproduce
every reported number. The design principle is _one shared, configuration-driven
codebase_ applied to both modalities through a common set of stages, with all
randomness fixed and all dependency versions pinned. The pipeline is deliberately
uniform across modalities; only two protocol parameters differ between heart and
lung, the grouping granularity and the test fraction, and both differences are
forced by the datasets rather than chosen freely (Section 2.1). The pipeline is organised as independent stages
(ingest, preprocess, segment, feature extraction, grouped split, training
and evaluation), each writing its output to disk so any single stage is
re-runnable in isolation.

== Datasets and leakage-safe splits

Heart sounds are taken from the PhysioNet/CinC 2016 training databases (subsets A
through E); respiratory sounds from the ICBHI 2017 database. The two raw
collections are indexed into a single recording-level manifest carrying, for each
recording, a grouping identifier (the patient where known, the recording otherwise),
a class label, the modality and the file path; a
separate cycle-level table records the annotated respiratory cycles of ICBHI with
their start and end times and one of four labels.

The single most important correctness requirement is _grouped_ (not random)
partitioning, so that no recording (and, where subject identity is recoverable, no
patient) appears on both sides of the split; this prevents the data leakage that
inflates many published results. For heart sounds the split is a seeded
`GroupShuffleSplit` (test fraction $0.20$, random state $42$) computed _within_
databases A–E. The public CinC 2016 training release does not publish a complete
recording-to-subject mapping, so the grouping key is the recording identifier: this
guarantees that no $3$-second window of a recording can fall on both sides of the
split (the dominant leakage risk for windowed audio) but the resulting partition
is recording-level rather than strictly subject-level (a limitation stated in the
Conclusion). The never-released private test set is not touched. For lung sounds,
where ICBHI publishes patient identifiers, the official $60\/40$ patient-independent
(genuinely subject-level) split is adopted and then _repaired_: two patients whom the official file places
on both sides of the split (identifiers $156$ and $218$) have all of their recordings
forced to the training side, so that every patient lands on exactly one side. If
the official split cannot be fetched and validated, the pipeline falls back to a
seeded patient-level `GroupShuffleSplit` at the same $60\/40$ ratio. Either way the
provenance of the split is logged.

Before any model is trained, a reusable assertion verifies that the training and
test patient sets are disjoint:

```python
def assert_no_patient_leakage(train_ids, test_ids):
    train, test = set(map(str, train_ids)), set(map(str, test_ids))
    overlap = train & test
    assert not overlap, f"PATIENT LEAKAGE: {len(overlap)} shared ids ..."
```

This check runs at the start of every experiment script, and its
`[leakage-check OK] ... overlap=0` line is recorded in the experiment logs.

== Preprocessing

All audio is loaded at its native sampling rate and resampled to a single common
rate of $4000$ Hz for both modalities, so that heart and lung sounds share one
feature space (heart recordings are upsampled from their native $2000$ Hz; lung
recordings are downsampled from mixed native rates). At $4000$ Hz the Nyquist limit
of $2000$ Hz comfortably covers the diagnostic bands of both modalities.

Each signal is then band-limited with a zero-phase Butterworth bandpass applied
through second-order sections. The cutoffs are modality-specific: $20"–"400$ Hz for
heart sounds (retaining S1/S2 energy and murmurs while removing sub-cardiac
drift) and $200"–"1800$ Hz for lung sounds (the band of crackles and wheezes). The
filter is fourth order and is applied with forward–backward filtering to preserve
waveform timing; for segments too short for the forward–backward padding length
the pipeline falls back to a single causal pass so that very short inputs remain
finite and same-length. Finally each clip is peak-normalised to the range
$[-1, 1]$; near-silent clips are left unchanged to avoid division blow-up. No
global normaliser is fitted at this stage; feature-space standardisation is
fitted on the training fold only (Section 2.5).

== Segmentation

Heart recordings are sliced into fixed $3.0$-second windows ($12 thin 000$ samples at
$4000$ Hz). Training recordings use a $1.5$-second hop ($50%$ overlap) to increase the
number of training windows, whereas test recordings use a $3.0$-second hop (no
overlap); this affects only the denominator of the recording-level majority vote,
not correctness. A ragged final window is dropped, and a window in which more than
$80%$ of samples are near-zero is discarded as silence. Respiratory sounds are not
windowed but segmented at the _annotated cycle_ boundaries supplied with ICBHI;
each cycle is zero-padded (or trimmed) to a fixed $3.0$-second length so that the
downstream feature extractor sees a uniform input.

== Feature extraction

Two parallel representations are produced from each fixed-length segment.
Feature extraction is implemented with librosa 0.11.0 @librosa for the classical
path and torchaudio for the spectrogram path.

_Classical features._ MFCC extraction follows the standard mel-filterbank + DCT
formulation @mfcc_standard. From each segment we compute $40$ mel-frequency cepstral
coefficients together with their first- and second-order temporal deltas; each of
these three blocks is summarised across frames by its mean and its standard
deviation, yielding a $240$-dimensional vector (feature set A). An extended set
(set B, $250$-dimensional) appends the mean and standard deviation of five spectral
statistics: spectral centroid, roll-off, bandwidth, zero-crossing rate and
RMS energy. For respiratory cycles the segment is padded to $3.0$ seconds _before_
the cepstral transform; this is mandatory, because a raw short cycle yields too
few analysis frames for the delta operator and would otherwise raise an error.
Every emitted vector is asserted to have the expected dimension and to be free of
non-finite values.

_Deep-learning features._ The same fixed $3.0$-second segment is transformed into a
$64 times 128$ log-mel "image": a mel spectrogram (power spectrogram, $64$ mel
bands, $512$-point FFT, hop $94$) followed by amplitude-to-decibel conversion with an
$80$ dB dynamic range. The hop of $94$ samples on a $12 thin 000$-sample window produces
exactly $128$ time frames. The mel band limits match the per-modality bandpass
cutoffs ($20"–"400$ Hz for heart, $200"–"1800$ Hz for lung); a $512$-point FFT is used for
both modalities to avoid an empty mel-filterbank warning that the narrow heart
band triggers at smaller FFT sizes.

== Models and training protocol

_Classical models._ Both feature sets are fed to four classifiers implemented in
scikit-learn @sklearn: logistic regression, an RBF-kernel support-vector machine
@cortes1995, a random forest @breiman2001 and gradient boosting (XGBoost @chen2016).
Standardisation is fitted on the training fold only, inside a single scikit-learn
pipeline, so no test-set statistics leak into training. Hyper-parameters are tuned
by patient-grouped grid-search cross-validation for the two models where they
materially affect performance (the SVM's $C$ and $gamma$, and XGBoost's tree depth
and estimator count); logistic regression and the random forest are left at fixed,
standard scikit-learn defaults, so the classifier comparison slightly favours the
two tuned models and should be read with that asymmetry in mind. Class imbalance is
handled by per-model class weighting applied inside the training fold only: balanced class weights for
logistic regression, the SVM and the random forest, and a positive-class weight
(binary heart) or balanced sample weights (four-class lung) for XGBoost. No synthetic
oversampling is applied, which removes any risk of synthesising minority samples
across the split.

_Deep models._ All deep models are implemented in PyTorch @pytorch. The compact
convolutional network (SmallCNN) consists of four
convolution–batch-normalisation–ReLU–max-pool blocks (baseline channel widths
$1 arrow.r 16 arrow.r 32 arrow.r 64 arrow.r 128$, treated as a tunable hyperparameter),
an adaptive average pool, and a head with dropout (at least $0.3$) before a linear
classifier. The transfer-learning model is
an EfficientNet-B0 backbone @efficientnet pre-trained on ImageNet @imagenet
(approximately $4.0$ million parameters), with the single-channel log-mel image
lifted to the three-channel $224 times 224$ input the backbone expects via
channel replication; an optional frozen-backbone mode trains only the classifier
head as a CPU-feasible fallback. Deep models are trained with the Adam optimiser
and a class-weighted cross-entropy loss, with early stopping on the validation
primary metric, a fixed training-time cap, and best-checkpoint
restoration; a train-versus-validation learning-curve figure is saved for each run
to check for overfitting. Deep-model hyperparameters (learning rate, batch size,
weight decay, dropout, augmentation strength, the class-imbalance sampling mode and,
for the SmallCNN, the convolutional channel widths) were selected by a $128$-trial
bounded random search ($32$ trials per modality–model combination) that ranked
configurations on the validation primary metric only, never on the test set. The
reported heart SmallCNN therefore uses the wider $32 arrow.r 64 arrow.r 128 arrow.r 256$
channel variant selected by the search, whereas the lung SmallCNN retained the
baseline widths; every final deep-learning result is reported as the mean ± standard
deviation over three seeds ($1, 2, 42$).

== Evaluation

Both tasks are scored with the same balanced figure of merit, the arithmetic mean
of sensitivity ($"Se"$) and specificity ($"Sp"$):

$ "MAcc" = ("Se" + "Sp") / 2, $ <eq-balanced>

where sensitivity and specificity are the true-positive and true-negative rates of
the confusion matrix,

$ "Se" = "TP" / ("TP" + "FN"), wide "Sp" = "TN" / ("TN" + "FP"), $ <eq-sesp>

and $"TP"$, $"FN"$, $"TN"$, $"FP"$ in @eq-sesp are the true-positive,
false-negative, true-negative and false-positive counts. @eq-balanced
weights the two classes equally irrespective of their prevalence, which is exactly
why it is robust to the heavy class imbalance of both datasets and is the official
metric of each challenge.

For heart sounds the primary metric is this _mean accuracy_ (MAcc), with $"Se"$ the
recall on the abnormal class and $"Sp"$ the recall on the normal class, the official
CinC 2016 metric. Per-window predictions are first reduced to one prediction per
recording by _majority vote_, because the recording is the clinically meaningful
unit, and @eq-balanced is then applied at the recording level; sensitivity,
specificity, macro-F1, accuracy and, where a recording-level score is available,
AUC-ROC are reported alongside.

For lung sounds the primary metric is the official _ICBHI score_. It uses the same
balanced score (@eq-balanced), but $"Se"$ is pooled over the three abnormal cycle classes
(crackle, wheeze, both) while $"Sp"$ is measured on the normal class; the score
therefore captures normal-versus-abnormal discrimination rather than four-way
accuracy. Per-class sensitivities and macro-F1 are reported in support.

Every model is additionally checked against a degenerate "predict-one-class"
failure (predictions must occupy at least two confusion-matrix columns), and a
confusion-matrix figure is saved for each run. Volumetric characteristics
(sample counts per split, model parameter counts and training wall-clock times)
are logged for the report's volumetric table (Annex B).

== Reproducibility

A single configuration module is imported first by every script; its import side
effect seeds Python's `random`, NumPy and PyTorch generators with the value $42$
and disables non-deterministic cuDNN behaviour. The software stack is pinned:
Python 3.11, librosa 0.11.0, scikit-learn 1.8.0, XGBoost 3.2.0, PyTorch 2.11.0
with torchaudio 2.11.0, timm ($gt.eq 1.0$), imbalanced-learn 0.14.1, with the exact
versions captured in a committed `requirements.txt`. The splits are
written to disk and reloaded rather than regenerated, so that the partition is
identical across runs. The full pinned environment is reproduced in Annex B.
The classical pipelines run on CPU; the deep models (including the multi-seed,
hyperparameter search and cross-modal runs) were trained on an NVIDIA A100 GPU,
with each configuration kept small enough to remain CPU-feasible if required.

=== Chapter summary

The methodology is a single configuration-driven pipeline applied uniformly to
both modalities, with leakage-safe grouped splits and an explicit zero-leakage
assertion, modality-matched preprocessing and features, a common
classical-versus-deep model set, and a shared balanced metric (@eq-balanced). This
uniformity is what makes the cross-modal comparison of Chapter 4 meaningful.
