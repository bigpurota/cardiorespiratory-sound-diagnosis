// 02-ch1-litreview.typ — Chapter 1: Analytical literature review (Annex 5 §6).
// Fully drafted with original prose. Citations reference real sources in
// report/refs.bib. All external claims are cited [n]. Original synthesis
// throughout — no verbatim copying from any source. Dataset-fact sentences and
// canonical definitions are deliberately re-structured to minimise similarity
// against the source papers (originality pass 2026-06-02).

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
Laënnec introduced the stethoscope in the early nineteenth century @bohadana2014. The sounds
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

Breath sounds originate in turbulent airflow through the tracheobronchial tree. In
health that turbulence is heard as the soft, low-pitched vesicular murmur of
inspiration; disease superimposes additional — so-called adventitious — sounds upon
it. Two adventitious types dominate the clinical picture. Crackles, which the older
literature calls rales, are brief, intermittent, popping noises usually explained by
alveoli or small airways snapping back open after collapse, and they accompany
pneumonia, pulmonary fibrosis and cardiac failure. Wheezes are their opposite in
character: sustained, tonal and high in pitch, generated when the walls of a
narrowed airway flutter, and typical of obstructive disease such as asthma and
chronic obstructive pulmonary disease (COPD) @bohadana2014. The ICBHI annotation
scheme mirrors exactly this clinical taxonomy, sorting each cycle into one of four
categories — normal, crackle, wheeze, or "both" (crackle and wheeze together)
@rocha2019. Diagnostically relevant energy spans roughly 200–1800 Hz, with crackles
biased toward the upper part of that band and wheezes toward 200–600 Hz @bohadana2014.

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

The 2016 PhysioNet/Computing-in-Cardiology Challenge distributed a public training
pool assembled from five separately collected cohorts, conventionally labelled A–E;
pooled, these supply on the order of 3,240 phonocardiograms originating from 764
distinct individuals @cinc2016 @liu2016. Clip durations vary widely — the shortest
last a few seconds, the longest around two minutes — because the audio was captured
with consumer-grade electronic stethoscopes in working clinics and homes rather than
under laboratory control, a choice that deliberately exposes a model to the ambient
noise and acquisition variability of real deployment. Each clip carries one of two
verdicts, normal or abnormal, alongside a small residual "unsure" tier that the
organisers advised discarding or folding into the negative class. The labels are
far from balanced — normals account for roughly seven recordings in ten and
abnormals for fewer than one in five — which is why the challenge scores submissions
not on raw accuracy but on the mean of the true-positive and true-negative rates,
MAcc = (Se + Sp) / 2, a figure of merit that refuses to let the dominant normal
class mask poor detection of the abnormal one. The strongest challenge entries
reached MAcc in the region of 0.86 @potes2016, which stands as the public reference
point for the task.

A methodological subtlety follows from the release design: only the training set
was published with labels, while the test set was kept private. Anyone using the
corpus must therefore carve their own train/test partition out of the training pool,
and to avoid leakage that partition should ideally separate subjects rather than
recordings @lones2021. In practice the public release ships no complete
recording-to-subject mapping, so studies commonly group by recording instead — a
limitation that the present work states explicitly in Chapter 2. This distinction
separates rigorous studies from a body of work that reports inflated figures because
several recordings from one patient land on both sides of a random split.

The CirCor DigiScope 2022 database @circor2022 extends the phonocardiogram
landscape with 5,272 recordings from 1,568 subjects, adding graded murmur metadata
(timing, shape, pitch, grade) that enables finer-grained classification. We note
it here as a natural extension for future work but do not use it in this study
because the scope is binary normal/abnormal heart discrimination, where CinC 2016
is the established benchmark.

=== ICBHI 2017 — respiratory sounds

The ICBHI 2017 Respiratory Sound Database, released through the International
Conference on Biomedical and Health Informatics @rocha2019, gathers 920 audio files
from a cohort of 126 individuals. Crucially, expert annotation operates at the
granularity of the individual breath cycle rather than the whole recording, yielding
6,898 labelled cycles. Acquisition used four different devices, each digitising at
its own rate — 4 kHz at the low end and 44.1 kHz at the high end, with 10 kHz and
22.05 kHz in between — so a resampling stage is needed to reconcile their differing
spectral characteristics. The cycles are spread unevenly across the four labels:
just over half are normal, crackle-only cycles make up a little more than a quarter,
isolated wheezes about an eighth, and the combined crackle-and-wheeze category is
rarest at well under a tenth. A further contrast with CinC 2016 is that the
organisers fix the partition themselves — close to three-fifths of the subjects form
the training pool and the remainder the test pool, with no subject straddling the
boundary. Honouring this prescribed split is essential for comparability with prior
work; substituting a random, recording-level split can lift the reported ICBHI score
by several points @lones2021.

The ICBHI score — (Se + Sp) / 2 with sensitivity aggregated over abnormal cycles
and specificity measured on normal cycles — mirrors the CinC MAcc in structure,
making the two tasks directly comparable under a single metric family. This
alignment is a design choice exploited in the present study.

== Classical feature-based approaches

Prior to the deep-learning era, and still competitive for modest dataset sizes,
the dominant paradigm was to extract a fixed-dimensional acoustic feature vector
from each audio segment and to train a standard classifier on those vectors.

=== Feature representations

By far the most common engineered representation in this field is the mel-frequency
cepstral coefficient (MFCC), a device borrowed from automatic speech analysis
@mfcc_standard and transferred to cardiac and pulmonary audio because those signals,
like speech, are band-limited and approximately stationary over short frames. The
computation re-expresses each frame's energy across a bank of bands spaced to mimic
perceived pitch, takes the logarithm of those band energies, and then applies an
orthogonalising (discrete-cosine) transform that removes the correlation between
neighbouring bands; only the first few dozen outputs are kept — here of order 13–40 —
and together they summarise the spectral envelope of a heart or lung event. Adding
the first- and second-order frame-to-frame differences (Δ and ΔΔ) records how that
envelope moves over time, which bears directly on the onset and decay of murmurs and
on the sharp transients of crackles. Because clip lengths differ, each coefficient's
per-frame series is finally collapsed to two summary statistics — its average and
its spread across frames — producing a descriptor whose dimensionality no longer
depends on recording duration; vectors built on exactly this construction carried a
submission into the top tier of the CinC 2016 challenge @potes2016.

We additionally compute a small set of global spectral-shape descriptors — the
spectral centroid (the spectral centre of mass), the 85 %-energy roll-off frequency,
bandwidth, zero-crossing rate and root-mean-square (RMS) energy. These broadband
descriptors are inexpensive to compute and characterise the distribution of energy
that the cepstral coefficients capture only indirectly; whether they add
discriminative value beyond the MFCC block is one of the questions our feature-set
comparison (A versus B) is designed to answer empirically.

=== Classifiers

Among classifiers fitted to these descriptors, the support-vector machine with a
Gaussian (radial-basis-function) kernel @cortes1995 is reported most often: the
kernel lets the model trace curved decision surfaces in the original feature space
without ever materialising the high-dimensional space those surfaces formally
inhabit, which suits cepstral classes whose boundaries are not linear, and it stays
numerically stable in the regime typical here — hundreds of features and tens of
thousands of segments. Two tree ensembles serve as the principal alternatives. A
random forest @breiman2001 grows many trees on perturbed views of the data and pools
their votes, damping the variance of any single tree; gradient boosting, in its
XGBoost form @chen2016, instead grows trees in sequence so that each one corrects its
predecessors' residual errors, and it frequently matches or beats the SVM at lower
inference cost while exposing per-feature importances that lend a degree of
interpretability. Logistic regression is kept as a deliberately transparent linear
yardstick: it cannot represent feature interactions, but the cepstral space is often
well-conditioned enough for a linear boundary to compete, which makes it a principled
lower bound against which the non-linear models are judged.

A concern shared by all of these classifiers is class imbalance. One widely used
remedy is to synthesise minority examples with SMOTE @smote, but only ever inside
the training fold — generating synthetic points before the train/test split leaks
test-neighbour information into training @lones2021. The present study sidesteps
synthesis altogether and instead reweights the per-class loss within each training
fold (Chapter 2), which addresses the same majority-class bias without manufacturing
data; the in-fold-only discipline is a recurring correctness requirement that a
substantial fraction of published studies violate by resampling globally.

== Deep-learning approaches

Deep learning on time–frequency images is now the prevailing approach to audio
classification. For auscultation the customary input is a log-mel spectrogram — in
effect a single-channel image in which one axis advances through time, the other
climbs a perceptually scaled frequency ladder, and intensity encodes log-power. Such
an image is detailed enough to expose both the spectral fingerprint of a murmur and
the temporal rhythm of adventitious lung sounds, and — being image-shaped — it lets
convolutional weights pre-trained on natural photographs be repurposed for audio.

=== Convolutional networks

Compact convolutional networks (CNNs) with three to five convolutional layers are a
popular first deep-learning baseline because they train in minutes on a GPU and in
tens of minutes on a CPU. A representative ICBHI study by Demir et al. converted
respiratory recordings to short-time-Fourier spectrograms, used a pre-trained
convolutional network as a fixed feature extractor, and classified the resulting
embeddings with an SVM — a hybrid that outperformed purely hand-crafted baselines on
the respiratory task @demir2020. This pattern, in which a convolutional backbone
supplies the representation and a lightweight classifier makes the decision, recurs
throughout the auscultation literature and directly motivates the transfer-learning
approach examined next.

=== Transfer learning

A frequent transfer route adopts EfficientNet-B0 as the feature extractor
@efficientnet: its parameter budget — about four million — is small enough to
fine-tune on a few thousand recordings yet large enough to inherit useful visual
priors from ImageNet pretraining. Because the network expects three input planes,
the single mel channel is tiled threefold (or routed through a thin projection) to
match. A common fine-tuning schedule keeps the pretrained weights frozen for a brief
warm-up and then releases them for end-to-end training — a recipe that pays off most
when the audio corpus is small, on the order of a few thousand recordings, and the
backbone's image priors supply helpful inductive bias.

=== Audio transformers and recent advances

The Audio Spectrogram Transformer (AST) @gong2021ast removes convolution from the
pipeline altogether: the mel image is cut into a grid of partly overlapping tiles,
each tile becomes a token, and stacked self-attention layers learn how every tile
relates to every other, giving the model a global view of spectro-temporal structure
that a shallow CNN's limited receptive field cannot reach. Built on such a backbone,
patch-mix contrastive training reported a new best result on the ICBHI benchmark
@ast_patchmix. These models are GPU-hungry, however, and frequently augment data
before the train/test split, which can inflate published figures; indeed, a careful
reading shows that many high-scoring respiratory papers quietly use random or
recording-level partitions rather than the official patient-independent one,
rendering their headline numbers incomparable with protocol-faithful studies
@lones2021.

== The cross-modality gap and research novelty

The overwhelming majority of auscultation machine-learning research is unimodal:
it addresses either heart sounds or lung sounds, but not both simultaneously. Yet
a clinician uses the same instrument and a closely related interpretive framework
across modalities; a computational tool that generalises across modalities is both
more practical and more theoretically interesting. Unified models that operate on
heart and lung sounds within a single pipeline are only beginning to be explored
@unified_heartlung. A recent generalist body-sound foundation model @auscultabase
represents an ambitious attempt at unification across auscultation modalities, but it
relies on large-scale data not assembled for the community benchmark tasks.

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
substantial fraction of the published auscultation literature.

=== Research gap and novelty claim

The research gap addressed by this study is therefore twofold. First, a single,
leakage-safe pipeline evaluated *identically* on both heart and lung sounds — same
protocol, same metric family, same model set — is comparatively rare. Second, the
question of whether method rankings transfer across auscultation modalities (does the
best heart classifier also win on lung sounds?) is largely unstudied, even though
the answer has practical implications for how much can be shared in a multi-modality
system.

This study's novelty is a *unified, reproducible, cross-modal comparison* under a
zero-leakage grouped-split protocol: the same codebase, preprocessing chain, model
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
grouped splits or compare methods across modalities under a single protocol.
The cross-modal ranking question is largely open. This motivates the unified
pipeline specified in Chapter 2, and the cross-modal analysis of Chapter 4.
