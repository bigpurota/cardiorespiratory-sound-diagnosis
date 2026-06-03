
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

Because heart and lung models were evaluated under one shared protocol and one
metric family (the same preprocessing, features, model set and balanced score;
only the grouping granularity and test fraction differ, as the datasets dictate),
their rankings can be compared directly. We examine the question in three
ways: (i) the ordinal ranking of classifiers within each modality, (ii) the
relative advantage of feature set B over feature set A, and (iii) the relative
standing of classical versus deep models.

=== Classifier ranking

For heart sounds the ranking (best to worst) by primary metric is:
XGBoost-B ($0.903$) $>$ XGBoost-A ($0.879$) $>$ SVM-B ($0.869$) $>$ SVM-A ($0.859$) $>$
RF-B ($0.831$) $>$ LogReg-B ($0.825$) $>$ RF-A ($0.817$) $>$ LogReg-A ($0.794$).
Collapsing per feature set, the ordering at the classifier level is
XGBoost $>$ SVM $>$ RF $>$ LogReg.

For lung sounds the ranking by ICBHI score is substantially different:
SVM-B ($0.537$) $approx$ SVM-A ($0.536$) $>$ LogReg-B ($0.533$) $>$ XGBoost-A ($0.511$) $>$
LogReg-A ($0.507$) $>$ XGBoost-B ($0.500$) $>$ RF-A ($0.476$) $approx$ RF-B ($0.472$).
The lung classifier ordering is SVM $approx$ LogReg $>$ XGBoost $>$ RF.

Two inversions are notable. First, XGBoost, the dominant model on heart sounds,
falls to third place on lung sounds and actually degrades from feature set A to B
($0.511$ to $0.500$), the reverse of its heart behaviour. Second, logistic regression
rises to near-parity with SVM on lung sounds, suggesting that the four-class
MFCC feature space is more linearly separable than the two-class heart space, or
that gradient boosting's additional capacity introduces over-fitting on the smaller
lung training set ($4262$ cycles versus $33 thin 246$ heart windows). These rank inversions
are the central qualitative observation of the cross-modal analysis; the rank
correlation below is computed first on these four classifiers and then re-checked on an
expanded nine-classifier panel (@tab-ninepanel) to confirm it is not an artefact of a
small sample.

A prediction-level paired test sharpens the same point and removes any suspicion that
the inversion is a rounding effect. On heart, a McNemar test on the $626$ recording-level
predictions confirms the rank-1 model is genuinely ahead of the rank-2: XGBoost-B beats
SVM-B on $39$ of their $48$ discordant recordings ($chi^2 = 17.5$, $p approx 1.5 times 10^(-5)$).
On lung the very same pair is reordered, and instructively so. At the raw cycle level
XGBoost-B is in fact _significantly more accurate_ than SVM-B (it wins $401$ of their $711$
discordant cycles, $chi^2 = 11.4$, $p approx 7 times 10^(-4)$), yet SVM-B attains the higher
balanced ICBHI score ($0.537$ vs. $0.500$), because XGBoost buys its raw-accuracy edge by
leaning on the majority normal class while the balanced metric refuses to reward that. The
model that wins on heart therefore does not win on lung under the metric that matters, and
the reason is visible in the individual predictions, not merely in rounded aggregates,
exactly the non-transfer of rankings this section documents.

=== Feature set effect

On heart sounds, adding spectral statistics (feature set B) consistently improves
every classifier, with the largest absolute gain for XGBoost ($+2.3$ pp) and modest
gains for SVM ($+1.1$ pp) and LogReg ($+3.0$ pp). On lung sounds, the feature-set
effect is inconsistent: SVM improves negligibly ($+0.1$ pp), LogReg improves ($+2.6$ pp),
but XGBoost worsens ($-1.1$ pp) and RF also worsens. The spectral statistics that
capture cardiac sound "brightness" (centroid, roll-off) are apparently less
informative for the respiratory task, where the shape of adventitious sound
transients is more relevant than broadband spectral shape.

=== Classical versus deep learning

With HPO-tuned hyperparameters and three-seed averaging, the picture on heart
sounds is more nuanced than the core run suggested. EfficientNet-B0 reaches
$"MAcc" = 0.898 plus.minus 0.008$ ($"Se" = 0.936$, $"Sp" = 0.860$), comparable with the
best classical model (XGBoost-B, $"MAcc" = 0.903$): the $0.005$ difference is below
EfficientNet-B0's $plus.minus 0.008$ seed standard deviation, though the classical model remains
numerically best. SmallCNN reaches $"MAcc" = 0.871 plus.minus 0.009$. The clear classical
advantage seen in the untuned core run thus shrinks to a margin within seed noise
after equivalent optimisation effort, without being reversed.

On lung sounds, the deep models achieve $"ICBHI" = 0.540 plus.minus 0.022$ (CNN) and
$0.555 plus.minus 0.016$ (EfficientNet-B0), placing them marginally ahead of but practically
indistinguishable from the best classical model (SVM-B, $0.537$). Both modalities
therefore converge on the same finding: with proper HPO, classical and deep methods
occupy the same performance tier on these tasks and dataset scales.

=== Summary of cross-modal findings (classical)

The Spearman rank correlation @spearman1904 between the per-classifier ICBHI scores
(lung) and the corresponding MAcc scores (heart), computed over the four classical
classifier types, is weak and feature-set-dependent: $rho = 0.60$ on feature set A but
$rho = 0.00$ on feature set B ($n = 4$ each); pooling both sets gives $rho = 0.24$
($n = 8$). None of these is significant at conventional thresholds given the small
sample sizes. Far from undermining the analysis, this inconsistency reinforces its
central point: classifier rankings do _not_ transfer reliably between modalities.
Concretely, XGBoost's dominance on heart does not carry over to lung (it falls to
third, and even reverses its feature-set preference), whereas SVM is the one model
that stays competitive on both. The practical implication is that a researcher who
tuned a model exclusively on heart data and redeployed it on lung data would incur a
non-trivial performance penalty, which is precisely the argument for per-modality
model selection even within a shared pipeline.

=== Robustness: a nine-classifier panel

A correlation over four classifiers is a thin estimate, so we tested whether the
non-transfer of rankings survives a larger panel. Five further class-weighted
classifiers were added under the identical protocol, Extra-Trees, AdaBoost, a ridge
linear classifier, an SGD-trained logistic model and a linear-kernel SVM, and all nine
were re-evaluated on both modalities (@tab-ninepanel). The five additions use fixed
defaults, like the untuned logistic-regression and random-forest baselines, so the
panel stays internally consistent.

The expansion reinforces the finding rather than dissolving it. The heart-versus-lung
Spearman correlation across the nine classifiers stays weak and non-significant on both
feature sets ($rho = 0.42$, $p = 0.26$ on set A; $rho = 0.13$, $p = 0.73$ on set B;
pooled $rho = 0.32$, $n = 18$, $p = 0.19$), and the headline inversion grows sharper:
XGBoost is rank 1 of 9 on heart but falls to rank 7 of 9 on lung, while the
radial-basis SVM is the only classifier in the top two of both. More informative is a
structure the four-model panel could not expose. The tree ensembles (XGBoost, AdaBoost,
random forest, Extra-Trees) occupy the upper half on heart but the lower half on lung,
whereas the linear models (ridge, logistic regression, SGD-logistic, linear SVM) do the
opposite, clustering near the top on lung. The non-transfer of rankings is therefore not
an artefact of a small panel but a model-family-by-modality interaction: non-linear tree
ensembles suit the binary heart task, linear models are relatively stronger on the
four-class lung task, and only the kernel SVM is robust to both. This is a concrete,
quantified reason to select the classifier per modality even inside one shared pipeline.

#figure(
  caption: [Nine-classifier panel under the shared protocol (feature set B). Heart score
  is recording-level MAcc; lung score is the cycle-level ICBHI Score; ranks are within
  each modality. The four classifiers of Chapter 3 are combined with five class-weighted
  additions (Extra-Trees, AdaBoost, ridge, SGD-logistic, linear SVM, fixed defaults).
  XGBoost leads on heart (rank 1) yet falls to rank 7 on lung; the linear family does the
  reverse. Spearman rank correlation heart-vs-lung: $rho = 0.13$ ($p = 0.73$).],
  table(
    columns: (1.7fr, 1fr, 1fr, 0.6fr, 1fr, 0.6fr),
    align: (left, left, center, center, center, center),
    table.header([*Model*], [*Family*], [*Heart MAcc*], [*Rk*], [*Lung ICBHI*], [*Rk*]),
    [*XGBoost*],       [boosting], [*0.903*], [*1*], [0.500], [7],
    [SVM (RBF)],       [kernel],   [0.869], [2], [*0.537*], [*1*],
    [AdaBoost],        [boosting], [0.868], [3], [0.503], [6],
    [Random forest],   [bagging],  [0.831], [4], [0.472], [8],
    [Ridge],           [linear],   [0.831], [5], [0.533], [2],
    [Linear SVM],      [linear],   [0.830], [6], [0.525], [5],
    [Logistic reg.],   [linear],   [0.825], [7], [0.533], [3],
    [SGD-logistic],    [linear],   [0.819], [8], [0.528], [4],
    [Extra-Trees],     [bagging],  [0.810], [9], [0.470], [9],
  ),
) <tab-ninepanel>

== Deep cross-modal transfer and joint multi-task experiments

To probe whether the shared log-mel pipeline enables feature transfer between
modalities, we ran three deep-learning experiments beyond the per-modality training
of Chapter 3, each repeated over the same three seeds ($1, 2, 42$) and reported as
mean±std: (i) direct transfer, where an encoder pre-trained on one modality is
fine-tuned and evaluated on the other; (ii) joint multi-task training, where a single
shared encoder with two task-specific heads is trained on both modalities at once; and
(iii) in-domain references trained under the identical (non-HPO) protocol, so that the
transfer comparison is strictly like-for-like. The transfer matrix in
@fig-cross-modal-heatmap summarises the EfficientNet-B0 results; @tab-joint gives all
per-setting scores with their seed standard deviations.

#figure(
  image("../../results/figures/cross_modal_heatmap.png", width: 78%),
  caption: [Cross-modal transfer matrix for EfficientNet-B0 (three-seed means). Each
  cell shows the primary metric (MAcc for a heart target; ICBHI Score for a lung
  target) when a model whose features originate in the source modality (rows) is
  fine-tuned and evaluated on the target modality (columns): the diagonal is in-domain,
  the off-diagonal cells are direct transfer, and the bottom row is the joint
  multi-task model. SmallCNN values appear in @tab-joint.],
) <fig-cross-modal-heatmap>

The central result is that, once seed variance is taken into account, deep
cross-modal transfer is benign and roughly symmetric rather than asymmetric.
Lung→heart transfer reaches $"MAcc" = 0.887 plus.minus 0.004$ (EfficientNet-B0),
indistinguishable from the in-domain heart reference ($0.887 plus.minus 0.010$); heart→lung
transfer reaches $"ICBHI" = 0.561 plus.minus 0.009$, at or slightly above the in-domain lung
reference ($0.556 plus.minus 0.003$). The SmallCNN behaves identically (lung→heart
$0.854 plus.minus 0.004$ vs. in-domain $0.852 plus.minus 0.008$; heart→lung $0.531 plus.minus 0.013$ vs.
in-domain $0.526 plus.minus 0.037$). In short, with full fine-tuning on the target the choice
of source modality is largely washed out: neither direction incurs a meaningful
penalty, and pre-training on the opposite modality is, if anything, marginally helpful.

This corrects an artefact of single-seed evaluation. A single representative run had
suggested a strong asymmetry, with heart→lung transfer apparently falling well below
the lung in-domain baseline; that gap did not survive three-seed averaging and lay
within the seed spread. The episode is a concrete illustration of why every deep
result in this study is reported over multiple seeds: a lone run can manufacture a
qualitative "finding" — here, transfer asymmetry — that dissolves under seed-robust
evaluation. The robust cross-modal conclusion is therefore the classical one of the
previous section: it is method _rankings_, not learned features, that fail to transfer
between modalities.

The joint multi-task model (shared encoder, two heads) reaches
$"MAcc" = 0.873 plus.minus 0.011 \/ 0.848 plus.minus 0.008$ for heart (EfficientNet-B0 / SmallCNN)
and $"ICBHI" = 0.521 plus.minus 0.018 \/ 0.554 plus.minus 0.014$ for lung. Relative to per-modality
training the joint model trades off slightly differently for the two backbones: it
holds heart within about one standard deviation of in-domain and shows a small lung
drop for EfficientNet-B0, whereas for the SmallCNN it slightly improves lung at a
negligible heart cost. No catastrophic interference appears in either case, consistent
with a shared encoder that has enough capacity to serve both tasks but cannot fully
optimise for each at once.

@tab-joint collects all per-setting scores for direct reference.

#figure(
  caption: [Per-modality deep-model scores under in-domain, transfer and joint
  settings, as mean±std over three seeds ($1, 2, 42$). Transfer scores are placed in
  the columns of the _target_ modality. In-domain references use the same non-HPO
  protocol as the transfer runs; the HPO-tuned in-domain values of Chapter 3 are
  slightly higher. The classical best is the per-modality recording/cycle-level score.],
  table(
    columns: (2.2fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    table.header(
      [*Setting*], [*Heart CNN*], [*Heart EffNet*], [*Lung CNN*], [*Lung EffNet*]),
    [In-domain],                  [0.852±0.008], [0.887±0.010], [0.526±0.037], [0.556±0.003],
    [Transfer heart→lung],        [—],     [—],     [0.531±0.013], [0.561±0.009],
    [Transfer lung→heart],        [0.854±0.004], [0.887±0.004], [—],     [—],
    [Joint multi-task],           [0.848±0.008], [0.873±0.011], [0.554±0.014], [0.521±0.018],
    [Per-modality classical best],[0.903], [—],     [0.537], [—],
  ),
) <tab-joint>

== Audio Spectrogram Transformer: a transformer baseline

As a fourth deep model, a pretrained Audio Spectrogram Transformer (AST)
@gong2021ast, originally trained on AudioSet, was fine-tuned on both modalities under
the same pipeline and, like the other deep models, evaluated as a mean ± standard
deviation over three seeds ($1, 2, 42$). One qualification still frames the result: the
AST is heavyweight ($86$ million parameters, against four million for EfficientNet-B0)
and, under the project's compute budget, each seed was fine-tuned for only a few epochs
(two on heart, eight to ten on lung), so its scores are an epoch-limited figure rather
than the model's ceiling.

On heart sounds the AST reaches $"MAcc" = 0.864 plus.minus 0.006$ at a recall-leaning
operating point ($"Se" = 0.928$, $"Sp" = 0.800$): it favours recovering abnormal
recordings at the cost of more false positives, a screening-friendly bias produced by
class-weighted training. That this reflects genuine discrimination rather than
over-fitting is confirmed by a high ROC-AUC ($0.950$ averaged over the three seeds). Its
balanced MAcc nonetheless trails the tuned classical XGBoost ($0.903$) and the
three-seed EfficientNet-B0 ($0.898$), so the transformer does not displace the simpler
heart models at this data scale and tuning budget (@fig-cm-heart-ast).

On lung sounds the picture differs: the AST reaches $"ICBHI" = 0.594 plus.minus 0.023$,
the single best lung score in the study, ahead of the three-seed EfficientNet-B0 mean
($0.555 plus.minus 0.016$) and every classical model. The lead is consistent rather than
a lucky seed: even the weakest AST seed ($0.576$) exceeds the EfficientNet-B0 mean. This
agrees with the literature position (Chapter 1) that transformer backbones with
large-scale audio pretraining lead on ICBHI, and it indicates the audio-pretrained AST
has a real, if modest, edge on the harder four-class respiratory task.

The AST thus both confirms that transformer audio models integrate into the shared
pipeline without modification and, on lung sounds, edges ahead of the convolutional
models over three seeds; training each seed to convergence at a larger epoch budget is
the natural next step.

#figure(
  image("../../results/figures/cm_heart_ast.png", width: 58%),
  caption: [Confusion matrix of the fine-tuned AST on heart sounds for a representative
  seed (seed $42$: $"MAcc" = 0.871$, $"Se" = 0.946$), at the recording level over $626$
  test recordings. The recall-leaning operating point shows as near-complete recovery of
  the abnormal class at the cost of more false positives on the normal class.],
) <fig-cm-heart-ast>

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
neck during systole, the half-cycle in which cardiac ejection drives peak flow
velocity. In spectral terms, bruits occupy roughly $50"–"800$ Hz, with the dominant
energy below $400$ Hz for mild stenoses and extending upward as stenosis severity
increases and peak flow velocity rises. Duncan et al. @duncan1975
demonstrated that this upward spectral shift is systematic enough to estimate the
residual lumen diameter directly from the bruit spectrum, a quantitative,
feature-based reading of an auscultation signal that closely anticipates the
classification approach pursued in this project for heart and lung sounds. This
overlap with the heart-sound band ($20"–"400$ Hz) means that bruit detection from a
stethoscope placed at the precordium (as opposed to the neck) is problematic
without careful device placement and directional filtering.

The clinical motivation for automating this assessment is strong. Manual
auscultation for carotid bruits has only modest, operator-dependent diagnostic
accuracy: the Rational Clinical Examination review by Sauvé et al. @sauve1993
found the presence of a bruit to be at best a weak-to-moderate predictor of
high-grade stenosis, with sensitivity and specificity that vary widely between
examiners. Yet an audible bruit carries genuine prognostic weight: a meta-analysis
pooling $17 thin 295$ patients reported that carotid bruits roughly double the rate of
cardiovascular death and myocardial infarction @pickett2008. An objective,
reproducible acoustic classifier could therefore both reduce inter-observer
variability and surface a clinically actionable risk marker, which is precisely
the contribution that the feature-engineering and deep-learning pipelines compared
in this report are designed to make for the heart and lung modalities.

The clinical pipeline for bruit assessment shares structural elements with the
heart- and lung-sound pipelines studied here: the signal is captured at fixed
sampling rates (typically $4000"–"10 thin 000$ Hz with modern digital stethoscopes),
bandpass-filtered to the bruit-relevant band, segmented in cardiac-cycle-
synchronised windows, and feature-extracted for classification. A recent smartphone
study @stroke2024 demonstrated on-device deep-learning detection of carotid stenosis
from neck auscultation, establishing proof-of-concept for machine-learning-based
screening. An earlier computer-assisted auscultation system @carotid_pmc was built
specifically to acquire carotid vascular sounds reproducibly (a prerequisite for
assembling a database of pathological bruits) yet, like the smartphone work, it
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
changes: the bandpass cutoffs (adjusted to $50"–"800$ Hz or a tighter passband as
appropriate) and the class label scheme (normal vs. bruit, or graded by stenosis
severity). All other pipeline stages (resampling, windowing, MFCC / log-mel
feature extraction, grouped (subject-level) splitting, evaluation) apply without
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

=== A synthetic proof-of-concept

To show that this infrastructure does run end-to-end on arterial-band audio, and not
merely in principle, we built a small _synthetic_ dataset and passed it through the
unchanged pipeline. No clinical data are used or claimed. The generator produces two
classes of artificial neck recordings at $4000$ Hz: a "bruit" class, in which a
systolic-gated band-limited component models the turbulent-flow spectrum of the
Lees-Dewey phonoangiography model @lees1970, and a "normal" class carrying a
comparable systolic component. The two classes are deliberately made hard to separate:
both always contain mid-band energy of overlapping strength, and the only systematic
difference is a subtle upward shift of the centre frequency in the bruit class (as a
tighter stenosis raises the peak flow velocity and the spectral peak), with overlapping
per-class distributions and a strong added noise floor. The signal is then processed by
exactly the pipeline stages of Chapter 2, only the passband is changed to the
arterial $50"–"800$ Hz band: Butterworth bandpass, peak normalisation, fixed $3$-second
windows and the feature-set-B descriptor, with a strict patient-level split
($84$ train / $36$ test synthetic subjects, $480$ recordings) and the same zero-leakage
assertion.

Under this deliberately overlapping contrast the classical models reach a
recording-level $"MAcc"$ of about $0.80$ (logistic regression $0.799$, random forest
$0.783$, RBF-SVM $0.740$; @fig-cm-arterial), comfortably above the chance level of
$0.5$ but far from the trivial separability of a cleanly separable toy task. The result
carries no clinical weight whatever: the data are synthetic and the difficulty is set by
hand. Its only claim is operational, that the unified pipeline ingests, splits, features
and classifies arterial-band audio without modification beyond a passband change,
exactly as the analytical sub-study argues it would for a future real dataset.

#figure(
  image("../../results/figures/cm_arterial_synth.png", width: 56%),
  caption: [Confusion matrix of the best classical model (logistic regression) on the
  _synthetic_ arterial proof-of-concept (recording level, $144$ test recordings,
  patient-level split). The data are artificial and the difficulty is set by hand; the
  figure illustrates only that the pipeline runs end-to-end on arterial-band audio, not
  any clinical accuracy.],
) <fig-cm-arterial>

=== Chapter summary

The cross-modal analysis delivers three findings. First, classical method rankings
do not fully transfer: XGBoost dominates on heart sounds but falls to third place
on lung (and to rank 7 of 9 once the panel is widened), while SVM is uniformly
competitive. The Spearman rank correlation is weak and non-significant whether measured
on the four core classifiers ($rho = 0.60$ / $0.00$ / $0.24$ for set A / B / pooled) or
on an expanded nine-classifier panel ($rho = 0.42$ / $0.13$ / $0.32$; all $p > 0.19$),
and the expansion reveals a model-family-by-modality interaction (tree ensembles favour
heart, linear models favour lung) that underlies the non-transfer. A paired McNemar test
confirms the inversion at the level of individual predictions. Second, deep cross-modal
transfer, evaluated over three seeds, is benign and roughly symmetric: under full
fine-tuning, both lung→heart and heart→lung transfer reach in-domain performance, so
learned features carry over either way. A single-seed run had suggested an asymmetry,
but it did not survive multi-seed averaging, a cautionary result that reinforces the
study's emphasis on seed-robust evaluation. Joint multi-task training holds both
modalities near per-modality performance with only small, backbone-dependent
trade-offs and no catastrophic interference. Third, with HPO-tuned deep
models, the deep-vs-classical gap on heart closes to within noise, while lung
remains equally hard for all method families. These results motivate per-modality
model selection even within a shared pipeline and quantify the cost of a naive
one-model-all-modalities deployment.

The fine-tuned AST confirms that transformer audio models integrate cleanly into the
pipeline and, on lung sounds, gives the single best score in the study over three seeds
($"ICBHI" = 0.594 plus.minus 0.023$); training each seed to convergence at a larger epoch
budget is left to future work.

The arterial sub-study establishes that the pipeline generalises to bruits in
principle, demonstrated end-to-end on a synthetic arterial-band benchmark
($"MAcc" approx 0.80$ on a deliberately hard contrast), but that empirical clinical work
is blocked by data unavailability, a gap that the community should address through a
coordinated open data release.
