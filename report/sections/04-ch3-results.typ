// 04-ch3-results.typ — Chapter 3: Experimental results (heart + lung).
// All numbers are taken directly from:
//   results/tables/unified_comparison.csv   (all 20 rows)
//   results/tables/metrics_heart_classical.csv
//   results/tables/metrics_lung_classical.csv
//   results/tables/metrics_heart_cnn.csv
//   results/tables/metrics_lung_cnn.csv
//   results/tables/volumetrics_classical.csv
//   results/tables/volumetrics_cnn.csv
// Classical results are FINAL. DL rows updated to HPO+multi-seed mean±std (FINAL).
// DL-RESULTS-DROPIN slot filled 2026-06-02.

#import "../helpers.typ": *

= Experimental results

This chapter reports the empirical results of the classical and deep-learning
experiments on heart and lung sounds under the leakage-safe grouped-split
protocol described in Chapter 2. All headline numbers are at the recording level
for heart sounds and at the cycle level for lung sounds, using the official
balanced metric of each task. Where a model predicted only one class the
result is marked degenerate in the discussion; no such degenerate case arose in
the final experiments. Deep-learning results are HPO-tuned (val-only selection,
$128$ trials) and reported as mean±std over three independent seeds ($1, 2, 42$).

== Unified comparison table

@tab-unified collects every model trained in this study under a single schema,
enabling direct cross-method and cross-modality comparison. All $20$ rows are
final. Rows are grouped first by modality, then by feature set, then by
model. Within each modality the best primary-metric score is shown in bold.
Deep-learning scores represent HPO-tuned mean±std over three seeds.

// Full 20-row comparison table built from unified_comparison.csv
#figure(
  caption: [Unified model comparison across both modalities.
  Heart metric: $"MAcc" = ("Se"+"Sp") \/ 2$ (CinC 2016 official). Lung metric: $"ICBHI Score" = ("Se"+"Sp") \/ 2$
  (ICBHI 2017 official). DL rows report HPO-tuned mean±std over three seeds ($1, 2, 42$),
  val-only HPO selection ($128$ trials).
  Feature sets: A = MFCC+Δ+ΔΔ ($240$-d); B = MFCC+Δ+ΔΔ+spectral ($250$-d); log-mel = $64 times 128$ image.],
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
    [*Heart*], [*B (MFCC+Δ+spec)*], [*XGBoost*], [*MAcc*],
    [*0.903*], [*0.946*], [*0.859*], [*0.839*],
    // ---- Heart DL ----
    [Heart], [log-mel 64×128], [SmallCNN],       [MAcc], [0.871 ± 0.009], [0.910], [0.831], [—],
    [Heart], [log-mel 64×128], [EfficientNet-B0], [MAcc], [0.898 ± 0.008], [0.936], [0.860], [—],
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
    [Lung], [log-mel 64×128], [SmallCNN],       [ICBHI], [0.540 ± 0.022], [0.755], [0.325], [—],
    [*Lung*], [*log-mel 64×128*], [*EfficientNet-B0*], [*ICBHI*],
    [*0.555 ± 0.016*], [*0.509*], [*0.601*], [*—*],
  ),
) <tab-unified>

#v(0.4em)
#text(size: 10pt)[DL rows: HPO-tuned ($128$-trial bounded random search, val-only selection) mean±std over seeds ${1, 2, 42}$.
  Macro-F1 not computed for multi-seed DL (indicated by —); macro-F1 figures for single-seed
  runs appear in Annex B.
]

== Heart-sound results (PhysioNet/CinC 2016)

=== Classical models

All eight classical configurations (two feature sets $times$ four classifiers) were
evaluated on the same held-out test set of $626$ recordings (split at the recording
level; see Chapter 2). The best classical result was achieved by XGBoost on feature set B
(MFCC+Δ+ΔΔ + spectral statistics), reaching $"MAcc" = 0.903$, with $"Se" = 0.946$ and
$"Sp" = 0.859$. This result confirms the practical value of the five additional
spectral statistics: XGBoost on feature set A scored $"MAcc" = 0.879$, so the richer
feature set adds a meaningful $2.3$ percentage points.

This figure is not directly comparable to the CinC 2016 challenge leaderboard
(top entries $approx 0.86$; Chapter 1): that ranking was computed on the withheld private
test set, whereas our value comes from a self-constructed recording-level partition
of the public training pool. Because the heart split is recording-level rather than
strictly subject-level (Section 2.1), the absolute score may be optimistic relative
to a fully subject-disjoint evaluation. The contribution of this study is therefore
the controlled cross-method comparison under one fixed protocol, not the absolute
number itself.

SVM was the second-best classical model across both feature sets ($"MAcc"$ $0.859$ on A,
$0.869$ on B). SVM and XGBoost both achieve high sensitivity ($"Se" >= 0.915$ in all
configurations), reflecting the fact that class-weighted training forces these
models to capture true positives. By contrast, random forest reaches high
specificity ($"Sp" >= 0.940$) but lower sensitivity ($"Se" approx 0.69"–"0.72$), suggesting that
RF's ensemble structure defaults more readily to the majority normal class.
Logistic regression, despite being the simplest linear model, achieves a
respectable $"MAcc"$ of $0.825$ on feature set B, which indicates that the MFCC+spectral
feature space is well-structured for linear separation.

@fig-cm-heart-best shows the confusion matrix for the best model (XGBoost, set B).
Out of $130$ abnormal test recordings, $123$ are correctly identified ($7$ false
negatives), and out of $496$ normal recordings, $426$ are correctly identified ($70$ false
positives), consistent with the reported $"Se" = 0.946$ and $"Sp" = 0.859$. For a
screening or triage tool this is a favourable operating point: at $"Se" = 0.946$ only
about $5%$ of abnormal recordings are missed (the error that matters most when the aim
is to flag cases for expert review), while the false positives implied by
$"Sp" = 0.859$ would be cleared by a confirmatory examination.

#figure(
  image("../../results/figures/cm_heart_B_xgb.png", width: 58%),
  caption: [Confusion matrix of the best heart-sound classical classifier (XGBoost,
  feature set B: MFCC+Δ+ΔΔ + spectral statistics). Rows: true class; columns:
  predicted class. Test set: $626$ heart recordings (recording-level split; CinC 2016
  publishes no recording-to-subject map, see Section 2.1).],
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

The deep-learning models were evaluated with HPO-tuned hyperparameters ($128$-trial
bounded random search, validation-only selection) and results are reported as
mean±std over three independent seeds. The compact SmallCNN trained on $64 times 128$ log-mel
spectrograms reached $"MAcc" = 0.871 plus.minus 0.009$ ($"Se" = 0.910$, $"Sp" = 0.831$), and
EfficientNet-B0 reached $"MAcc" = 0.898 plus.minus 0.008$ ($"Se" = 0.936$, $"Sp" = 0.860$). With
hyperparameter tuning, EfficientNet-B0 reaches a mean accuracy comparable with the
best classical model (XGBoost-B, $"MAcc" = 0.903$): the difference of $0.005$ is smaller
than EfficientNet-B0's own seed-to-seed standard deviation ($plus.minus 0.008$), so the two
models are of comparable accuracy on this task, with the classical model remaining
numerically best. We do not claim statistical equivalence: no formal significance
test was run, because the classical models were trained as single deterministic
fits without a seed distribution to test against. This nonetheless reframes the
earlier core-run observation that classical methods dominated: after equivalent
HPO, the two families are of comparable accuracy on heart sounds.

The loss curves for both deep models (@fig-lc-heart-cnn and @fig-lc-heart-eff) show
the training loss falling steadily; the validation loss trends downward for SmallCNN
and plateaus for EfficientNet-B0, with no runaway overfitting. Early stopping restores
the best validation checkpoint, confirming that the dropout ($>= 0.3$) and early-stopping
regularisation are effective.

#figure(
  image("../../results/figures/learning_curve_heart_cnn.png", width: 72%),
  caption: [Training and validation loss per epoch for SmallCNN on heart sounds.
  The training loss falls steadily and the (noisier) validation loss trends downward,
  indicating convergence without runaway overfitting.],
) <fig-lc-heart-cnn>

#figure(
  image("../../results/figures/learning_curve_heart_effnet.png", width: 72%),
  caption: [Training and validation loss per epoch for EfficientNet-B0 on heart sounds.
  The validation loss plateaus while the training loss keeps falling, the modest
  train–validation gap typical of fine-tuning; early stopping selects the best checkpoint.],
) <fig-lc-heart-eff>

== Lung-sound results (ICBHI 2017)

=== Classical models

All eight classical configurations were evaluated on the official ICBHI $60\/40$
patient-independent test set of $2636$ cycles from $47$ test patients. ICBHI scores
are markedly lower than heart MAcc across all models, reflecting the fundamentally
harder nature of this task: four imbalanced classes (three minority abnormal types)
versus two classes.

The best classical result is SVM on feature set B with $"ICBHI score" = 0.537$
($"Se" = 0.615$, $"Sp" = 0.458$). The score on feature set A is nearly identical
($0.536$), suggesting that the spectral statistics add marginal value for lung
classification compared to the clear boost they provided for heart classification.
XGBoost on feature set A scores $0.511$ and declines on feature set B ($0.500$),
which is the opposite pattern from heart sounds, an observation directly relevant
to the cross-modal analysis of Chapter 4.

A striking pattern in the lung results is the contrast between random forest and
all other classifiers. RF achieves very high specificity ($"Sp" approx 0.85$) but extremely
low sensitivity on the abnormal classes ($"Se" approx 0.10$). This "majority-class
collapse" (predicting almost exclusively normal) is a failure mode specific to
RF's averaging behaviour when classes are highly imbalanced and class weighting
alone has not fully counteracted the imbalance within the training fold.
No separate figure is shown for the random forest; its collapse is evident directly
in the per-class sensitivities of @tab-unified (abnormal-class $"Se" approx 0.10$). For contrast,
@fig-cm-lung-best shows the best classical model, the SVM, whose predictions are
distributed across all four classes rather than collapsing to the majority.

#figure(
  image("../../results/figures/cm_lung_B_svm.png", width: 58%),
  caption: [Confusion matrix of the best lung-sound classical classifier (SVM,
  feature set B). Rows: true class; columns: predicted class. Test set: $2636$
  cycles from $47$ patients.],
) <fig-cm-lung-best>

Logistic regression on feature set B achieves $"ICBHI" = 0.533$, comparable to SVM.
Both linear models benefit from class-weighted loss, which forces predictions
across all four classes rather than collapsing to the majority.

The per-class sensitivities in Annex A locate the difficulty. The rare combined
crackle-and-wheeze class is barely recovered by any classifier: its per-class
sensitivity runs from about $0.11$ to $0.16$ for the linear and boosting models and
falls to almost zero for random forest. Crackle and wheeze are captured only
partially, with sensitivities of roughly $0.2$ to $0.4$. Because the ICBHI score
pools sensitivity over these three abnormal classes, it is the scarcity of the
minority classes, not a weakness of any single algorithm, that holds every classical
method near $0.54$. The deep models meet the same bottleneck, which is why their
larger capacity buys so little on this task. Clinically the consequence is concrete:
a tool built on this data would reliably separate normal from abnormal breathing but
could not yet be trusted to name which adventitious sound is present.

=== Deep-learning models

The deep-learning lung models are also reported as HPO-tuned mean±std over three
seeds. The SmallCNN reached $"ICBHI" = 0.540 plus.minus 0.022$ ($"Se" = 0.755$, $"Sp" = 0.325$), and
EfficientNet-B0 reached $"ICBHI" = 0.555 plus.minus 0.016$ ($"Se" = 0.509$, $"Sp" = 0.601$).
Note that the multi-seed mean for CNN ($0.540$) is slightly below an earlier
single-seed run ($0.551$ reported at the core-run stage); this honest regression
reflects seed variance on the harder four-class task rather than a model change.
The EfficientNet-B0 is the overall best lung model ($0.555$), marginally ahead of
SmallCNN and the best classical model (SVM-B, $0.537$). The deep-vs-classical
difference on lung is small (roughly $1.5"–"2$ pp) and within the multi-seed standard
deviation, placing classical and deep methods in the same performance tier on this
task. Both deep models show better sensitivity on minority abnormal classes compared
to random forest, which is reflected in a more balanced confusion matrix
(@fig-cm-lung-cnn).

This best ICBHI score (0.555) sits below the strongest published results on the
official split (patch-mix AST and related transformer systems reach the low $0.60$s),
and the gap reflects scope rather than a methodological flaw: those systems use
transformer backbones with large-scale audio pretraining and heavy augmentation,
whereas our deep models are a compact CNN and an ImageNet-pretrained EfficientNet-B0
trained under a deliberately modest, leakage-controlled protocol. As Chapter 1 notes,
a number of higher reported ICBHI scores additionally rely on non-official or
recording-level splits that are not comparable to the official patient-independent
partition adopted here.

#figure(
  image("../../results/figures/cm_lung_cnn.png", width: 58%),
  caption: [Confusion matrix of the lung-sound SmallCNN classifier (on log-mel
  spectrograms). The log-mel representation enables better discrimination
  of crackle and wheeze relative to classical models.],
) <fig-cm-lung-cnn>

@fig-lc-lung-cnn shows the loss curve for the lung CNN. The training loss falls while
the validation loss flattens and then begins to climb after roughly epoch 9 (the
over-fitting expected on this harder four-class task) so early stopping restores the
earlier best checkpoint.

#figure(
  image("../../results/figures/learning_curve_lung_cnn.png", width: 72%),
  caption: [Training and validation loss per epoch for SmallCNN on lung sounds.
  The validation loss starts rising after roughly epoch 9 (over-fitting on the harder
  four-class task), so early stopping restores the earlier best checkpoint.],
) <fig-lc-lung-cnn>

== Volumetric characteristics

@tab-volumetrics collects the dataset sizes, model parameter counts and training
times that characterise the experimental scale. These are required by Annex 5
§2.5. In total the study evaluates $20$ distinct model configurations under one
protocol ($4$ classical algorithms $times 2$ feature sets $+ 2$ deep models, on
each of the two modalities); each deep configuration is additionally the outcome
of a $128$-trial hyperparameter search and is reported as a mean over $3$ seeds, so
the deep results alone rest on several hundred individual training runs. Beyond the
data and model sizes, the experimental pipeline itself comprises $7 thin 012$ lines
of Python across $31$ modules in `src/` and `scripts/` (approximately $290$ KB of
source code), supported by a further $2 thin 961$ lines of automated tests.
Full details including per-class cycle counts appear in Annex B.

#figure(
  caption: [Volumetric characteristics of the experiments. Classical training times
  are CPU wall-clock seconds; deep-learning training times are GPU (NVIDIA A100)
  wall-clock seconds for the HPO-selected configuration. Heart patient counts are
  not reported: CinC 2016 provides no recording-to-subject map, so the heart split
  is grouped at the recording level (Section 2.1).],
  table(
    columns: (2fr, 1fr, 1fr),
    align: (left, center, center),
    table.header([*Quantity*], [*Heart*], [*Lung*]),
    [Train segments/cycles],         [33 246], [4 262],
    [Test segments/cycles],          [4 167],  [2 636],
    [Train recordings / patients],   [2 500 / —], [551 / 79],
    [Test recordings / patients],    [626 / —],   [369 / 47],
    [Best classical model (SVM-B) train time (s)], [659], [31],
    [Best classical model (XGB-B) train time (s)], [21],  [26],
    [SmallCNN parameters (HPO-tuned)], [389 314], [98 148],
    [SmallCNN train time (s), GPU],    [262],    [26],
    [EfficientNet-B0 parameters],      [4 010 110], [4 012 672],
    [EfficientNet-B0 train time (s), GPU],[1 345], [204],
    [Log-mel data volume (MB)],      [1 227],  [226],
    [Classical feature data volume (MB)], [74], [14],
  ),
) <tab-volumetrics>

== Chapter summary

On heart sounds, the best overall result is XGBoost with MFCC+Δ+ΔΔ + spectral
statistics (feature set B), reaching $"MAcc" = 0.903$ ($"Se" = 0.946$, $"Sp" = 0.859$). After
HPO tuning, the deep EfficientNet-B0 reaches $"MAcc" = 0.898 plus.minus 0.008$, comparable with
the best classical model (the $0.005$ difference is below the model's $plus.minus 0.008$ seed
standard deviation), though the classical model remains numerically best. SmallCNN
reaches $"MAcc" = 0.871 plus.minus 0.009$. After equivalent tuning effort the two method families
are of comparable accuracy on heart sounds, in contrast to the larger classical
advantage seen in the untuned core run.

On lung sounds, the best result across all $20$ configurations is EfficientNet-B0
with $"ICBHI" = 0.555 plus.minus 0.016$, narrowly ahead of the best classical model (SVM-B,
$0.537$) and SmallCNN ($0.540 plus.minus 0.022$). Classical and deep models occupy the same
performance tier on this task; the four-class imbalanced structure limits all
methods equally. The gap between heart and lung scores ($"MAcc" approx 0.90$ vs.
$"ICBHI" approx 0.55$) reflects the fundamental difficulty difference: a binary task on
longer recordings versus a four-class imbalanced task on short cycles. Chapter 4
analyses what these differences imply for cross-modal transfer of method rankings.
