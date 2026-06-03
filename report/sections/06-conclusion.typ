
#heading(numbering: none, outlined: true)[Conclusion]

This project set out to compare, honestly and reproducibly, classical and
deep-learning methods for diagnosing cardiorespiratory pathologies from
auscultation sound, under a single leakage-safe protocol spanning two modalities.
The objectives stated in the Introduction were met as follows.

A reproducible, configuration-driven pipeline shared by the heart and lung
modalities was assembled, with fixed random seeds, pinned dependencies,
leakage-safe grouped splits (patient-level for lung, recording-level for heart) and
an explicit zero-leakage assertion executed before every training run. On heart sounds the best classical configuration, XGBoost with the
MFCC+Δ+ΔΔ + spectral-statistics feature set, reached $"MAcc" = 0.903$
($"Se" = 0.946$, $"Sp" = 0.859$). After HPO tuning ($128$-trial bounded random search,
val-only selection), EfficientNet-B0 reaches $"MAcc" = 0.898 plus.minus 0.008$ (three seeds),
comparable with the best classical model: the gap of $0.005$ falls
within one standard deviation, so deep learning closes the
classical advantage on heart sounds once the tuning effort is equalised. A paired
McNemar test on recording-level predictions makes the point concrete: the two are
statistically indistinguishable at the seed where EfficientNet-B0 reaches $0.898$
($p = 0.68$), while XGBoost is significantly ahead on the weaker seeds, so neither
family is reliably best.
On lung sounds the models were evaluated at the cycle level with the official
ICBHI score. The best configuration, HPO-tuned EfficientNet-B0, achieves
$"ICBHI" = 0.555 plus.minus 0.016$, marginally ahead of the best classical model (SVM-B, $0.537$)
and within the same performance tier. The four-class imbalanced structure limits
all methods approximately equally on this task.

The cross-modal analysis, the project's principal novelty, combined ranking
comparisons, direct transfer experiments and a joint multi-task probe. Classical
method rankings do not transfer reliably: XGBoost dominates on heart but falls to
third on lung, while SVM is competitive on both modalities (the heart–lung rank
correlation is weak and non-significant whether measured on the four core classifiers
(Spearman $rho = 0.60$ / $0.00$ / $0.24$ for set A / B / pooled) or on an expanded
nine-classifier panel ($rho = 0.42$ / $0.13$ / $0.32$, all $p > 0.19$), where XGBoost
falls from rank 1 on heart to rank 7 of 9 on lung). The widened panel also exposes a
model-family-by-modality interaction, tree ensembles favouring heart and linear models
favouring lung, that underlies the non-transfer. A paired McNemar test on the individual
predictions confirms the inversion is real rather than a rounding effect: XGBoost is
significantly ahead of SVM on heart ($chi^2 = 17.5$, $p approx 1.5 times 10^(-5)$), yet on
lung its raw-accuracy edge does not convert into the balanced-metric win, which SVM
holds. Deep cross-modal transfer, by contrast, is
benign and roughly symmetric once evaluated over three seeds: under fine-tuning both
directions reach in-domain performance, and a transfer asymmetry suggested by a single
seed did not survive multi-seed averaging, a cautionary result that underscores the
value of seed-robust evaluation. Joint multi-task training holds both modalities near
per-modality performance with only small, backbone-dependent trade-offs. It is thus
method _rankings_, not learned features, that fail to transfer; modality-specific
classifier selection remains necessary even within a shared pipeline.

An extension to the CirCor DigiScope 2022 dataset, on a genuinely patient-level
split, added a finer heart task, murmur detection, and reversed the
classical-versus-deep ordering seen on CinC: the deep SmallCNN reached
$"MAcc" = 0.810 plus.minus 0.007$, clearly ahead of the best classical model ($0.688$),
showing that which method family wins is itself task-dependent.
The arterial sub-study established that the pipeline generalises in principle to
a third modality, demonstrated end-to-end on a synthetic arterial-band benchmark
($"MAcc" approx 0.80$ on a deliberately hard, purely synthetic contrast with no clinical
weight), but that empirical clinical work is blocked by the absence of any openly
licensed, labelled dataset of arterial bruits. A pretrained Audio Spectrogram Transformer (AST) was
fine-tuned on both modalities over three seeds: it gave the single best lung score in the
study ($"ICBHI" = 0.594 plus.minus 0.023$) and a recall-leaning heart model
($"MAcc" = 0.864 plus.minus 0.006$, $"Se" = 0.928$), though under its epoch-limited budget
it does not displace the EfficientNet-B0 and classical baselines on heart.

The work's positive outcomes are a leakage-safe, fully reproducible comparison
across two modalities and an original cross-modal analysis; its negative outcomes
are recorded honestly below.

#heading(level: 2, numbering: none, outlined: true)[Limitations]

The following limitations bound every claim in this report.

- _Dataset scope._ Only two open datasets were used (PhysioNet/CinC 2016 and
  ICBHI 2017). Results may not transfer to recordings from other devices,
  populations or acquisition conditions.
- _No clinical validation._ The study is a methodological comparison on
  retrospective public data. No claim is made about clinical accuracy, safety,
  diagnostic equivalence to a physician, or readiness for deployment. The
  intended framing throughout is screening and triage support, not diagnosis.
- _Label noise and class imbalance._ Both datasets are imbalanced and carry
  annotation noise; the smallest lung class ("both") is especially scarce, which
  limits how reliably it can be learned and reported.
- _No open arterial data._ Arterial-bruit diagnosis could only be treated
  analytically, because no suitable open dataset exists; the corresponding clinical
  claims are conceptual, not empirical. The synthetic proof-of-concept demonstrates only
  that the pipeline runs end-to-end on arterial-band audio; its data are artificial, its
  difficulty is set by hand, and it carries no clinical weight.
- _Scope of deep models._ The deep-learning comparison covers a compact CNN, an
  EfficientNet-B0 transfer model and a fine-tuned Audio Spectrogram Transformer. The
  AST was trained over three seeds but for only a few epochs per seed under the compute
  budget, so its scores (best on lung, recall-leaning on heart) are a lower bound on its
  ceiling, limited by the epoch budget rather than by seed variance.
- _Binary heart tasks._ Both heart tasks are binary: normal versus abnormal on CinC
  and murmur present versus absent on CirCor. Neither grades or localises a specific
  pathology (for example individual murmur timing, shape or grade), which the richer
  CirCor metadata would in principle support but which is left to future work.
- _Split granularity for heart._ The public CinC 2016 release provides no complete
  recording-to-subject mapping, so the heart split is grouped at the recording level
  rather than the patient level. This eliminates window-level leakage (all windows of
  a recording stay on one side) but cannot rule out the same subject contributing
  recordings to both sides. The ICBHI lung split and the CirCor heart-murmur split are
  both genuinely patient-independent, so the patient-level CirCor result complements
  the recording-level CinC evaluation.

#heading(level: 2, numbering: none, outlined: true)[Future work]

Several extensions follow naturally. The Audio Spectrogram Transformer, here fine-tuned
for only a few epochs per seed, should be trained to convergence at a larger epoch budget
to confirm whether its three-seed lead on lung sounds and its recall-leaning heart
behaviour widen against the EfficientNet-B0 baseline. Richer heart-sound labels
(for example the murmur metadata of the CirCor DigiScope database) would turn the
binary task into graded murmur classification. The joint multi-task architecture
studied here could be scaled with larger shared encoders to close the remaining
heart trade-off. Finally, if an openly licensed arterial-bruit dataset
becomes available, the existing pipeline could be applied to a third modality with
only a change of passband and configuration, completing the
cardiorespiratory-and-arterial scope of the project's title.
