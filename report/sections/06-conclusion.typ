
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
classical advantage on heart sounds once the tuning effort is equalised.
On lung sounds the models were evaluated at the cycle level with the official
ICBHI score. The best configuration, HPO-tuned EfficientNet-B0, achieves
$"ICBHI" = 0.555 plus.minus 0.016$, marginally ahead of the best classical model (SVM-B, $0.537$)
and within the same performance tier. The four-class imbalanced structure limits
all methods approximately equally on this task.

The cross-modal analysis, the project's principal novelty, combined ranking
comparisons, direct transfer experiments and a joint multi-task probe. Classical
method rankings transfer only partially: XGBoost dominates on heart but falls to
third on lung, while SVM is competitive on both modalities (Spearman $rho = 0.60$,
$p = 0.40$, $n = 4$ classical methods, feature set A). Deep cross-modal transfer is
asymmetric: lung-pretrained features transfer strongly to heart (near in-domain
performance), but heart-pretrained features do not transfer to lung. Joint
multi-task training preserves both modalities with a slight heart trade-off,
consistent with shared-capacity constraints. Modality-specific classifier selection
is therefore necessary even within a shared pipeline, and the experiments quantify
the cost of a naive one-model-all-modalities deployment.
The arterial sub-study established that the pipeline generalises in principle to
a third modality but is blocked by the absence of any openly licensed, labelled
dataset of arterial bruits. An Audio Spectrogram Transformer (AST) was integrated
and verified at the code level; its full fine-tuning and evaluation are reserved
for future work.

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
  analytically, because no suitable open dataset exists; the corresponding claims
  are conceptual, not empirical.
- _Scope of deep models._ The deep-learning comparison covers a compact CNN and an
  EfficientNet-B0 transfer model. The Audio Spectrogram Transformer is integrated as
  a methodological extension, with full fine-tuning left to future work rather than
  reported as a headline result.
- _Binary heart task._ The heart task is binary (normal versus abnormal) and does
  not localise or grade specific pathologies such as individual murmur types.
- _Split granularity for heart._ The public CinC 2016 release provides no complete
  recording-to-subject mapping, so the heart split is grouped at the recording level
  rather than the patient level. This eliminates window-level leakage (all windows of
  a recording stay on one side) but cannot rule out the same subject contributing
  recordings to both sides. The ICBHI lung split is genuinely patient-independent.

#heading(level: 2, numbering: none, outlined: true)[Future work]

Several extensions follow naturally. The Audio Spectrogram Transformer (AST),
already integrated into the pipeline, should be fine-tuned and evaluated on both
modalities to determine whether self-supervised AudioSet pretraining offers a
further gain over the EfficientNet-B0 transfer baseline. Richer heart-sound labels
(for example the murmur metadata of the CirCor DigiScope database) would turn the
binary task into graded murmur classification. The joint multi-task architecture
studied here could be scaled with larger shared encoders to close the remaining
heart trade-off. Finally, if an openly licensed arterial-bruit dataset
becomes available, the existing pipeline could be applied to a third modality with
only a change of passband and configuration, completing the
cardiorespiratory-and-arterial scope of the project's title.
