// 06-conclusion.typ — Conclusion (Annex 5 §8). Annex 7 §1.2: Conclusion has no
// per-chapter team-role note (Annex 5 §2.3 excludes Conclusion). DRAFTED prose
// structured as findings-per-objective, with [TODO] only where a final number is
// required, plus the MANDATORY Limitations section and future work. The
// overclaiming test is applied: no clinical-validation or deployment claims.

#heading(numbering: none, outlined: true)[Conclusion]

This project set out to compare, honestly and reproducibly, classical and
deep-learning methods for diagnosing cardiorespiratory pathologies from
auscultation sound, under a single leakage-safe protocol spanning two modalities.
The objectives stated in the Introduction were met as follows.

A reproducible, configuration-driven pipeline shared by the heart and lung
modalities was assembled, with fixed random seeds, pinned dependencies,
leakage-safe grouped splits (patient-level for lung, recording-level for heart) and
an explicit zero-leakage assertion executed before every training run. On heart sounds the best classical configuration — XGBoost with the
MFCC+Δ+ΔΔ + spectral-statistics feature set — reached MAcc = 0.903
(Se = 0.946, Sp = 0.859). After HPO tuning (128-trial bounded random search,
val-only selection), EfficientNet-B0 reaches MAcc = 0.898 ± 0.008 (three seeds),
bringing it comparable with the best classical model: the gap of 0.005 falls
within one standard deviation, demonstrating that deep learning closes the
classical advantage on heart sounds when given equivalent tuning effort.
On lung sounds the models were evaluated at the cycle level with the official
ICBHI score. The best configuration — HPO-tuned EfficientNet-B0 — achieves
ICBHI = 0.555 ± 0.016, marginally ahead of the best classical model (SVM-B, 0.537)
and within the same performance tier. The four-class imbalanced structure limits
all methods approximately equally on this task.

The cross-modal analysis — the project's principal novelty — combined ranking
comparisons, direct transfer experiments and a joint multi-task probe. Classical
method rankings transfer only partially: XGBoost dominates on heart but falls to
third on lung, while SVM is competitive on both modalities (Spearman ρ = 0.60,
p = 0.40, n = 4 classical methods, feature set A). Deep cross-modal transfer is
asymmetric: lung-pretrained features transfer strongly to heart (near in-domain
performance), but heart-pretrained features do not transfer to lung. Joint
multi-task training preserves both modalities with a slight heart trade-off,
consistent with shared-capacity constraints. These findings demonstrate that
modality-specific classifier selection is necessary even within a shared pipeline,
and quantify the cost of a naive one-model-all-modalities deployment.
The arterial sub-study established that the pipeline generalises in principle to
a third modality but is blocked by the absence of any openly licensed, labelled
dataset of arterial bruits. An Audio Spectrogram Transformer (AST) was integrated
and verified at the code level; a complete fine-tune-and-evaluation run was not
finalised within the project's time budget and is reserved for future work.

The work's positive outcomes are a leakage-safe, fully reproducible comparison
across two modalities and an original cross-modal analysis; its negative outcomes
are recorded honestly below.

#heading(level: 2, numbering: none, outlined: true)[Limitations]

The following limitations bound every claim in this report.

- *Dataset scope.* Only two open datasets were used (PhysioNet/CinC 2016 and
  ICBHI 2017). Results may not transfer to recordings from other devices,
  populations or acquisition conditions.
- *No clinical validation.* The study is a methodological comparison on
  retrospective public data. No claim is made about clinical accuracy, safety,
  diagnostic equivalence to a physician, or readiness for deployment. The
  intended framing throughout is screening and triage support, not diagnosis.
- *Label noise and class imbalance.* Both datasets are imbalanced and carry
  annotation noise; the smallest lung class ("both") is especially scarce, which
  limits how reliably it can be learned and reported.
- *No open arterial data.* Arterial-bruit diagnosis could only be treated
  analytically, because no suitable open dataset exists; the corresponding claims
  are conceptual, not empirical.
- *Compute constraints.* Deep-learning experiments were HPO-tuned on GPU where
  feasible; however, the Audio Spectrogram Transformer (AST) fine-tune was not
  completed within the project's time budget and is reported as a methodological
  extension rather than a headline result.
- *Binary heart task.* The heart task is binary (normal versus abnormal) and does
  not localise or grade specific pathologies such as individual murmur types.
- *Split granularity for heart.* The public CinC 2016 release provides no complete
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
heart trade-off. Most importantly, if an openly licensed arterial-bruit dataset
becomes available, the existing pipeline could be applied to a third modality with
only a change of passband and configuration, completing the
cardiorespiratory-and-arterial scope of the project's title.
