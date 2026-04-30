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
patient-level splits and an explicit zero-leakage assertion executed before every
training run. On heart sounds the classical and deep models were trained and
evaluated at the recording level, with the best configuration reaching a mean
accuracy of #text(fill: rgb("#b00020"), weight: "bold")[[TODO: best heart MAcc]].
On lung sounds the models were evaluated at the cycle level with the official
ICBHI score, the best configuration reaching
#text(fill: rgb("#b00020"), weight: "bold")[[TODO: best lung ICBHI score]]. Across
both tasks the method family that performed best was
#text(fill: rgb("#b00020"), weight: "bold")[[TODO: classical vs deep, per
modality]].

The cross-modal analysis — the project's principal novelty — found that
#text(fill: rgb("#b00020"), weight: "bold")[[TODO: cross-modal finding: do method
rankings transfer between heart and lung?]]. The arterial sub-study established
that the pipeline generalises in principle to a third modality but is blocked by
the absence of any openly licensed, labelled dataset of arterial bruits.

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
- *Compute constraints.* Experiments were designed to be CPU-feasible; the
  deep-learning models were deliberately compact and lightly tuned, so the
  reported scores are honest baselines rather than tuned upper bounds.
- *Binary heart task.* The heart task is binary (normal versus abnormal) and does
  not localise or grade specific pathologies such as individual murmur types.

#heading(level: 2, numbering: none, outlined: true)[Future work]

Several extensions follow naturally. Richer heart-sound labels (for example the
murmur metadata of the CirCor DigiScope database) would turn the binary task into
graded murmur classification. Stronger deep models — audio transformers and
larger transfer backbones — could be evaluated under the same leakage-safe
protocol, on GPU, to quantify the headroom above the present baselines. The
cross-modal probe could be extended to a properly shared multi-task model. Most
importantly, if an openly licensed arterial-bruit dataset becomes available, the
existing pipeline could be applied to a third modality with only a change of
passband and configuration, completing the cardiorespiratory-and-arterial scope
of the project's title.
