// 05-ch4-novelty.typ — Chapter 4: Cross-modal analysis (novelty) + arterial
// sub-study. SKELETON. The cross-modal analysis reads the already-saved results
// (no new experiments). Heatmap and joint-training table slots carry [TODO].

#import "../helpers.typ": *

= Cross-modal analysis and arterial sub-study

This chapter develops the project's distinctive contribution: an analysis of how
method families transfer across auscultation modalities, computed from the
results of Chapter 3 without additional experiments, followed by an analytical
treatment of arterial sounds for which no open dataset exists.

== Transfer of method rankings across modalities

Because heart and lung models are evaluated under one protocol and one metric
family, their *rankings* can be compared directly. We quantify the agreement of
the method ordering between modalities with a rank-correlation statistic and
visualise the per-modality scores as a heatmap.

#text(fill: rgb("#b00020"), weight: "bold")[[TODO: report the Spearman rank
correlation of method rankings between heart and lung, and interpret it — do the
strong classical models on heart also lead on lung, or does the harder lung task
reorder them?]]

#figure(
  kind: image,
  rect(width: 70%, height: 4.5cm, stroke: 0.5pt + gray)[
    #align(center + horizon)[#text(fill: gray)[Figure slot — cross-modal score heatmap (`results/figures/...`)]]
  ],
  caption: [Cross-modal heatmap of model scores by modality (placeholder)],
) <fig-crossmodal>

== Joint-training probe

As a single additional probe, a shared feature extractor is trained on the pooled
heart-and-lung data and evaluated per modality, to test whether a common
representation helps or hurts relative to per-modality training.

#figure(
  caption: [Per-modality scores under per-modality versus joint training (placeholder)],
  table(
    columns: 3,
    align: (left, center, center),
    table.header([*Setting*], [*Heart (MAcc)*], [*Lung (ICBHI)*]),
    [Per-modality training], [TODO], [TODO],
    [Joint (pooled) training], [TODO], [TODO],
  ),
) <tab-joint>

#text(fill: rgb("#b00020"), weight: "bold")[[TODO: interpret — the honest claim is
"the same pipeline applies to both modalities", NOT "heart features transfer to
lung diagnosis". Keep the claim bounded.]]

== Arterial sounds: an analytical sub-study

Arterial bruits — the turbulent flow sounds of a stenosed carotid artery — are
part of the project's topic but cannot be addressed by supervised learning here,
because an exhaustive search found no openly licensed, labelled dataset suitable
for training @stroke2024 @carotid_pmc. Reported systems rely on proprietary
recordings that are not released.

We therefore treat arterial sound analytically. The same preprocessing chain
(resampling, Butterworth bandpass, peak normalisation) and the same feature
representations would apply directly, with a bruit-appropriate passband; the
methodological obstacle is data, not pipeline. This sub-study documents the
acoustic characteristics of bruits, the reasons open data is absent, and exactly
what would be required to extend the unified pipeline to a third modality, which
we record as future work rather than as an empirical result.

#text(fill: rgb("#b00020"), weight: "bold")[[TODO: 2–3 paragraphs of original
synthesis on carotid-bruit acoustics and the data-availability situation, with
the real citations already seeded in refs.bib.]]

=== Chapter summary

The cross-modal analysis shows #text(fill: rgb("#b00020"), weight: "bold")[[TODO:
the headline cross-modal finding]], and the arterial sub-study establishes that
the pipeline generalises in principle but is blocked by the absence of open data.

#team-note[
  The cross-modality transfer analysis and joint-training probe were carried out
  by #MEMBER("10"); the arterial analytical sub-study was prepared by #MEMBER("11")
  with the clinical-literature synthesis by #MEMBER("03").
]
