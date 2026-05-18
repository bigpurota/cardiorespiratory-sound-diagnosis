// =============================================================================
// main.typ — Master document for the HSE DSBA Year-2 research report.
//
// Project: "Diagnosis of pathologies of the cardiorespiratory system and main
// arteries based on the analysis of sound data."
//
// This file encodes the Annex-7 formatting contract (verified against
// "ПАД_Приложение 7.pdf" on disk) and the Annex-5 structure (verified against
// "ПАД_Приложение 5.pdf"). It #includes the section files in report/sections/.
//
// Annex-7 typography (verified):
//   - Margins: left 25 mm, right 10 mm, top 20 mm, bottom 20 mm.
//   - Body font: Times New Roman, 12 pt; line spacing 1.5; justified;
//     first-line paragraph indent 1.25 cm.
//   - Page numbers: bottom-centre, consecutive; the title page is counted but
//     left UNNUMBERED (numbering printed from the page after the title).
//   - Headings numbered 1, 1.1, 1.1.1 with NO trailing period; the heading is
//     never separated from its body text (kept-with-next).
//   - Abstract and Bibliography are NOT numbered and each starts a new page;
//     Contents, Abstract, Introduction and Bibliography start on a new page.
//   - Annexes are lettered alphabetically (Annex A, B, C, ...).
//   - Tables: caption ABOVE, left-aligned, no indent, "Table N — Name",
//     Times New Roman 10 pt, single (1.0) line spacing.
//   - Figures: caption BELOW, centred, "Figure N — Name".
//   - Formulas: centred, equation number right-aligned in parentheses.
//   - Code fragments: monospace 10 pt, no indent, single spacing.
//   - Citations: square-bracket [n], numbered in order of first appearance.
//
// FONT NOTE: "Times New Roman" is installed in this build environment (verified
// via `typst fonts`). If a future build host lacks it, the fallback list below
// degrades gracefully to "TeX Gyre Termes" / "Liberation Serif" (metric-
// compatible Times clones). Monospace falls back to a Courier/Consolas analogue.
// =============================================================================

// ---- Global state for the order-of-appearance [n] citation mechanism --------
// Typst's native `cite`/`bibliography` already numbers in order of appearance
// with the "ieee" style, which is exactly the Annex-7 [n] convention. We use a
// .bib file (report/refs.bib) and `bibliography(style: "ieee")` at the end.

// Fonts: Times New Roman is verified present in this build host. The fallbacks
// (DejaVu) are kept only so a host missing TNR still renders something; the
// metric-compatible "TeX Gyre Termes"/"Liberation Serif" clones are omitted from
// the list because they are not installed here and would only emit warnings.
#let body-font = ("Times New Roman", "DejaVu Serif")
#let mono-font = ("Courier New", "DejaVu Sans Mono")

// Shared helper functions (TODO/CITE/MEMBER/team-note) live in helpers.typ and
// are imported here for use in main.typ's own body (the title page TODO).
#import "helpers.typ": *

// ---- Page geometry + base typography (Annex 7 §1.1, §1.3) --------------------
#set page(
  paper: "a4",
  margin: (left: 25mm, right: 10mm, top: 20mm, bottom: 20mm),
  numbering: none,            // title page is unnumbered; enabled after it below
)

#set text(
  font: body-font,
  size: 12pt,
  lang: "en",
  hyphenate: false,
)

// 1.5 line spacing ≈ 0.5 * font size of extra leading; justified body.
#set par(
  justify: true,
  leading: 0.85em,            // ≈ 1.5 line spacing for a 12 pt body
  first-line-indent: (amount: 1.25cm, all: true),
  spacing: 0.85em,
)

// ---- Heading numbering (Annex 7 §1.2): 1, 1.1, 1.1.1; no trailing period -----
#set heading(numbering: "1.1")
#show heading: it => {
  // Keep heading with following text (never orphaned at a page bottom).
  // Graduated sizes give a clear, paper-like hierarchy. The BODY text stays
  // Times New Roman 12 (Annex 7 §1.3) — headings are not "main text", so larger
  // section titles remain compliant. Top-level section headers are set to 16 pt
  // to also satisfy the DSBA Regulation (Methodological Guidelines, App. 8 §2
  // item 5: section headers 16–18 pt), on which Annex 7 is silent.
  set block(above: 1.3em, below: 0.75em)
  let sz = if it.level == 1 { 16pt } else if it.level == 2 { 14pt } else { 13pt }
  set text(weight: "bold", size: sz)
  // Per-section figure/table numbering: reset both counters at every top-level
  // heading so figures/tables number as <chapter>.<n> (and <Letter>.<n> in annexes).
  if it.level == 1 {
    counter(figure.where(kind: image)).update(0)
    counter(figure.where(kind: table)).update(0)
  }
  it
}

// ---- Figures: caption BELOW, centred; per-section number "Figure 3.1" --------
// Official CourseworkTemplateEng numbers figures/tables by section
// (\counterwithin{figure}{section}). We replicate: <chapter>.<n> in the body and
// <Letter>.<n> in the annexes (annex-mode from helpers.typ flips the scheme).
#set figure(numbering: (..n) => context {
  let chap = counter(heading).get().first()
  let idx = n.pos().first()
  if annex-mode.get() { numbering("A.1", chap, idx) } else { numbering("1.1", chap, idx) }
})
// Paper-style captions: a bold "Figure N –" / "Table N –" label at 10 pt, with
// the descriptive text in regular weight. Position (below figures, above tables)
// and the " – " separator stay exactly as Annex 7 §1.4 prescribes.
#show figure.caption: it => context {
  set text(size: 10pt)
  [#it.supplement #it.counter.display(it.numbering)#it.separator]
  it.body
}
#set figure.caption(separator: [ – ], position: bottom)
// Annex 7 §1.4.1: figure captions centred. Table captions are left-aligned
// (Annex 7 §1.4.2) — handled in the table show rule below.
#show figure.where(kind: image): set align(center)

// Paper-style polish (Annex-7-safe): bold the top-level Contents entries and add
// a little vertical breathing room around figures and tables.
#show outline.entry.where(level: 1): set text(weight: "bold")
#show figure: set block(above: 1.5em, below: 1.5em)

// ---- Tables: caption ABOVE, left, no indent; 10 pt; "Table N — Name" ---------
// (Per Annex 7 §1.4.2 the table title is placed above the table, left-aligned.)
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.where(kind: table): set figure(supplement: [Table])
#show figure.where(kind: table): it => {
  set text(size: 10pt)
  set par(leading: 0.65em, first-line-indent: 0pt)
  set align(left)            // Annex 7 §1.4.2: table title left-aligned, no indent
  // Clean academic ("booktabs") look: heavy rule at the top and under the header,
  // light grey separators between data rows, NO vertical lines.
  set table(
    stroke: (_, y) => (
      top: if y <= 1 { 0.9pt + black } else { 0pt },
      bottom: 0.4pt + luma(180),
    ),
    inset: (x: 6pt, y: 3.6pt),
  )
  it
}

// ---- Equations: numbered, right-aligned number in parentheses (Annex 7 §1.4.3)
#set math.equation(numbering: "(1)")
// Math font: STIX Two Math is metric-compatible with Times New Roman, so inline
// numbers/metrics ($"MAcc" = 0.903$, $0.898 plus.minus 0.008$, $rho$ …) and the
// display equations render in the SAME visual style as the TNR body — math is
// set as real math (Annex 7 §1.4.3) without a jarring font switch.
#show math.equation: set text(font: "STIX Two Math")

// ---- Raw/code blocks: monospace 10 pt, no indent, single spacing -------------
#show raw: set text(font: mono-font, size: 10pt)
#show raw.where(block: true): block.with(
  fill: luma(250),
  inset: 8pt,
  width: 100%,
  stroke: 0.5pt + luma(210),
)

// =============================================================================
// TITLE PAGE (unnumbered) — official HSE / Faculty of Computer Science
// individual Research Project format (per "Title page (Research project) ind,
// 1 supervisor.docx" from Title pages (research).zip).
// =============================================================================
#[
  #set align(center)
  #v(1.5cm)
  #text(size: 14pt)[NATIONAL RESEARCH UNIVERSITY] \
  #text(size: 14pt)[HIGHER SCHOOL OF ECONOMICS] \
  #v(0.5cm)
  #text(size: 12pt)[Faculty of Computer Science] \
  #text(size: 12pt)[Bachelor's Programme "Data Science and Business Analytics"]

  #v(1.2cm)
  // UDC is mandatory on the title page of a *research* project (per the official
  // CourseworkTemplateEng/title_kr.tex). Left-aligned, between programme and title.
  #align(left)[#text(size: 12pt)[UDC 004.85:616.12]]
  #v(2cm)

  #text(size: 13pt)[Research Project Report on the Topic:] \
  #v(0.5cm)
  #text(size: 16pt, weight: "bold")[
    Diagnosis of Pathologies of the Cardiorespiratory System and Main
    Arteries Based on the Analysis of Sound Data
  ]
  #v(3cm)

  #set align(left)
  #pad(left: 1.5cm)[
    #text(size: 12pt)[*Submitted by the Student:*] \
    #text(size: 12pt)[Group БПАД244, 2nd year of study #h(1.5cm) Tsember Andrei Alekseevich] \
    #v(1.2cm)
    #text(size: 12pt)[*Approved by the Project Supervisor:*] \
    #text(size: 12pt)[Tomashchuk Kornei Kirillovich] \
    #text(size: 12pt)[Lecturer] \
    #text(size: 12pt)[Faculty of Computer Science, HSE University]
  ]

  #v(1fr)
  #set align(center)
  #text(size: 12pt)[Moscow 2026]
]
#pagebreak()

// Enable bottom-centre page numbering from the page AFTER the title page.
#set page(numbering: "1")
#counter(page).update(2)

// =============================================================================
// CONTENTS (starts a new page; Annex 7 §1.2)
// =============================================================================
#outline(title: [Contents], depth: 3, indent: auto)
#pagebreak()

// =============================================================================
// FRONT MATTER + CHAPTERS + BACK MATTER
// =============================================================================
#include "sections/00-abstract.typ"
#pagebreak()

#include "sections/01-introduction.typ"
#pagebreak()

#include "sections/02-ch1-litreview.typ"
#pagebreak()

#include "sections/03-ch2-methods.typ"
#pagebreak()

#include "sections/04-ch3-results.typ"
#pagebreak()

#include "sections/05-ch4-novelty.typ"
#pagebreak()

#include "sections/06-conclusion.typ"
#pagebreak()

// ---- Bibliography (NOT numbered as a heading; starts a new page) -------------
#include "sections/07-bibliography.typ"
#pagebreak()

// ---- Annexes (lettered A, B, C, ...) ----------------------------------------
#include "sections/08-annexes.typ"
