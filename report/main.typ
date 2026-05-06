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
  set block(above: 1.2em, below: 0.8em)
  set text(weight: "bold")
  it
}

// ---- Figures: caption BELOW, centred; "Figure N — Name" (Annex 7 §1.4.1) -----
#set figure(numbering: "1")
#show figure.caption: it => [
  #set text(size: 11pt)
  #set align(center)
  #it
]
#set figure.caption(separator: [ — ], position: bottom)

// ---- Tables: caption ABOVE, left, no indent; 10 pt; "Table N — Name" ---------
// (Per Annex 7 §1.4.2 the table title is placed above the table, left-aligned.)
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.where(kind: table): set figure(supplement: [Table])
#show figure.where(kind: table): it => {
  set text(size: 10pt)
  set par(leading: 0.65em, first-line-indent: 0pt)
  it
}

// ---- Equations: numbered, right-aligned number in parentheses (Annex 7 §1.4.3)
#set math.equation(numbering: "(1)")

// ---- Raw/code blocks: monospace 10 pt, no indent, single spacing -------------
#show raw: set text(font: mono-font, size: 10pt)
#show raw.where(block: true): block.with(
  fill: luma(245),
  inset: 6pt,
  width: 100%,
  radius: 2pt,
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
  #v(3.5cm)
  #text(size: 13pt)[Research Project Report on the Topic:] \
  #v(0.5cm)
  #text(size: 16pt, weight: "bold")[
    Diagnosis of Pathologies of the Cardiorespiratory System and Main
    Arteries Based on the Analysis of Sound Data
  ]
  #v(3cm)

  #set align(left)
  #pad(left: 1.5cm)[
    #text(size: 12pt)[*Fulfilled by:*] \
    #text(size: 12pt)[Student of the Group БПАД244] \
    #text(size: 12pt)[Tsember Andrei Alekseevich] \
    #v(1.2cm)
    #text(size: 12pt)[*Assessed by the Project Supervisor:*] \
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
