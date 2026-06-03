
#let body-font = ("Times New Roman", "DejaVu Serif")
#let mono-font = ("Courier New", "DejaVu Sans Mono")

#import "helpers.typ": *

#set page(
  paper: "a4",
  margin: (left: 25mm, right: 10mm, top: 20mm, bottom: 20mm),
  numbering: none,
)

#set text(
  font: body-font,
  size: 12pt,
  lang: "en",
  hyphenate: false,
)

#set par(
  justify: true,
  leading: 1.05em,
  first-line-indent: (amount: 1.25cm, all: true),
  spacing: 1.05em,
)

#set heading(numbering: "1.1")
#show heading: it => {
  set block(above: 1.3em, below: 0.75em)
  let sz = if it.level == 1 { 16pt } else if it.level == 2 { 14pt } else { 13pt }
  set text(weight: "bold", size: sz)
  if it.level == 1 {
    counter(figure.where(kind: image)).update(0)
    counter(figure.where(kind: table)).update(0)
  }
  it
}

#set figure(numbering: (..n) => context {
  let chap = counter(heading).get().first()
  let idx = n.pos().first()
  if annex-mode.get() { numbering("A.1", chap, idx) } else { numbering("1.1", chap, idx) }
})
#show figure.caption: it => context {
  set text(size: 10pt)
  [#it.supplement #it.counter.display(it.numbering)#it.separator]
  it.body
}
#set figure.caption(separator: [ – ], position: bottom)
#show figure.where(kind: image): set align(center)

#show outline.entry.where(level: 1): set text(weight: "bold")
#show figure: set block(above: 1.5em, below: 1.5em)

#show figure.where(kind: table): set figure.caption(position: top)
#show figure.where(kind: table): set figure(supplement: [Table])
#show figure.where(kind: table): it => {
  set text(size: 10pt)
  set par(leading: 0.65em, first-line-indent: 0pt)
  set align(left)
  set table(
    stroke: (_, y) => (
      top: if y <= 1 { 0.9pt + black } else { 0pt },
      bottom: 0.4pt + luma(180),
    ),
    inset: (x: 6pt, y: 3.6pt),
  )
  it
}

#set math.equation(numbering: "(1)")
#show math.equation: set text(font: "STIX Two Math")

#show raw: set text(font: mono-font, size: 10pt)
#show raw.where(block: true): block.with(
  fill: luma(250),
  inset: 8pt,
  width: 100%,
  stroke: 0.5pt + luma(210),
)

#[
  #set align(center)
  #v(1.5cm)
  #text(size: 14pt)[NATIONAL RESEARCH UNIVERSITY] \
  #text(size: 14pt)[HIGHER SCHOOL OF ECONOMICS] \
  #v(0.5cm)
  #text(size: 12pt)[Faculty of Computer Science] \
  #text(size: 12pt)[Bachelor's Programme "Data Science and Business Analytics"]

  #v(1.2cm)
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
    #text(size: 12pt)[*Fulfilled by:*] \
    #v(0.3cm)
    #text(size: 12pt)[Student of the Group БПАД244, 2nd year of study,] \
    #text(size: 12pt)[Tsember Andrei Alekseevich] \
    #v(1.2cm)
    #text(size: 12pt)[*Assessed by the Project Supervisor:*] \
    #v(0.3cm)
    #text(size: 12pt)[Tomashchuk Kornei Kirillovich] \
    #text(size: 12pt)[Lecturer] \
    #text(size: 12pt)[Faculty of Computer Science, HSE University]
  ]

  #v(1fr)
  #set align(center)
  #text(size: 12pt)[Moscow 2026]
]
#pagebreak()

#set page(numbering: "1")
#counter(page).update(2)

#outline(title: [Contents], depth: 3, indent: auto)
#pagebreak()

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

#include "sections/07-bibliography.typ"
#pagebreak()

#include "sections/08-annexes.typ"
