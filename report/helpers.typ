// helpers.typ — shared helper functions imported by every section file.
// (Typst `#include` does NOT propagate top-level `#let` bindings into the
// included file's scope, so section files `#import "../helpers.typ": *`.)

// TODO / CITE / MEMBER highlight helpers — render placeholders in distinct
// colours so reviewers can find every spot needing a real number or reference.
#let TODO(body) = text(fill: rgb("#b00020"), weight: "bold")[[TODO: #body]]
#let CITE(body) = text(fill: rgb("#0050b0"), weight: "bold")[[CITE: #body]]
#let MEMBER(id) = text(fill: rgb("#555555"), raw("<MEMBER_" + id + ">"))

// Team-role note block (Annex 5 §2.3): placed at the end of the Introduction and
// at the end of every chapter to attribute the work to the responsible members.
#let team-note(body) = block(
  width: 100%,
  inset: (top: 6pt),
  text(size: 11pt, style: "italic")[*Distribution of work.* #body],
)
