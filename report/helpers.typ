// helpers.typ — shared helper functions imported by every section file.
// (Typst `#include` does NOT propagate top-level `#let` bindings into the
// included file's scope, so section files `#import "../helpers.typ": *`.)

// TODO / CITE / MEMBER highlight helpers — render placeholders in distinct
// colours so reviewers can find every spot needing a real number or reference.
#let TODO(body) = text(fill: rgb("#b00020"), weight: "bold")[[TODO: #body]]
#let CITE(body) = text(fill: rgb("#0050b0"), weight: "bold")[[CITE: #body]]
#let MEMBER(id) = text(fill: rgb("#555555"), raw("<MEMBER_" + id + ">"))

// Project group with a single member (the sole author): there is no work to
// distribute among multiple people, so the per-chapter team-role note renders
// nothing. The macro is kept as a no-op so existing #team-note[...] call sites
// across the chapters need no edits.
#let team-note(body) = none
