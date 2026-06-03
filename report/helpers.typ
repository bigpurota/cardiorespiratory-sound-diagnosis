
#let TODO(body) = text(fill: rgb("#b00020"), weight: "bold")[[TODO: #body]]
#let CITE(body) = text(fill: rgb("#0050b0"), weight: "bold")[[CITE: #body]]
#let MEMBER(id) = text(fill: rgb("#555555"), raw("<MEMBER_" + id + ">"))

#let team-note(body) = none

#let annex-mode = state("annex-mode", false)
