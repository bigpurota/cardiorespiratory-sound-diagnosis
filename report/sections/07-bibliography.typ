// 07-bibliography.typ — Bibliography (Annex 5 §9; Annex 7 §1.2, §1.4.4).
// NOT numbered as a chapter; starts a new page. The IEEE style numbers entries in
// order of first [n] citation, which is exactly the Annex-7 square-bracket
// convention. Entries are seeded in report/refs.bib from the REAL sources listed
// in .planning/research/SUMMARY.md §Sources — no fabricated references. Gaps that
// still need a citation are marked [TODO: add ref] at the point of use in the
// chapters, not invented here.

#bibliography(
  "../refs.bib",
  title: [Bibliography],
  style: "ieee",
  full: false,   // only entries actually cited appear, in order of appearance
)
