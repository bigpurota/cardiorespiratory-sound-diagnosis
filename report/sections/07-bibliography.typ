// 07-bibliography.typ — Bibliography (Annex 5 §9; Annex 7 §1.2, §1.4.4).
// NOT numbered as a chapter; starts a new page. Per the DSBA Regulation
// (Methodological Guidelines, App. 8 §2 item 8) references follow Russian
// National Standard GOST R 7.0.5-2008. We use the official numeric CSL
// (order-of-citation, square-bracket [n] in-text — matching Annex 7 §1.4.4),
// with the locale set to en-US so connector terms render in English to match the
// report language. Entries are seeded in report/refs.bib from the REAL sources
// listed in .planning/research/SUMMARY.md §Sources — no fabricated references.

#bibliography(
  "../refs.bib",
  title: [Bibliography],
  style: "../gost-r-7-0-5-2008-numeric.csl",
  full: false,   // only entries actually cited appear, in order of appearance
)
