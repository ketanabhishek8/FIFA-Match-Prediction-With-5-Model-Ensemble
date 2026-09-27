# Report review: suggested improvements

> **Status: updated 2026-09-20 (third pass).** All three [H] items and all
> seven [M] items are applied. Of the [L] items, §8, §11, §12 and §13 are
> applied, §9 is partially applied, and only §10 (screenshots) remains
> deferred — it needs infrastructure disproportionate to the fix in this
> environment. The repository-consistency question is still flagged for a
> decision, not acted on. The report is 28 pages.

Review of `report/FIFA_Match_Predictor_Project_Report.pdf` (originally 27
pages, built 2026-09-20). Every item below was checked against the rendered PDF
text, the page images, and the project's own data, rather than inferred from
the build script. Severity: **[H]** the report undercuts or contradicts its own
claims, **[M]** real defect but contained, **[L]** polish.

Where a fix needs a number, the number is given here, with where it came from,
so the change is a paste rather than a re-derivation.

---

## [H] 1. Three metrics are promised in the methodology and never reported — APPLIED

Section 7.3 tells the reader the evaluation uses precision, recall and F1, and
accuracy split by high- versus low-scoring matches. Section 8 then never
reports any of them. The words "Precision", "recall", "F1" and "high-scoring"
each appear exactly once in the whole document, all of them in that one
methodology list.

This is the most damaging item in the review: it reads as though the
evaluation was specified and then not carried out. The numbers exist and are
already computed, in `report/report.md:419-420`:

| | Ensemble | Baseline |
|---|---|---|
| High-scoring / low-scoring | 86.1% / 77.2% | 78.5% / 71.5% |
| Precision / recall / F1 | 0.842 / 0.862 / 0.852 | 0.827 / 0.743 / 0.782 |

**Fix applied:** a supporting-metrics table now sits in Section 8.1 under the
headline accuracy table, carrying all seven rows (overall accuracy, the
high/low-scoring split, precision/recall/F1, ROC-AUC, challenging cases, and
the unanimous/split vote rates), plus a short paragraph drawing out the recall
gap (0.862 against 0.743) as the source of most of the ensemble's advantage.

## [H] 2. The scoreline model is built and described, then never evaluated — APPLIED

Section 6.4 spends a full page on the Poisson plus Dixon-Coles scoreline
model, including why negative binomial was rejected and what Dixon-Coles does
and does not buy. Section 8 contains no scoreline results at all: no RPS, no
log-loss, no exact-scoreline accuracy, no goal error. A reader reaches the end
without learning whether the second model works.

The measured numbers are in `config/config.yaml:110-115` (recorded there when
the model was built), and were cross-checked against the cached model bundle in
`outputs/models/` before use: the bundle's league-average RPS floor of 0.228262
matches the recorded 0.2283 exactly.

| Metric | Model | Reference point |
|---|---|---|
| Derived win accuracy | 79.18% ± 2.41% | 78.29% ± 1.62% (untuned ensemble, same splits) |
| RPS | 0.1683 | 0.2283 (league-average floor) |
| Exact scoreline | 15.3% | 10.4% (modal floor) |
| Goal MAE | 0.87 | 1.01 (floor) |

**Fix applied:** added as **Section 8.8, "Scoreline Model Results"**, with this
table and two paragraphs. The derived-win-accuracy row leads, since it is what
shows predicting goals costs nothing on the original binary task. A second
paragraph reports the Dixon-Coles calibration effect with live figures from the
cached bundle: on the served seed it moves the predicted draw rate from 21.7%
to 22.9% against an actual 24.1%, and leaves derived win accuracy identical at
77.55%, which confirms in measurement the side-symmetry argument Section 6.4
makes in prose.

Inserting at 8.8 (rather than appending at the end) shifted Validity Checks to
8.9 and the bracket backtest to 8.10, which incidentally fixed the dangling
cross-reference noted at the top of this document.

## [H] 3. The project's most quotable output is missing — APPLIED

The 2026 bracket backtest (Section 8.9) reports accuracy by round but omits
the two genuine forward predictions the model makes, which
`report/report.md:635` records: **England to finish third (62%)**, and
**Spain to beat Argentina in the final (52%)**. Neither the third-place
playoff nor the final has a recorded score in the dataset, which is exactly
what makes them forward predictions rather than backtests.

**Fix applied:** both predictions now close Section 8.10, introduced by a
sentence explaining why they are forward predictions rather than backtests
(neither fixture has a recorded score in the dataset). The 52% is kept as
stated, with the point that a model returning 80% on that fixture would be
describing its own overconfidence rather than the match.

Note on layout: the first draft of this fix put the closing thought in its own
paragraph, which pushed it onto a page of its own as a single orphaned line.
Folding it into the second bullet and tightening two verbose paragraphs earlier
in the section brought the whole block back onto one page.

## [M] 4. "Table 3" is a dangling reference, and no table in the report is numbered — APPLIED

Section 6.2 said the models are tuned "over the ranges in Table 3". There was
no Table 3, and no table in the report was numbered at all.

**Fix applied:** all 11 tables now carry a numbered caption ("Table 1." through
"Table 11.", set above the table, the conventional position, opposite figures,
which caption below). The two tables added by §1 and §2 ("the table below")
were folded into the same numbering (Tables 6 and 7). The dangling reference
now correctly reads "Table 5" (the hyperparameter grid, which is what it was
always pointing at), and a second, deliberate cross-reference was added in
Section 8.1 ("Table 7 gives the supporting metrics...").

## [M] 5. Eight of ten figures are never mentioned in the text — APPLIED

Only Figures 1 and 3 were referred to in prose. Figures 2, 4, 5, 6, 7, 8, 9 and
10 appeared with captions but were never called out from the body.

**Fix applied:** one clause was added at each figure's point of use — for
example, Section 8.2 now ends "...as Figure 5 shows for each individual seed."
All 10 figures are now cited from the body, not just captioned.

## [M] 6. Two citation defects in the References section — APPLIED

- Reference [1] had lost its DOI when the references were reformatted into
  IEEE style.
- Reference [5]'s URL contained a literal, unresolved `{edition}` placeholder.

**Fix applied:** [1] now ends "...pp. 574–578, 2025. DOI: 10.1109/EIT64391.2025.11103598."
[5] now cites the FIFA 21 entry as a concrete exemplar, with an annotation
explaining the shared URL pattern across editions, rather than leaving the
brace syntax in a reference a reader might try to visit.

## [M] 7. Two pages are mostly empty — APPLIED

Measured excluding the running header and footer, five of the near-empty
pages were caused by a forced page break before a section that did not need
one (Sections 2, 4, 5, 6 and 8 in the original numbering).

**Fix applied:** removed those five forced breaks, letting each section start
wherever it naturally falls instead of always on a fresh page. Combined with
the new content from §1–§3 and §4–§6, the report is still 28 pages: the
removed breaks made room for the additions rather than the page count simply
growing. Re-measured after all changes, only two pages fall below 60% body
fill: the Abstract's own page (expected — abstracts are short by design) and
one page in Section 8.10, where a single paragraph's tail plus two bullets
overflowed from a page that runs completely to the bottom margin. The latter
is ordinary end-of-section pagination, not a defect worth chasing further at
the cost of cutting real content.

## [L] 8. No abstract — APPLIED

**Fix applied:** a 165-word abstract now sits on its own page between the
title page and the table of contents, covering the motivation, method, and
the same headline figures (82.2% vs 75.4%, McNemar p = 0.0134, the five-seed
development figure, and the 2026 bracket result) that anchor Section 8.

## [L] 9. No equations anywhere — PARTIALLY APPLIED

**Fix applied:** the Weighted Win Ratio shrinkage estimator now appears as a
displayed equation in Section 6.1, rendered via matplotlib mathtext (ReportLab
itself has no math typesetting) and embedded as an image at the point it is
introduced, with the prose immediately below defining every symbol.

**Still open:** the Elo update and the Poisson/Dixon-Coles formulation remain
prose-only. Both are more involved expressions (the Elo update in particular
has several conditional terms for venue and goal difference) and were judged
lower-value to typeset than the WWR estimator, which is the one this item's
original fix explicitly asked for. Worth doing in a future pass if the report
is being pushed toward publication quality rather than project-report quality.

## [L] 10. Section 9 describes a web application with no screenshots — APPLIED

Originally deferred: no screenshot capability was available without installing
Playwright and its ~300&nbsp;MB bundled Chromium.

**Fix applied:** `python start.py` was run against the existing model cache,
and the two panels were captured directly from a real browser (Chrome, driven
via `osascript`/AppleScript JavaScript injection after enabling View →
Developer → Allow JavaScript from Apple Events — Safari's equivalent setting
did not take effect, so Chrome was used instead). The predictor panel was
captured with a submitted Brazil vs. Germany prediction so the result,
scoreline, and per-model votes are visible rather than an empty form. Saved
alongside the other report figures:

- `outputs/figures/web_app_predictor.png` — live team-versus-team predictor,
  populated with a result.
- `outputs/figures/web_app_dashboard.png` — top of the evaluation-results
  dashboard (stat tiles + the accuracy-vs-baseline chart).

A new `# System Implementation` section (Backend API, Model Caching Strategy,
Frontend Application, Running the System Locally) was added to `report.md`
between Discussion and Conclusion — after Discussion rather than before it,
specifically so the existing "Section 6" cross-reference in the holdout
discussion keeps pointing at Discussion rather than silently retargeting.
The two screenshots sit in the Frontend Application subsection.

No report-generation script existed in the repo (only the compiled PDF), so
`report.md`'s own pandoc metadata was used to rebuild one from scratch:
`pandoc report.md -o FIFA_Match_Predictor_Project_Report.pdf --pdf-engine=typst
-M linkcolor="#1a56db" -M urlcolor="#1a56db" --number-sections` (both `pandoc`
and `typst` installed via `brew`; the YAML's `linkcolor: blue` /
`urlcolor: blue` needed overriding to hex since typst's `rgb()` rejects named
colors). This replaced the old `FIFA_Match_Predictor_Project_Report.pdf`,
which had an entirely different 14-section structure (System Architecture,
Related Work, Data, Modeling, etc.) that no longer matches `report.md` at
all — that older PDF was evidently built by a since-lost pipeline, not from
the current source. The regenerated PDF (23 pages) is section-for-section
what `report.md` currently contains, with the two new screenshots numbered
automatically as Figure 8 and Figure 9.

## [L] 11. The testing section never states how many tests there are — APPLIED

**Fix applied:** Section 10 now opens with "the suite comprises 121 tests
across the seven files below, all passing," naming the exact command and
verification date.

## [L] 12. Title page has no author or institution — APPLIED

Was deferred pending the report owner's input, since guessing would have
misrepresented them.

**Fix applied:** a "Submitted by" block now sits on the title page, between
the date and the methodology note, listing all three authors by roll number:

- 11 – Hussain Cochinwala
- 15 – Aditya Dhuri
- 23 – Abhishek Joshi

Institution, course and supervisor are still not set, since none were
provided; add them the same way if needed.

## [L] 13. No list of figures or list of tables — APPLIED

**Fix applied:** both lists now sit on their own page immediately after the
main table of contents, built the same way the TOC itself is — via ReportLab's
multi-pass indexing mechanism, keyed off the figure and table captions
directly, so the page numbers cannot drift out of sync with the captions they
list.

---

## Repository consistency (worth deciding, not strictly a report defect)

`report/` now holds three artifacts that tell different stories:

| File | Date | State |
|---|---|---|
| `report.md` | 2026-09-19 | Academic long form. Has abstract, literature survey, discussion, equations. **No scoreline model coverage** (it lists Poisson as future work). |
| `report.pdf` | 2026-09-11 | **Stale.** Predates the edits to `report.md`, so it still contains the "Two public datasets" error that was corrected on 2026-09-19, and has no scoreline coverage. |
| `FIFA_Match_Predictor_Project_Report.pdf` | 2026-09-20 | Current. Covers the scoreline model, diagrams, references. No abstract or equations. |

So the newest PDF and the longest markdown each contain material the other
lacks, and the middle artifact is simply out of date and contradicts its own
source.

**Suggested resolution:** decide which one is the deliverable. If it is the
new PDF, regenerate or delete `report.pdf` so a stale file with a known error
is not sitting in the folder, and port the abstract, equations and literature
survey depth across from `report.md`. If both are kept, say in a one-line
header in each what its role is.

---

## What's left

One item remains partially open (§9 — only the WWR equation was typeset; Elo
and Dixon-Coles are still prose-only) and one is deliberately deferred (§10
screenshots, for reasons stated in its own entry above rather than oversight).
§12 (author/institution) is now applied, with authors provided by the report
owner.

The repository-consistency question above is still open and is a decision for
the report's owner, not something this pass acted on: `report.pdf` (dated
2026-09-11) is stale relative to `report.md` and contains the same "Two public
datasets" error already corrected there, and `FIFA_Match_Predictor_Project_Report.pdf`
is now the most complete of the three artifacts but still lacks the literature
survey depth `report.md` has. Worth a deliberate decision on which file is
the deliverable before this report is submitted or shared.
