# Documentation fixes

> **Status: applied 2026-09-19.** Every item below has been fixed except **§9**,
> which is half-done — the `.gitignore` half is in, but whether the root
> `package.json` / `node_modules/` should exist at all is a call for whoever
> added them. The findings are kept as the record of what was wrong and why;
> each item's "Fix" paragraph describes what was done.

Findings from a review of `README.md`, `docs/*.md`, `report/report.md`,
`CODE_REVIEW.md`, and the in-file config comments, each checked against the
code rather than against the other docs. Severity: **[H]** the docs mislead a
reader about the results or block reproduction, **[M]** wrong or incomplete but
bounded, **[L]** hygiene / link rot.

Every item below was verified — line numbers are current as of this review.
What was *not* re-verified: the per-change accuracy deltas (`README.md:220-231`),
the "100 of 1504 matches are finals" split (`README.md:200`), and the figures in
`outputs/figures/`, none of which were re-run.

---

## [H] 1. The data-source table omits the source the headline result is built on

`README.md:126-131` lists two datasets as the project's "confirmed sources".
`src/data/collect_players.py:114-122` (`MODERN_SOURCES`) pulls two more,
because `stefanoleone992` stopped publishing after EA FC 24:

| Year | Slug | File |
|---|---|---|
| 2025 | `nyagami/ea-sports-fc-25-database-ratings-and-stats` | `male_players.csv` |
| 2026 | `justdhia/ea-sports-fc-26-player-ratings` | `ea_fc26_players.csv` |

The 2025 holdout — the **82.2%** headline — is built entirely from the FC 25
source, and the 2026 bracket backtest uses FC 25 ratings too
(`README.md:346-347`). Someone following the README's Phase-2 instructions gets
neither. `report/report.md:134-135` has the FC 25 row; the README never
acquired it.

**Fix:** add both rows to the README table, note that the original author stops
at EA FC 24, and mark the FC 26 entry as wired but unexercised — `data.years`
stops at 2025, and `src/data/adapters.py:74-77` says that file's column names
"have not been checked".

**Related:** `README.md:133-136` says the project maps `edition = year - 2000`
"matching the paper's 2015–2023 scope". That mapping only describes 2015–2024;
2025+ comes from different authors on different schemas.

## [H] 2. The FC-era data caveats appear nowhere outside a module docstring

`src/data/adapters.py:11-21` records two consequences of the FIFA→FC source
switch:

1. Neither FC-era source carries `nation_position`, so **every 2025+ squad is a
   top-N proxy**, never a real call-up list.
2. FC-era files populate `goalkeeping_*` only for actual keepers, where the
   FIFA-era files gave every outfielder a low value — so a 2025 profile's
   goalkeeping means are, in the docstring's words, "**NOT comparable**" to how
   earlier years were built.

Both bear directly on the headline, since the holdout is 2025-only.
`report/report.md:177-182` states the first plainly. `README.md` (the holdout
caveats at :198-200 and :303-326) and `docs/LIMITATIONS.md` state neither, and
`docs/LIMITATIONS.md:50-53` attributes proxy squads to "unlicensed federations"
only — which understates it for precisely the year being reported on.

**Fix:** add both to the README's holdout caveat paragraph and to
`LIMITATIONS.md`'s data section. The GK discontinuity is a train/test
distribution shift across the 2024/2025 boundary, not just a data-quality note.

## [H] 3. Three different home-win rates for the same idea, none labeled by population

| Source | Claim |
|---|---|
| `README.md:243-245` | qualifiers: home wins **63.6%**; neutral ground **46.8%** |
| `docs/API.md:81` | "**62.8%** of *non-neutral* matches are home wins" (no neutral figure) |
| `api/main.py:66-68` | "**62.8%** of non-neutral … vs **49.6%** on neutral ground" |

Recomputed from `data/raw/results.csv` over `data.years` (2015–2025), World Cup
finals + qualifiers, draws dropped — the classifier's actual population:

| Population | non-neutral | neutral |
|---|---|---|
| All in-scope competitions | 63.1% | 48.6% |
| Qualifiers only | 63.2% | 46.7% |
| Development years only (2015–2024) | 63.1% | 49.2% |

So each pair is roughly right *for a different population* — the README's is
qualifiers-only, the code comment's is all competitions — and the small residual
gaps are the squad-profile join dropping matches. Neither doc says which
population it means, so a reader comparing the README to API.md sees a flat
contradiction.

**Fix:** state the population inline next to each figure, and quote both sides
of the pair in API.md rather than just the non-neutral one.

## [M] 4. The duplicated seed list is miscounted

`docs/LIMITATIONS.md:94-96` says three copies of `SEEDS`
(`api/main.py`, `src/pipeline.py`, `run_checks.py`); `CODE_REVIEW.md:60` says
the same three. There are **four**:

- `api/main.py:20` — `SEEDS = [0, 1, 17, 42, 123]`
- `src/make_figures.py:15` — `SEEDS = [0, 1, 17, 42, 123]` ← missing from both lists
- `src/pipeline.py:362` — inline literal in `__main__`
- `run_checks.py:144` — inline literal

`docs/DEVELOPMENT.md:67-69` lists all four correctly. Worth noting in whichever
doc survives: only two of the four are a named `SEEDS` constant, so grepping
`SEEDS` finds half of them.

**Fix:** correct `LIMITATIONS.md:94-96` and `LIMITATIONS.md:149-150` to four
locations.

## [M] 5. API.md cross-references a defect that is already fixed

`docs/API.md:139-141` tells the reader the `team_a != team_b` rule is a
`CODE_REVIEW.md` §1 item — but that item describes the check as *missing*, and
`api/main.py:184-185` implements it (`400`, "Pick two different teams."). The
pointer now asserts an open defect that isn't one.

`docs/API.md:34-35` (early requests raise `KeyError` → unhandled 500) is still
accurate — verified, there is no `/health` route and no readiness guard in
`api/main.py` — but it aims at a file now banner-marked historical.

**Fix:** drop the §1 reference on the `team_a`/`team_b` line entirely; repoint
the startup-500 note at `LIMITATIONS.md`. Same for `docs/DEVELOPMENT.md:89` and
`:69`, whose §1/§4 items *are* still open but shouldn't route readers through a
stale file.

## [M] 6. DEVELOPMENT.md's test gaps defer to a list that is half-resolved

`docs/DEVELOPMENT.md:106` sends the reader to `CODE_REVIEW.md` §5 "for the full
list" of test gaps. Two of §5's five items — no tests for
`build_features.py`, none for `clean.filter_national_team_squad` — were since
written: `tests/test_build_features.py` and `tests/test_clean_profiles.py`
exist and are listed in DEVELOPMENT.md's own table three lines above, at
`:99-100`.

**Fix:** point at `LIMITATIONS.md#testing`, which carries the current list.

## [M] 7. There is no dependency documentation, and `requirements.txt` is incomplete

- **No Python version is stated anywhere** — not in `README.md`, not in
  `docs/*.md`, not in `requirements.txt`. The checked-in `venv/` runs 3.13.3,
  and `api/main.py:62` declares `year: int | None`, a PEP 604 annotation
  Pydantic resolves at runtime, so the real floor is 3.10+. `README.md:113-117`
  just says `python -m venv venv`.
- **`scipy` and `joblib` are missing** from `requirements.txt` despite being
  imported directly at `src/evaluation/evaluate.py:12`,
  `src/models/score_model.py:39`, `src/models/train.py:17`, and
  `api/model_cache.py:11`. They resolve today only as transitive scikit-learn
  dependencies.
- **No Node floor is recorded**, though `start.py:115` shells out to `npm` and
  `frontend/package.json` pins Vite 8 / React 19, which have their own minimum.
- Nothing documents the frontend dependency set as dependencies —
  `docs/FRONTEND.md` covers app structure, not the toolchain.

**Fix:** one "Prerequisites / dependencies" section (README setup, or
DEVELOPMENT.md next to the venv instructions) stating the Python floor, the
Node floor Vite 8 requires, and what each dependency group is for; add `scipy`
and `joblib` to `requirements.txt`.

## [M] 8. The module map doesn't mark dead or divergent code

`docs/ARCHITECTURE.md:100` describes `src/models/train.py` purely as
`tune_all_models()`. The file also contains `run_training_pipeline()`, a second
training path that splits, tunes and fits with no scaling and no PCA, and that
nothing calls (`CODE_REVIEW.md` §3). Someone treating the module map as the map
can reasonably call it and get results that don't match the pipeline's. Same
pattern for `drop_redundant_features` (`src/data/clean.py:134`, defined, never
called) and `add_net_score` (`src/data/collect_matches.py`).

**Fix:** mark all three as dead in the module map, or delete the code. Note
`drop_redundant_features` is the paper's non-redundancy step, so deleting it
should come with a line saying PCA stands in for it.

## [M] 9. An undocumented root `package.json` / `node_modules/`, not gitignored

The repo root holds a `package.json` declaring one dependency,
`impeccable ^4.1.0`, plus an installed `node_modules/` (`impeccable`,
`@impeccable`). Nothing in the project imports it, no doc mentions it, and
`README.md`'s structure block doesn't list it — the project's own frontend
lives in `frontend/` with its own `package.json`.

`.gitignore:39` ignores `frontend/node_modules/` only, so the **root**
`node_modules/` tree is not ignored and would be committed.

**Fix (partial):** `.gitignore` now uses a bare `node_modules/`, which covers
both the root and `frontend/`, so neither can be committed by accident.

**Still open:** whether the root `package.json` / `package-lock.json` /
`node_modules/` should exist at all. They were left in place — nothing in the
repo imports `impeccable`, but deleting another person's tooling on suspicion
is not a documentation fix. Either document what it's for, or remove all three.

## [L] 10. The README project-structure tree is stale and lists a file twice

`README.md:33-66`. `config/config.yaml` appears **twice** — once inside the
`config/` node at :35-36 and again as the tree's last entry at :65.

Missing from the tree, though all exist and are documented in
ARCHITECTURE.md: `src/config.py`, `src/pipeline.py`, `src/score_pipeline.py`,
`src/holdout.py`, `src/backtest_wc2026.py`, `src/make_figures.py`,
`src/data/adapters.py`, `src/features/history_features.py`,
`src/models/score_model.py`; and, at the top level, `start.py`, `stop.py`,
`run_checks.py`, `docs/`, `report/`, `CODE_REVIEW.md`, `CONTEXT/`.

**Fix:** regenerate the tree, or cut it down to the top-level directories and
defer to `ARCHITECTURE.md`'s module map, which is accurate — every function and
class it names was verified to exist.

## [L] 11. CODE_REVIEW.md's line references no longer resolve

The file is pinned to `1aa0ad6`. Its §1 cites `src/pipeline.py:146` for the
hardcoded `random_forest`/`xgboost` feature-importance step; that code is now at
`src/pipeline.py:314`. Its `[src/pipeline.py:146](src/pipeline.py:146)` link
form also doesn't resolve as a link on GitHub.

**Fix:** it's already banner-marked historical — either strip the line numbers
or annotate them as commit-pinned, so they aren't followed.

## [L] 12. `config.yaml` still carries "tune later" TODOs the docs say aren't there

`docs/DEVELOPMENT.md:73-75` says the config file is "heavily commented in place
with the *measured* effect of each knob". Two knobs still read as open work:
`config/config.yaml:93` (`wwr_m: 10 # … tune empirically in Phase 5`) and
`:145` (`high_scoring_goal_threshold` … "tune during EDA").

**Fix:** record the measured justification or drop the TODO phrasing —
otherwise these read as unfinished work in a finished project.

## [L] 13. `report.md` says "Two public datasets" above a three-row table

`report/report.md:130-136`. The table lists two player sources and one match
source.

## [L] 14. `frontend/README.md` is still the stock Vite template

`docs/FRONTEND.md:8-9` says it "Replaces the generic Vite template
`frontend/README.md`", but that file still exists verbatim ("This template
provides a minimal setup to get React working in Vite…"). A reader who opens
`frontend/` first gets boilerplate with no pointer onward.

**Fix:** replace its contents with a one-line link to `docs/FRONTEND.md`.

---

## Suggested order

1. **§1, §2, §3** — these are the ones that change what a reader believes about
   the headline result or stop them reproducing it.
2. **§7, §9** — dependency/prerequisite gaps; both block a clean setup on a new
   machine, and §9 is a commit-hygiene risk.
3. **§4, §5, §6, §11** — cross-reference rot, best done in one pass now that
   `CODE_REVIEW.md` is marked historical.
4. **§8, §10, §12, §13, §14** — accuracy cleanup, batchable.
