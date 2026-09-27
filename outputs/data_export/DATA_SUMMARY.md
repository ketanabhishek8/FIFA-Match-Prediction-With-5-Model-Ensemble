# Data Summary — FIFA World Cup Match Predictor

Generated 2026-09-07 from `config/config.yaml` (years 2015–2025, qualifiers included).
All counts below are measured from the actual files in `data/raw/`, not estimated.

---

## 1. Sources

Everything comes from **two public Kaggle dataset families**, pulled by
`src/data/collect_players.py` and `src/data/collect_matches.py`. No scraping, no
private data, no manual entry.

### A. Player attributes — EA FIFA / EA Sports FC ratings

| Years | Kaggle dataset | Author | Notes |
|---|---|---|---|
| 2015–2024 | `stefanoleone992/fifa-{15..24}-complete-player-dataset` | Stefano Leone | One dataset per game edition, scraped from sofifa.com. Edition = year − 2000. |
| 2025 | `nyagami/ea-sports-fc-25-database-ratings-and-stats` | nyagami | Different schema; remapped by `src/data/adapters.py`. |

11 files, `data/raw/players_2015.csv` … `players_2025.csv`, **195,123 player rows**,
119 columns in the widest source file.

| Year | Rows | | Year | Rows |
|---|---|---|---|---|
| 2015 | 16,155 | | 2021 | 18,944 |
| 2016 | 15,623 | | 2022 | 19,239 |
| 2017 | 17,596 | | 2023 | 18,533 |
| 2018 | 17,954 | | 2024 | 18,350 |
| 2019 | 18,085 | | 2025 | 16,161 |
| 2020 | 18,483 | | | |

### B. Match history — international results since 1872

`martj42/international-football-results-from-1872-to-2017` (slug is frozen at 2017,
the dataset is maintained well past it). Four files ship in the bundle:

| File | Rows | Used? |
|---|---|---|
| `results.csv` | 49,520 | **Yes** — the match spine |
| `shootouts.csv` | 682 | **Yes** — penalty winners in the 2026 backtest |
| `goalscorers.csv` | 47,903 | No — ships with the bundle, no code reads it |
| `former_names.csv` | 36 | No — same |

`results.csv` covers **1872-11-30 → 2026-07-19**, 201 distinct tournaments.
Top competitions by volume: Friendly (18,387), FIFA World Cup qualification (8,771),
UEFA Euro qualification (2,824), African Cup of Nations qualification (2,327),
**FIFA World Cup (1,068)**.

**Total raw footprint: 124 MB across 15 CSVs.**

---

## 2. How the raw data becomes model input

### Player side — 195,123 rows → 1,749 team-year profiles

1. **Squad filter** (`clean.filter_national_team_squad`) — 195,123 → **24,220 rows**.
   Where EA licensed a federation in that edition, the real call-up list is used via
   the `nation_position` tag (**10,029 rows**). Where it didn't, the top 23 by `overall`
   for that nationality/year stand in (**14,191 rows**, 59%). The fallback is
   load-bearing, not cosmetic: without it, 13 of the 40 teams at the 2018 + 2022 World
   Cups vanish and only 35 of 128 matches survive.
2. **Completeness filter** (`clean.clean_player_data`) — drops any column >20% null.
   119 → **96 columns**. Casualties include `nation_position` itself (its job is done),
   `sofifa_id`, all `*_url` columns, `release_clause_eur`, `goalkeeping_speed`.
3. **Aggregation** (`clean.build_all_team_year_profiles`) — squad mean per numeric
   attribute, one row per (team, year): **1,749 profiles, 189 unique nations,
   37 attributes each** (~160 nations per year).

The 37 attributes are the paper's eight groups: age/height/weight, attacking (5),
skill (5), movement (5), power (5), mentality (6), defending (3), goalkeeping (5).
See `data_dictionary.csv`.

### Match side — 49,520 rows → 1,495 training rows

| Step | Rows | Lost |
|---|---|---|
| `results.csv`, all tournaments 1872–2026 | 49,520 | — |
| World Cup finals + qualifiers, all years | 9,839 | 39,681 |
| ↳ restricted to config years 2015–2025 | 2,730 | 7,109 |
| ↳ both teams have a profile for that year | 1,964 | 766 |
| ↳ draws removed (binary task) | **1,495** | 469 |

Of the 2,730 in-scope matches, 2,602 are qualifiers and 128 are finals — including
qualifiers is what makes the sample viable at all, since 2015–2025 contains only two
tournaments' worth of finals.

**Final training matrix: 1,495 rows × 111 features.**
Label balance: 917 home wins / 578 away wins (61.3% / 38.7%). 179 nations appear.

Each of the 37 profile attributes becomes three columns — `<attr>_a` (home),
`<attr>_b` (away), `<attr>_diff` (home − away) — giving 111. These are then
standard-scaled and PCA-reduced at 95% retained variance inside each seed's fit.

### Rows by year (final matrix)

2015: 90 · 2016: 160 · 2017: 216 · 2018: 51 · 2019: 33 · 2020: 16 ·
2021: 370 · 2022: 118 · 2023: 62 · 2024: 98 · 2025: 281

(2020 is thin for the obvious reason — the pandemic wiped out most qualifying.)

---

## 3. Two datasets serve two different models

- **ML ensemble** — the 1,495 × 111 matrix above, 80/20 split, 5 seeds `[0, 1, 17, 42, 123]`.
- **WWR baseline** — fit on the **full 9,839-match** World Cup history (all years, not just
  2015–2025), so the baseline isn't handicapped by the player-data window. Test rows are
  excluded by exact `(date, home_team, away_team)` key to prevent leakage.

**2026 is deliberately excluded from both.** `results.csv` already contains the 2026 World
Cup; holding it out keeps `src/backtest_wc2026.py` an honest out-of-sample test, with team
strength taken from EA FC 25 ratings published before the tournament kicked off.

---

## 4. Known caveats worth stating in the presentation

1. **59% of squads are proxies, not real call-ups.** EA licenses only a subset of
   federations per edition (50 nations in FIFA 18, 33 in FIFA 22), so most team profiles
   are top-23-by-rating approximations.
2. **2025 is a proxy throughout.** The FC 25 source carries no national-team tags at all,
   so every 2025 squad goes down the fallback path.
3. **2025 goalkeeping means aren't comparable to earlier years.** FIFA-era files gave every
   outfielder a low GK value; the FC 25 source populates GK attributes only for keepers, so
   the squad mean measures something different.
4. **Video-game ratings are a proxy for ability**, subject to EA's own scouting bias and
   popularity effects.
5. **Draws are dropped** (469 matches, 24% of joined rows). The task is strictly home-win
   vs. away-win, per the paper's framing.
6. **Home-win class imbalance is real** (61/39) and partly an artifact — World Cup finals are
   played at neutral venues, but `results.csv` still lists one team as "home".

---

## 5. Files in this export

| File | Rows | What it is |
|---|---|---|
| `team_year_profiles.csv` | 1,749 | The aggregated (team, year) profiles — the paper's core artifact |
| `model_matches.csv` | 1,495 | Exact matches used for training/testing, with labels |
| `worldcup_matches_all_years.csv` | 9,839 | Full WC finals + qualifier history the WWR baseline is fit on |
| `data_dictionary.csv` | 45 | Every source column, its group, and whether it survives into the profile |
| `audit.json` | — | Machine-readable version of every count in this document |
| `DATA_SUMMARY.md` | — | This file |

Raw inputs are not included here (124 MB); re-fetch them with
`python -m src.data.collect_players` and `python -m src.data.collect_matches`
(needs a Kaggle API token at `~/.kaggle/kaggle.json`).
