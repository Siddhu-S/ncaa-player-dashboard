# D-II multi-season data

The dashboard's original D-II file (`current_d2_all_players_10mpg_dashboard_schema.csv`)
is a single 2025-26 snapshot whose `Year` column is *class standing*, not season.
That makes cross-season work impossible. This directory holds the multi-season
rebuild.

## Pipeline

```
d2_schools.csv                 seed: 293 schools, conference, stats base URL
  └─ scrape_d2_seasons.py      fetch + parse box scores, one page per school-season
       ├─ raw/<season>/*.html  cached page source (re-runs are free)
       ├─ d2_players_raw.csv   one row per player-season, PLUS team/opponent totals
       └─ scrape_manifest.csv  per-page fetch/parse outcome -- check this first
            └─ build_d2_rate_stats.py
                 └─ d2_player_seasons.csv   rate stats, D-I-commensurable
```

Transfers are separate and do not depend on the scrape:

```
backfill_transfers.py
  ├─ transfers_backfill.csv    all portal rows 2021-2026, labeled from_div/to_div
  └─ transfers_unmatched.csv   school names resolving to neither division
```

## Why team and opponent totals are kept

`ORB%`, `DRB%`, `TOV%`, `usage`, `STL%` and `BLK%` all need team possession
context, and the rebound rates need *opponent* rebounds specifically. College
box-score pages publish Total and Opponents rows, so the scraper keeps them
(`row_type` is `player` / `team` / `opponent`). Without them the D-II side has
only counting stats and per-40 proxies, which cannot share a style space with
the D-I rate stats.

## Scale conventions

Matched to the D-I files, which are internally inconsistent. Do not "fix" these
without changing both sides:

| convention | columns |
| --- | --- |
| percentage 0-100 | `usg`, `ORB_pct`, `DRB_pct`, `AST_pct`, `TOV_pct`, `Stl_pct`, `Blk_pct`, `eFG`, `TS_pct`, `FTR` |
| proportion 0-1 | `3P_pct`, `FT_pct`, `three_share` |

## Known gaps

- **Seed list is incomplete.** `transfers_unmatched.csv` currently surfaces ~74
  D-II schools referenced by the portal that are missing from `d2_schools.csv`
  (N.M. Highlands, UT Tyler, Daemen, Virginia Union, USC Aiken, ...). Add them
  before treating the D-II universe as complete.
- **No play-by-play.** Shot-location fields the D-I side carries (`rim_share`,
  `mid_share`, `dunk_share`, `assisted_fg_pct`) have no D-II equivalent. A
  shared style space can only use features present on both sides.
- **No height/class in the scrape.** Those live on roster pages, not the stats
  page, and would need a second pass.
- **Name-only player identity.** There is no stable cross-season ID, so
  `season_player_id` is `season::name::team` and cross-season joins match on
  normalized name. Expect some collision noise.

## Running it

```bash
python scripts/scrape_d2_seasons.py --seasons 2021-22:2025-26 --verbose
python scripts/build_d2_rate_stats.py
python scripts/backfill_transfers.py --years 2021:2026
python tests/test_d2_scrape.py        # offline, no network needed
```

Start with `--limit 5` to confirm the layouts still parse before a full run.
`raw/` is cached, so an interrupted run resumes for free; pass `--refetch` to
force fresh pages.
