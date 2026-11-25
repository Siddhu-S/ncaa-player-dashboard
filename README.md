# ncaa-player-dashboard
Player Dashboard for NCAA D1, D2 &amp; D3 Players

### Player Ratings

The [Ratings folder](ratings/README.md) contains the daily BPM, RAPM, and EPM
pipeline, the player forecast exporter, and the Ratings page.

### Division I data and pages

- `data/raw/division_i/<season>/<retrieval>/` holds the original Torvik player
  CSV, shooting JSON, and retrieval time for each pull.
- `data/current/division-i-<season>.csv` is the validated snapshot consumed by
  Division I and Triton Zone. A failed build leaves the previous file in place.
- `data/historical/` holds the completed-season index and precomputed historical
  comparison tables.
- `pages/division_i/`, `pages/triton_zone/`, and `pages/historical_players/`
  contain their Shiny layouts, styles, and page-specific data or scoring code.
  Shared UI, navigation, and reactive wiring live in `app.py`.

#### Refreshing 2026–27 Division I data

The live dashboard loads `data/current/division-i-2027.csv`. The Daily
Division I preview snapshot workflow runs at 6:30 a.m. Pacific and uploads
raw inputs and the validated CSV as separate artifacts. To publish an update,
replace the tracked CSV with the validated artifact or build it locally from
the repository root. Then commit the CSV to trigger the GitHub Pages rebuild.

```sh
python scripts/build_division_i_snapshot.py --season 2027
shiny run app.py
```

For a reproducible rebuild, pass the two saved raw inputs with `--players`
and `--shooting`. To test a separate file, set `D1_PREVIEW_SNAPSHOT` to its path
before starting Shiny. Preview saves use separate browser storage.
Historical Players includes 2025–26. When a season has no precomputed
Historical Comps tables, current player cards rank archived players using
the active Division I snapshot.
