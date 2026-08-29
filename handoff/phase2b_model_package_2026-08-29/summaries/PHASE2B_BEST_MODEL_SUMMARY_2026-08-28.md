# Phase 2B Best Model Summary

Date: 2026-08-28

This summary shows the transfer-projection targets we tried to predict, the best model we found for each target, and the holdout metrics for that winning run.

Selection rule:

- best run chosen by lowest holdout `MAE`
- `RMSE` used as a tiebreaker when needed

Runs compared:

- `all_stats`
- `handcrafted`
- `archetypes_only`
- `stats_plus_archetypes`

## Predicted stats

- `post_bpm`
- `post_PORPAG`
- `post_TS_pct`
- `post_usg`
- `post_3P_pct`
- `post_AST_pct`
- `post_TOV_pct`
- `post_ORB_pct`
- `post_DRB_pct`
- `post_FTR`
- `post_Stl_pct`
- `post_Blk_pct`
- `post_adj_drtg`

## Best model by target

| Target | Best feature set | Best model | Holdout R² | Holdout MAE | Holdout RMSE | CV MAE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `post_bpm` | `stats_plus_archetypes` | `extra_trees` | 0.411122 | 2.664605 | 3.803328 | 2.565122 |
| `post_PORPAG` | `all_stats` | `ridge` | 0.319414 | 0.869756 | 1.092154 | 0.839704 |
| `post_TS_pct` | `all_stats` | `knn` | 0.085730 | 5.963394 | 9.401541 | 6.086354 |
| `post_usg` | `stats_plus_archetypes` | `xgboost` | 0.249027 | 3.446479 | 4.356707 | 3.362071 |
| `post_3P_pct` | `stats_plus_archetypes` | `gradient_boosting` | 0.263606 | 0.082573 | 0.127721 | 0.087903 |
| `post_AST_pct` | `all_stats` | `hist_gbm` | 0.530727 | 3.954084 | 5.382692 | 3.950237 |
| `post_TOV_pct` | `all_stats` | `elastic_net` | 0.080539 | 4.183514 | 6.497335 | 4.228187 |
| `post_ORB_pct` | `stats_plus_archetypes` | `xgboost` | 0.580419 | 1.733630 | 2.664607 | 1.710505 |
| `post_DRB_pct` | `stats_plus_archetypes` | `xgboost` | 0.427469 | 2.847268 | 3.884505 | 2.789190 |
| `post_FTR` | `all_stats` | `random_forest` | 0.183459 | 12.543132 | 22.375240 | 12.085420 |
| `post_Stl_pct` | `all_stats` | `random_forest` | 0.217038 | 0.610383 | 0.911570 | 0.582161 |
| `post_Blk_pct` | `all_stats` | `xgboost` | 0.535600 | 1.033827 | 1.813614 | 0.914212 |
| `post_adj_drtg` | `all_stats` | `elastic_net` | 0.464717 | 4.366564 | 5.606046 | 4.145347 |

## Quick takeaways

- `all_stats` won 8 of the 13 targets.
- `stats_plus_archetypes` won 5 of the 13 targets.
- `archetypes_only` did not produce the best model for any target.
- The targets where `stats_plus_archetypes` won were:
  - `post_bpm`
  - `post_usg`
  - `post_3P_pct`
  - `post_ORB_pct`
  - `post_DRB_pct`

## Current default recommendation

- Use `all_stats` as the default base setup.
- Use `stats_plus_archetypes` selectively for the five targets where it produced the best holdout MAE.
- Do not use `archetypes_only` as the primary modeling setup.
