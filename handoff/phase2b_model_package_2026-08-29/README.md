# Phase 2B Transfer Model Package

Date: 2026-08-29

This folder is a copy-only handoff package for the Phase 2B transfer-projection work.

Everything here is either:

- a copied dataset/spec/result from the original project outputs, or
- a newly created summary file for handoff/documentation

The original project files were not moved or overwritten.

## What this package contains

### `data/`

Copied modeling datasets and feature specs:

- `phase2b_model_table_all_stats.csv`
- `phase2b_feature_spec_all_stats.csv`
- `phase2b_target_spec.csv`
- `phase2b_model_table_stats_plus_archetypes.csv`
- `phase2b_feature_spec_stats_only.csv`
- `phase2b_feature_spec_stats_plus_archetypes.csv`
- `phase2b_feature_spec_archetypes_only.csv`

### `results/best_by_target/`

One folder per predicted stat. Each folder contains copied artifacts from the winning run for that target:

- `*_overall_results.csv`
- `*_holdout_predictions.csv`
- `*_winner_cohort_breakdown.csv`
- `*_feature_missingness.csv`
- `*_manifest.json`
- `*_progress.json`

These are copied from the original tuning outputs.

### `summaries/`

Summary documents and tables:

- `PHASE2B_BEST_MODEL_SUMMARY_2026-08-28.md`
- `best_models_by_target.csv`

### `package_manifest.json`

A small machine-readable manifest for this handoff package.

## Process summary

The Phase 2B work tried to predict transfer-season player outcomes using the frozen Phase 1 D1-to-D1 transfer table.

The targets predicted were:

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

The main feature-set variants tested were:

- `all_stats`
- `handcrafted`
- `archetypes_only`
- `stats_plus_archetypes`

The main model families tested were:

- `linear_regression`
- `ridge`
- `elastic_net`
- `knn`
- `random_forest`
- `extra_trees`
- `gradient_boosting`
- `hist_gbm`
- `xgboost`
- `lightgbm`

Selection rule:

- best model/run for each target chosen by lowest holdout `MAE`
- `RMSE` used as tiebreaker where needed

## Best models

These were the best overall target-level winners across the tested runs:

| Target | Best feature set | Best model | Holdout R² | Holdout MAE | Holdout RMSE |
| --- | --- | --- | ---: | ---: | ---: |
| `post_bpm` | `stats_plus_archetypes` | `extra_trees` | 0.411122 | 2.664605 | 3.803328 |
| `post_PORPAG` | `all_stats` | `ridge` | 0.319414 | 0.869756 | 1.092154 |
| `post_TS_pct` | `all_stats` | `knn` | 0.085730 | 5.963394 | 9.401541 |
| `post_usg` | `stats_plus_archetypes` | `xgboost` | 0.249027 | 3.446479 | 4.356707 |
| `post_3P_pct` | `stats_plus_archetypes` | `gradient_boosting` | 0.263606 | 0.082573 | 0.127721 |
| `post_AST_pct` | `all_stats` | `hist_gbm` | 0.530727 | 3.954084 | 5.382692 |
| `post_TOV_pct` | `all_stats` | `elastic_net` | 0.080539 | 4.183514 | 6.497335 |
| `post_ORB_pct` | `stats_plus_archetypes` | `xgboost` | 0.580419 | 1.733630 | 2.664607 |
| `post_DRB_pct` | `stats_plus_archetypes` | `xgboost` | 0.427469 | 2.847268 | 3.884505 |
| `post_FTR` | `all_stats` | `random_forest` | 0.183459 | 12.543132 | 22.375240 |
| `post_Stl_pct` | `all_stats` | `random_forest` | 0.217038 | 0.610383 | 0.911570 |
| `post_Blk_pct` | `all_stats` | `xgboost` | 0.535600 | 1.033827 | 1.813614 |
| `post_adj_drtg` | `all_stats` | `elastic_net` | 0.464717 | 4.366564 | 5.606046 |

## Where to find each winning model result

For each target, the copied winning run artifacts are in:

- `results/best_by_target/post_bpm/`
- `results/best_by_target/post_PORPAG/`
- `results/best_by_target/post_TS_pct/`
- `results/best_by_target/post_usg/`
- `results/best_by_target/post_3P_pct/`
- `results/best_by_target/post_AST_pct/`
- `results/best_by_target/post_TOV_pct/`
- `results/best_by_target/post_ORB_pct/`
- `results/best_by_target/post_DRB_pct/`
- `results/best_by_target/post_FTR/`
- `results/best_by_target/post_Stl_pct/`
- `results/best_by_target/post_Blk_pct/`
- `results/best_by_target/post_adj_drtg/`

The easiest file to read in each of those folders is:

- `*_overall_results.csv`

That file shows the tested models for that target and the winning model’s metrics.

## Important note about saved model objects

This package contains copied model-result artifacts, not serialized fitted model binaries.

In other words:

- the datasets are included
- the feature specs are included
- the evaluation outputs are included
- the holdout predictions are included
- the fitted `.pkl` / `.joblib` model objects were not saved by the original training pipeline

So when this package says a model like `xgboost` or `gradient_boosting` was the best for a target, that information is recorded in the copied result files, but the exact fitted Python object is not included as a standalone binary artifact.

## Recommended reading order

1. Read `summaries/best_models_by_target.csv`
2. Read `summaries/PHASE2B_BEST_MODEL_SUMMARY_2026-08-28.md`
3. Open the matching folder under `results/best_by_target/` for any target you care about
4. Use the files in `data/` to see the exact dataset/spec for the modeling runs
