from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core_v3_reconstruction import (
    CORE_V3_FEATURES,
    load_core_v3_artifacts,
    resolve_core_v3_model_dir,
    score_core_v3_memberships,
)


def summarize_alignment(predicted: pd.DataFrame, reference: pd.DataFrame) -> dict[str, float]:
    pred_weights = predicted[[f"A{i}" for i in range(8)]].copy()
    ref_weights = reference[[f"A{i}" for i in range(8)]].copy()

    abs_errors = (pred_weights - ref_weights).abs()
    pred_primary = pred_weights.idxmax(axis=1)
    ref_primary = ref_weights.idxmax(axis=1)
    pred_top2 = pred_weights.apply(lambda row: tuple(row.sort_values(ascending=False).index[:2]), axis=1)
    ref_top2 = ref_weights.apply(lambda row: tuple(row.sort_values(ascending=False).index[:2]), axis=1)

    pred_secondary = pred_top2.map(lambda pair: pair[1])
    ref_secondary = ref_top2.map(lambda pair: pair[1])

    return {
        "row_count": float(len(reference)),
        "mean_abs_error": float(abs_errors.to_numpy().mean()),
        "max_abs_error": float(abs_errors.to_numpy().max()),
        "primary_match_rate": float((pred_primary == ref_primary).mean()),
        "secondary_match_rate": float((pred_secondary == ref_secondary).mean()),
        "top2_set_match_rate": float(
            pd.Series(
                [
                    set(pred_pair) == set(ref_pair)
                    for pred_pair, ref_pair in zip(pred_top2, ref_top2)
                ]
            ).mean()
        ),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = resolve_core_v3_model_dir(repo_root)
    if model_dir is None:
        raise SystemExit("Could not resolve core_v3 handoff model directory.")

    artifacts = load_core_v3_artifacts(model_dir)
    training_path = model_dir / "trank_core_v3_feature_dataset_2025_2026_min150.csv"
    reference_path = model_dir / "memberships.csv"

    training_df = pd.read_csv(training_path)
    reference_df = pd.read_csv(reference_path)

    predicted = score_core_v3_memberships(
        training_df[CORE_V3_FEATURES],
        artifacts,
        membership_steps=60,
        learning_rate=0.12,
        entropy_regularization=0.0,
    )
    predicted.index = (
        training_df["playerId"].astype(str)
        + "|"
        + training_df["teamName"].astype(str)
        + "|"
        + training_df["season"].astype(str)
    )
    reference_df.index = (
        reference_df["playerId"].astype(str)
        + "|"
        + reference_df["teamName"].astype(str)
        + "|"
        + reference_df["season"].astype(str)
    )

    shared_index = predicted.index.intersection(reference_df.index)
    summary = summarize_alignment(
        predicted.loc[shared_index],
        reference_df.loc[shared_index],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
