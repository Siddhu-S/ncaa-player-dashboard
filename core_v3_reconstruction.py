from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CORE_V3_FEATURES = [
    "ht_inches",
    "usg_%",
    "TS_%",
    "ORB_%",
    "DRB_%",
    "AST_%",
    "FT_%",
    "FT_rate",
    "TP_%",
    "ast_tov",
    "TOV%",
    "rim_pct_of_total_attempts",
    "rim_%",
    "mid_pct_of_total_attempts",
    "mid_%",
    "threePA_per_100",
    "pct_rim_made_assisted",
    "pct_three_made_assisted",
    "pct_total_made_assisted",
    "blk_%",
    "stl_%",
    "pf_per_40",
    "stops_per_40",
]


@dataclass(frozen=True)
class CoreV3Artifacts:
    scaler_mean: pd.Series
    scaler_std: pd.Series
    archetype_profiles: pd.DataFrame


def resolve_core_v3_model_dir(base_path: Path) -> Path | None:
    relative = (
        Path("..")
        / "NCAA-lineup-analyzer"
        / "handoff"
        / "core_v3_dashboard_transfer_2026-08-18"
        / "model"
    )
    direct = (base_path / relative).resolve()
    if direct.exists():
        return direct
    for parent in [base_path, *base_path.parents]:
        candidate = (
            parent
            / "NCAA-lineup-analyzer"
            / "handoff"
            / "core_v3_dashboard_transfer_2026-08-18"
            / "model"
        )
        if candidate.exists():
            return candidate
    return None


def load_core_v3_artifacts(model_dir: Path) -> CoreV3Artifacts:
    training_path = model_dir / "trank_core_v3_feature_dataset_2025_2026_min150.csv"
    profile_path = model_dir / "archetype_feature_profiles.csv"

    training_df = pd.read_csv(training_path)
    scaler_mean = training_df[CORE_V3_FEATURES].apply(pd.to_numeric, errors="coerce").mean()
    scaler_std = (
        training_df[CORE_V3_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .std(ddof=0)
        .replace(0, 1.0)
    )
    profiles = (
        pd.read_csv(profile_path)
        .sort_values("archetypeIndex")
        .reset_index(drop=True)
    )
    return CoreV3Artifacts(
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        archetype_profiles=profiles[CORE_V3_FEATURES].apply(pd.to_numeric, errors="coerce"),
    )


def build_dashboard_core_v3_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw = raw_df.copy()

    gp = pd.to_numeric(raw.get("GP"), errors="coerce").fillna(0.0)
    mpg = pd.to_numeric(raw.get("mins_per_game"), errors="coerce").fillna(0.0)
    total_minutes = gp * mpg

    rim_made = pd.to_numeric(raw.get("pbp_rim_made"), errors="coerce").fillna(0.0)
    rim_missed = pd.to_numeric(raw.get("pbp_rim_missed"), errors="coerce").fillna(0.0)
    rim_assisted = pd.to_numeric(raw.get("pbp_rim_assisted"), errors="coerce").fillna(0.0)
    mid_made = pd.to_numeric(raw.get("pbp_mid_made"), errors="coerce").fillna(0.0)
    mid_missed = pd.to_numeric(raw.get("pbp_mid_missed"), errors="coerce").fillna(0.0)
    three_made = pd.to_numeric(raw.get("pbp_three_made"), errors="coerce").fillna(0.0)
    three_missed = pd.to_numeric(raw.get("pbp_three_missed"), errors="coerce").fillna(0.0)
    three_assisted = pd.to_numeric(raw.get("pbp_three_assisted"), errors="coerce").fillna(0.0)
    dunk_made = pd.to_numeric(raw.get("pbp_dunk_made"), errors="coerce").fillna(0.0)
    dunk_missed = pd.to_numeric(raw.get("pbp_dunk_missed"), errors="coerce").fillna(0.0)
    dunk_assisted = pd.to_numeric(raw.get("pbp_dunk_assisted"), errors="coerce").fillna(0.0)

    rim_made_total = rim_made + dunk_made
    rim_attempts_total = rim_made + rim_missed + dunk_made + dunk_missed
    rim_assisted_total = rim_assisted + dunk_assisted
    mid_attempts_total = mid_made + mid_missed
    three_attempts_total = three_made + three_missed

    total_made = rim_made_total + mid_made + three_made
    total_attempts = rim_attempts_total + mid_attempts_total + three_attempts_total
    total_assisted_made = rim_assisted_total + pd.to_numeric(
        raw.get("pbp_mid_assisted"), errors="coerce"
    ).fillna(0.0) + three_assisted

    stops = pd.to_numeric(raw.get("stops"), errors="coerce").fillna(0.0)

    features = pd.DataFrame(index=raw.index)
    features["ht_inches"] = pd.to_numeric(raw.get("height_inches"), errors="coerce").fillna(0.0)
    features["usg_%"] = pd.to_numeric(raw.get("usg"), errors="coerce").fillna(0.0)
    features["TS_%"] = pd.to_numeric(raw.get("TS_pct"), errors="coerce").fillna(0.0)
    features["ORB_%"] = pd.to_numeric(raw.get("ORB_pct"), errors="coerce").fillna(0.0)
    features["DRB_%"] = pd.to_numeric(raw.get("DRB_pct"), errors="coerce").fillna(0.0)
    features["AST_%"] = pd.to_numeric(raw.get("AST_pct"), errors="coerce").fillna(0.0)
    features["FT_%"] = pd.to_numeric(raw.get("FT_pct"), errors="coerce").fillna(0.0) * 100.0
    features["FT_rate"] = pd.to_numeric(raw.get("FTR"), errors="coerce").fillna(0.0)
    features["TP_%"] = pd.to_numeric(raw.get("3P_pct"), errors="coerce").fillna(0.0) * 100.0
    features["ast_tov"] = pd.to_numeric(raw.get("AST_TOV"), errors="coerce").fillna(0.0)
    features["TOV%"] = pd.to_numeric(raw.get("TOV_pct"), errors="coerce").fillna(0.0)
    features["rim_pct_of_total_attempts"] = (
        pd.to_numeric(raw.get("rim_share"), errors="coerce").fillna(0.0) * 100.0
    )
    features["rim_%"] = pd.to_numeric(raw.get("rim_pct"), errors="coerce").fillna(0.0) * 100.0
    features["mid_pct_of_total_attempts"] = (
        pd.to_numeric(raw.get("mid_share"), errors="coerce").fillna(0.0) * 100.0
    )
    features["mid_%"] = pd.to_numeric(raw.get("mid_pct"), errors="coerce").fillna(0.0) * 100.0
    features["threePA_per_100"] = pd.to_numeric(
        raw.get("3P_per_100_team_pos"), errors="coerce"
    ).fillna(0.0)
    features["pct_rim_made_assisted"] = (
        rim_assisted_total / rim_made_total.replace(0.0, np.nan)
    ).fillna(0.0) * 100.0
    features["pct_three_made_assisted"] = (
        three_assisted / three_made.replace(0.0, np.nan)
    ).fillna(0.0) * 100.0
    features["pct_total_made_assisted"] = (
        total_assisted_made / total_made.replace(0.0, np.nan)
    ).fillna(0.0) * 100.0
    features["blk_%"] = pd.to_numeric(raw.get("Blk_pct"), errors="coerce").fillna(0.0)
    features["stl_%"] = pd.to_numeric(raw.get("Stl_pct"), errors="coerce").fillna(0.0)
    features["pf_per_40"] = pd.to_numeric(
        raw.get("personal_fouls_per_40"), errors="coerce"
    ).fillna(0.0)
    features["stops_per_40"] = (
        stops * 40.0 / total_minutes.replace(0.0, np.nan)
    ).fillna(0.0)

    # Keep this around for convenience when auditing source completeness.
    features["_total_known_fga"] = total_attempts
    return features


def scale_core_v3_features(feature_df: pd.DataFrame, artifacts: CoreV3Artifacts) -> pd.DataFrame:
    values = feature_df[CORE_V3_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return (values - artifacts.scaler_mean) / artifacts.scaler_std


def project_to_simplex(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values) - 1.0
    indexes = np.arange(1, len(values) + 1)
    condition = sorted_values - (cumulative / indexes) > 0
    rho = indexes[condition][-1]
    theta = cumulative[condition][-1] / rho
    return np.maximum(values - theta, 0.0)


def solve_core_v3_membership(
    player_vector: np.ndarray,
    archetype_profiles: np.ndarray,
    membership_steps: int = 60,
    learning_rate: float = 0.12,
    entropy_regularization: float = 0.0,
) -> np.ndarray:
    weights = np.full(archetype_profiles.shape[0], 1.0 / archetype_profiles.shape[0], dtype=float)
    for _ in range(membership_steps):
        reconstruction = weights @ archetype_profiles
        residual = reconstruction - player_vector
        gradient = 2.0 * (archetype_profiles @ residual)
        if entropy_regularization > 0:
            gradient = gradient + entropy_regularization * (np.log(np.maximum(weights, 1e-12)) + 1.0)
        weights = project_to_simplex(weights - (learning_rate * gradient))
    return weights


def score_core_v3_memberships(
    feature_df: pd.DataFrame,
    artifacts: CoreV3Artifacts,
    membership_steps: int = 60,
    learning_rate: float = 0.12,
    entropy_regularization: float = 0.0,
) -> pd.DataFrame:
    scaled = scale_core_v3_features(feature_df, artifacts)
    archetypes = artifacts.archetype_profiles.to_numpy(dtype=float)
    memberships = [
        solve_core_v3_membership(
            row.to_numpy(dtype=float),
            archetypes,
            membership_steps=membership_steps,
            learning_rate=learning_rate,
            entropy_regularization=entropy_regularization,
        )
        for _, row in scaled.iterrows()
    ]
    out = pd.DataFrame(memberships, columns=[f"A{i}" for i in range(archetypes.shape[0])], index=feature_df.index)
    out["weight_sum"] = out.sum(axis=1)
    return out


def attach_primary_secondary_labels(membership_df: pd.DataFrame, label_map: dict[str, str]) -> pd.DataFrame:
    weights = membership_df[[f"A{i}" for i in range(8)]].copy()
    primary_code = weights.idxmax(axis=1)
    primary_weight = weights.max(axis=1)

    secondary_code = []
    secondary_weight = []
    for _, row in weights.iterrows():
        ordered = row.sort_values(ascending=False)
        code = ordered.index[1] if len(ordered.index) > 1 else ordered.index[0]
        secondary_code.append(code)
        secondary_weight.append(float(ordered.loc[code]))

    labeled = membership_df.copy()
    labeled["primaryArchetype"] = primary_code
    labeled["primaryArchetypeLabel"] = primary_code.map(label_map)
    labeled["primaryArchetypeWeight"] = primary_weight
    labeled["secondaryArchetype"] = secondary_code
    labeled["secondaryArchetypeLabel"] = pd.Series(secondary_code, index=labeled.index).map(label_map)
    labeled["secondaryArchetypeWeight"] = secondary_weight
    return labeled
