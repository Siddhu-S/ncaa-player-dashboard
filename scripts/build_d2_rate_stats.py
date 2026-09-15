#!/usr/bin/env python3
"""Derive true rate stats for D-II players from scraped box scores.

This is the script that makes D-II commensurable with D-I. The dashboard's
existing D-II file carries only counting stats and per-40 proxies, so the
archetype features that matter (usage, AST%, ORB%, DRB%, TOV%, STL%, BLK%)
could not be computed and the two divisions could not share a style space.

Those rates need team and OPPONENT context -- ORB% needs opponent defensive
rebounds, TOV%/usage need possessions -- which is exactly why
scrape_d2_seasons.py keeps the Total and Opponents rows of each box score.
Given those, the standard Oliver possession estimates apply unchanged.

Scale conventions deliberately MATCH the D-I files, which are internally
inconsistent (verified against d1_historical_player_index.csv.gz):
    percentages (0-100): usg, ORB_pct, DRB_pct, AST_pct, TOV_pct,
                         Stl_pct, Blk_pct, eFG, TS_pct, FTR
    proportions (0-1)  : 3P_pct, FT_pct, three_share
Matching the quirk is intentional -- a pooled model must not silently mix
a 0-1 column with a 0-100 one.

Usage:
    python scripts/build_d2_rate_stats.py
    python scripts/build_d2_rate_stats.py --min-mpg 10 --out data/d2/d2_style.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RAW_IN = HERE / "data" / "d2" / "d2_players_raw.csv"
OUT = HERE / "data" / "d2" / "d2_player_seasons.csv"

COUNTS = ["gp", "gs", "min", "fgm", "fga", "3ptm", "3pta", "ftm", "fta",
          "orb", "drb", "trb", "pf", "ast", "tov", "blk", "stl", "pts"]

# Free throw -> possession weight. 0.475 is the standard Oliver coefficient
# and is what Barttorvik's D-I numbers use, so keep it identical here.
FT_POSS_WEIGHT = 0.475


def possessions(fga, fta, tov, orb) -> pd.Series:
    return fga - orb + tov + FT_POSS_WEIGHT * fta


def safe_div(num, den):
    """Element-wise divide, NaN where the denominator is zero or missing."""
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return np.where((den > 0) & den.notna(), num / den.replace(0, np.nan), np.nan)


def build(raw: pd.DataFrame, min_mpg: float, min_gp: int) -> tuple[pd.DataFrame, dict]:
    for col in COUNTS:
        raw[col] = pd.to_numeric(raw.get(col), errors="coerce")

    players = raw[raw.row_type == "player"].copy()
    teams = raw[raw.row_type == "team"].copy()
    opps = raw[raw.row_type == "opponent"].copy()

    key = ["season", "team"]
    tm = teams.groupby(key, as_index=False)[COUNTS].first().add_prefix("tm_")
    tm = tm.rename(columns={"tm_season": "season", "tm_team": "team"})
    op = opps.groupby(key, as_index=False)[COUNTS].first().add_prefix("op_")
    op = op.rename(columns={"op_season": "season", "op_team": "team"})

    df = players.merge(tm, on=key, how="left").merge(op, on=key, how="left")

    # Team minutes: prefer the totals row, fall back to summed player minutes,
    # then to 200 per game (5 players x 40 min, ignoring overtime).
    summed = players.groupby(key, as_index=False)["min"].sum().rename(columns={"min": "sum_min"})
    df = df.merge(summed, on=key, how="left")
    team_min = df["tm_min"].where(df["tm_min"] > 0)
    team_min = team_min.fillna(df["sum_min"]).fillna(df["tm_gp"] * 200)
    df["team_min"] = team_min
    # Oliver's formulas are expressed per "team minutes / 5" (i.e. game minutes).
    df["tm_min5"] = df["team_min"] / 5.0

    tm_poss = possessions(df.tm_fga, df.tm_fta, df.tm_tov, df.tm_orb)
    op_poss = possessions(df.op_fga, df.op_fta, df.op_tov, df.op_orb)
    df["tm_poss"], df["op_poss"] = tm_poss, op_poss

    m = df["min"]
    df["usg"] = 100 * safe_div(
        (df.fga + FT_POSS_WEIGHT * df.fta + df.tov) * df.tm_min5,
        m * (df.tm_fga + FT_POSS_WEIGHT * df.tm_fta + df.tm_tov),
    )
    df["ORB_pct"] = 100 * safe_div(df.orb * df.tm_min5, m * (df.tm_orb + df.op_drb))
    df["DRB_pct"] = 100 * safe_div(df.drb * df.tm_min5, m * (df.tm_drb + df.op_orb))
    df["TRB_pct"] = 100 * safe_div(df.trb * df.tm_min5,
                                   m * (df.tm_trb + df.op_trb))
    # AST%: assists as a share of teammate field goals made while on the floor.
    df["AST_pct"] = 100 * safe_div(
        df.ast, (safe_div(m, df.tm_min5) * df.tm_fgm) - df.fgm)
    df["TOV_pct"] = 100 * safe_div(df.tov, df.fga + FT_POSS_WEIGHT * df.fta + df.tov)
    df["Stl_pct"] = 100 * safe_div(df.stl * df.tm_min5, m * df.op_poss)
    # Opponent two-point attempts, for block rate.
    op_2pa = df.op_fga - df["op_3pta"]
    df["Blk_pct"] = 100 * safe_div(df.blk * df.tm_min5, m * op_2pa)

    # Shooting / style — no team context needed.
    df["eFG"] = 100 * safe_div(df.fgm + 0.5 * df["3ptm"], df.fga)
    df["TS_pct"] = 100 * safe_div(df.pts, 2 * (df.fga + FT_POSS_WEIGHT * df.fta))
    df["FTR"] = 100 * safe_div(df.fta, df.fga)
    df["3P_pct"] = safe_div(df["3ptm"], df["3pta"])
    df["FT_pct"] = safe_div(df.ftm, df.fta)
    df["three_share"] = safe_div(df["3pta"], df.fga)
    df["AST_TOV"] = safe_div(df.ast, df.tov)
    df["mins_per_game"] = safe_div(df["min"], df.gp)
    df["pts_per_game"] = safe_div(df.pts, df.gp)
    df["ast_per_game"] = safe_div(df.ast, df.gp)
    df["treb_per_game"] = safe_div(df.trb, df.gp)
    per40 = 40.0 / df["min"].replace(0, np.nan)
    df["personal_fouls_per_40"] = df.pf * per40
    # Same shape as the D-I index's stops_per_40 (steals + blocks per 40).
    df["stops_per_40"] = (df.stl + df.blk) * per40

    df["has_opponent_totals"] = df.op_fga.notna()
    df["has_team_totals"] = df.tm_fga.notna()

    before = len(df)
    kept = df[(df.mins_per_game >= min_mpg) & (df.gp >= min_gp)].copy()
    kept["player_key"] = (kept.player.astype(str).str.strip().str.lower()
                          .str.replace(r"[^a-z ]", "", regex=True).str.strip())
    kept["season_player_id"] = (kept.season.astype(str) + "::" + kept.player_key
                                + "::" + kept.team.astype(str).str.strip().str.lower())

    stats = {
        "player_season_rows_in": before,
        "rows_after_filter": len(kept),
        "with_team_totals": int(kept.has_team_totals.sum()),
        "with_opponent_totals": int(kept.has_opponent_totals.sum()),
        "rate_stats_computable_pct": round(100 * float(kept.ORB_pct.notna().mean()), 1),
        "seasons": sorted(kept.season.dropna().unique().tolist()),
        "teams": int(kept.team.nunique()),
    }
    return kept, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW_IN))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--min-mpg", type=float, default=10.0,
                    help="minutes-per-game floor (default 10, matches the D-I pool)")
    ap.add_argument("--min-gp", type=int, default=5)
    args = ap.parse_args(argv)

    raw = pd.read_csv(args.raw, low_memory=False)
    out, stats = build(raw, args.min_mpg, args.min_gp)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    stats["out"] = args.out
    print(json.dumps(stats, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
