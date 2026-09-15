#!/usr/bin/env python3
"""Offline tests for the D-II scraping and rate-stat pipeline.

Runs without network access: parsing is exercised against fixture HTML and the
rate math against hand-checkable totals. Run with:

    python tests/test_d2_scrape.py
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from scrape_d2_seasons import (  # noqa: E402
    NUMERIC_COLUMNS, classify_row, parse_page, season_list, season_url,
    split_made_att, to_number,
)
from build_d2_rate_stats import build  # noqa: E402
from backfill_transfers import label, norm_player, norm_team  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "sidearm_stats.html"


class TestCellParsing(unittest.TestCase):
    def test_split_made_att(self):
        self.assertEqual(split_made_att("145-322"), ("145", "322"))
        self.assertEqual(split_made_att("12 / 34"), ("12", "34"))
        self.assertEqual(split_made_att(".450"), ("", ""))
        self.assertEqual(split_made_att(""), ("", ""))

    def test_to_number(self):
        self.assertEqual(to_number("1,004"), "1004")
        self.assertEqual(to_number("32:30"), "32.5")   # mm:ss minutes
        self.assertEqual(to_number("-"), "")
        self.assertEqual(to_number(".450"), "")        # leading-dot pct, not a count
        self.assertEqual(to_number("29"), "29")

    def test_classify_row(self):
        self.assertEqual(classify_row("Total"), "team")
        self.assertEqual(classify_row("Opponents"), "opponent")
        self.assertEqual(classify_row("Jaylen Alexander"), "player")


class TestSeasonUrls(unittest.TestCase):
    def test_season_list_range(self):
        self.assertEqual(season_list("2021-22:2023-24"),
                         ["2021-22", "2022-23", "2023-24"])

    def test_season_list_explicit(self):
        self.assertEqual(season_list("2021-22,2024-25"), ["2021-22", "2024-25"])

    def test_season_url_plain(self):
        self.assertEqual(
            season_url("https://x.com/sports/mens-basketball/stats", "2023-24"),
            "https://x.com/sports/mens-basketball/stats/2023-24")

    def test_season_url_replaces_existing_season(self):
        self.assertEqual(
            season_url("https://x.com/sports/mens-basketball/stats/2025-26?path=mbball", "2022-23"),
            "https://x.com/sports/mens-basketball/stats/2022-23?path=mbball")


class TestTableExtraction(unittest.TestCase):
    def setUp(self):
        self.rows, self.status = parse_page(FIXTURE.read_text())

    def test_picks_stats_table_not_schedule(self):
        self.assertEqual(self.status, "ok")
        names = [r["player"] for r in self.rows]
        self.assertIn("Jaylen Alexander", names)
        self.assertNotIn("Regis", names)          # schedule table ignored

    def test_strips_starter_marker(self):
        self.assertIn("Jaylen Alexander", [r["player"] for r in self.rows])

    def test_keeps_team_and_opponent_totals(self):
        types = {r["row_type"] for r in self.rows}
        self.assertEqual(types, {"player", "team", "opponent"})

    def test_made_attempted_split_into_columns(self):
        row = next(r for r in self.rows if r["player"] == "Jaylen Alexander")
        self.assertEqual(row["fgm"], "145")
        self.assertEqual(row["fga"], "322")
        self.assertEqual(row["3ptm"], "61")

    def test_no_stats_table(self):
        rows, status = parse_page("<html><body><p>No stats posted.</p></body></html>")
        self.assertEqual(rows, [])
        self.assertEqual(status, "no_stats_table_found")


def _fixture_frame() -> pd.DataFrame:
    rows, _ = parse_page(FIXTURE.read_text())
    return pd.DataFrame([{
        "season": "2023-24", "team": "Adams State", "conference": "RMAC",
        "host": "x.com", "row_type": r["row_type"], "jersey": r.get("jersey", ""),
        "player": r["player"], "source_url": "", "source_method": "sidearm",
        "parse_status": "ok",
        **{c: r.get(c, "") for c in NUMERIC_COLUMNS},
    } for r in rows])


class TestRateStats(unittest.TestCase):
    def setUp(self):
        self.out, self.stats = build(_fixture_frame(), min_mpg=10.0, min_gp=5)
        self.guard = self.out[self.out.player == "Jaylen Alexander"].iloc[0]
        self.big = self.out[self.out.player == "Marcus Webb"].iloc[0]

    def test_totals_rows_excluded_from_players(self):
        self.assertEqual(len(self.out), 2)
        self.assertEqual(self.stats["with_opponent_totals"], 2)

    def test_tov_pct_matches_hand_calculation(self):
        # TOV% = TO / (FGA + 0.475*FTA + TO) = 68 / (322 + 45.125 + 68)
        expected = 100 * 68 / (322 + 0.475 * 95 + 68)
        self.assertAlmostEqual(self.guard.TOV_pct, expected, places=6)

    def test_efg_matches_hand_calculation(self):
        expected = 100 * (145 + 0.5 * 61) / 322
        self.assertAlmostEqual(self.guard.eFG, expected, places=6)

    def test_orb_pct_uses_opponent_defensive_rebounds(self):
        # ORB% = ORB * (TeamMin/5) / (Min * (TeamORB + OppDRB))
        expected = 100 * 21 * (6000 / 5) / (952 * (310 + 701))
        self.assertAlmostEqual(self.guard.ORB_pct, expected, places=6)

    def test_big_outrebounds_guard(self):
        self.assertGreater(self.big.ORB_pct, self.guard.ORB_pct)
        self.assertGreater(self.big.DRB_pct, self.guard.DRB_pct)

    def test_guard_creates_more_than_big(self):
        self.assertGreater(self.guard.AST_pct, self.big.AST_pct)

    def test_scale_conventions_match_d1(self):
        # Proportions 0-1 ...
        for col in ("3P_pct", "FT_pct", "three_share"):
            self.assertLessEqual(self.guard[col], 1.0, msg=col)
        # ... and percentages 0-100.
        for col in ("usg", "eFG", "TS_pct", "ORB_pct", "AST_pct", "TOV_pct"):
            self.assertGreater(self.guard[col], 1.0, msg=col)

    def test_values_in_plausible_ranges(self):
        for col, lo, hi in [("usg", 5, 45), ("eFG", 20, 80), ("TS_pct", 20, 80),
                            ("ORB_pct", 0, 25), ("DRB_pct", 0, 45),
                            ("AST_pct", 0, 60), ("TOV_pct", 0, 45)]:
            for row in (self.guard, self.big):
                self.assertTrue(lo <= row[col] <= hi,
                                f"{col}={row[col]} outside [{lo},{hi}] for {row.player}")

    def test_season_player_id_carries_season(self):
        self.assertTrue(self.guard.season_player_id.startswith("2023-24::"))

    def test_missing_opponent_totals_yields_nan_not_crash(self):
        frame = _fixture_frame()
        frame = frame[frame.row_type != "opponent"]
        out, stats = build(frame, min_mpg=10.0, min_gp=5)
        self.assertEqual(stats["with_opponent_totals"], 0)
        self.assertTrue(math.isnan(out.iloc[0].ORB_pct))
        self.assertFalse(math.isnan(out.iloc[0].eFG))   # no opponent context needed


class TestTransferLabeling(unittest.TestCase):
    def test_norm_team_collapses_saint_and_state(self):
        self.assertEqual(norm_team("Saint Cloud State"), norm_team("St. Cloud St."))
        self.assertEqual(norm_team("Cal State San Bernardino"),
                         norm_team("Cal St. San Bernardino"))

    def test_norm_player_drops_suffix(self):
        self.assertEqual(norm_player("Amondo Miller, Jr."), "amondo miller")
        self.assertEqual(norm_player("Aaron Hall II"), "aaron hall")

    def test_label_assigns_divisions(self):
        df = pd.DataFrame({
            "player": ["A", "B", "C", "D"],
            "from": ["Point Loma", "Duke", "Nowhere Tech", "Point Loma"],
            "to": ["UC Riverside", "Point Loma", "Duke", ""],
        })
        out = label(df, d1={"duke", "uc riverside"}, d2={"point loma"})
        self.assertEqual(list(out.move_type), [
            "D2->D1", "D1->D2", "unknown->D1", "D2->uncommitted"])

    def test_blank_destination_is_uncommitted_not_unknown(self):
        df = pd.DataFrame({"player": ["A"], "from": ["Duke"], "to": [""]})
        out = label(df, d1={"duke"}, d2=set())
        self.assertEqual(out.iloc[0].to_div, "uncommitted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
