#!/usr/bin/env python3
"""Backfill multi-season transfer portal history and label division moves.

scripts/update_transfer_portal.py refreshes only the current cycle, so
transfer_portal_cache.csv holds year=2026 alone. A D-II -> D-I translation
study needs completed moves from earlier cycles, which the same upstream
sources expose per year.

The second job here is labeling. A raw portal row is just two school name
strings; the cohort question ("who went D-II -> D-I?") needs each side
resolved to a division. That is done by matching against two known rosters:
  D-I  <- the team column of the historical player index (2016-2026)
  D-II <- data/d2/d2_schools.csv (plus any teams seen while scraping)
Anything matching neither is labeled 'unknown' rather than guessed, and is
written to the audit file so the name-alias list can be extended.

Usage:
    python scripts/backfill_transfers.py --years 2021:2026
    python scripts/backfill_transfers.py --years 2021:2026 --source api

Output:
    data/d2/transfers_backfill.csv   all rows, with from_div/to_div
    data/d2/transfers_unmatched.csv  school names that resolved to neither
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
D2_SCHOOLS = HERE / "data" / "d2" / "d2_schools.csv"
D1_INDEX = HERE / "historical_comps_output" / "d1_historical_player_index.csv.gz"
D2_PLAYERS = HERE / "data" / "d2" / "d2_player_seasons.csv"
OUT = HERE / "data" / "d2" / "transfers_backfill.csv"
UNMATCHED_OUT = HERE / "data" / "d2" / "transfers_unmatched.csv"

API_URL = "https://api.cbbstat.com/players/transfers"
BARTTORVIK_URL = "https://barttorvik.com/playerstat.php"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
)
AVAILABLE_VALUES = {"", "na", "nan", "none", "uncommitted", "undecided", "tbd"}

OUTPUT_COLUMNS = [
    "id", "player", "player_key", "from", "to", "from_div", "to_div",
    "move_type", "exp", "year", "available", "status", "source", "updated_at",
]

# School-name spellings that differ between the portal feeds and the roster
# files. Extend from transfers_unmatched.csv as new ones appear.
TEAM_ALIASES = {
    "uc san diego": "uc san diego", "ucsd": "uc san diego",
    "st.": "st", "saint": "st", "state": "st",
}


def norm_team(value: object) -> str:
    """Normalize a school name for matching: lowercase, no punctuation,
    'State'/'St.' collapsed, directional words kept (they disambiguate)."""
    text = "" if value is None else str(value)
    text = text.lower().strip()
    text = text.replace("&", " and ")
    # Collapse every spelling of Saint/State to 'st' BEFORE stripping
    # punctuation, so 'Saint Cloud State', 'St. Cloud St.' and 'St Cloud State'
    # all land on the same key.
    text = re.sub(r"\bsaint\b", "st", text)
    text = re.sub(r"\bst\.?\b", "st", text)
    text = re.sub(r"\bstate\b", "st", text)
    text = re.sub(r"\buniv(ersity)?\b", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_player(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.lower().strip()
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", text)
    text = re.sub(r"[^a-z ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_api(year: int, timeout: int = 45) -> pd.DataFrame:
    url = f"{API_URL}?{urlencode({'year': year})}"
    req = Request(url, headers={"User-Agent": "ncaa-player-dashboard/backfill"})
    with urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    df = pd.DataFrame(payload)
    if not df.empty:
        df["source"] = "cbbstat"
        df["year"] = year
    return df


def fetch_barttorvik(year: int, timeout: int = 45) -> pd.DataFrame:
    """BartTorvik embeds the portal as a JS array literal in the page."""
    query = urlencode({
        "link": "y", "xvalue": "trans", "year": year,
        "start": f"{year - 1}1101", "end": f"{year}0501",
    })
    req = Request(f"{BARTTORVIK_URL}?{query}", headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        html = response.read().decode("utf-8", errors="replace")
    match = re.search(r"var\s+transfers\s*=\s*(\[.*?\]);", html, flags=re.S)
    if not match:
        return pd.DataFrame()
    rows = []
    for idx, item in enumerate(json.loads(match.group(1)), start=1):
        rows.append({
            "id": f"barttorvik-{year}-{idx}",
            "player": item[0] if len(item) > 0 else "",
            "from": item[1] if len(item) > 1 else "",
            "to": item[2] if len(item) > 2 and item[2] is not None else "",
            "exp": "", "year": year, "source": "barttorvik",
        })
    return pd.DataFrame(rows)


def load_division_rosters() -> tuple[set[str], set[str]]:
    d1: set[str] = set()
    if D1_INDEX.exists():
        idx = pd.read_csv(D1_INDEX, usecols=["team"])
        d1 = {norm_team(t) for t in idx.team.dropna().unique()}
    d2: set[str] = set()
    if D2_SCHOOLS.exists():
        d2 |= {norm_team(t) for t in pd.read_csv(D2_SCHOOLS).team.dropna().unique()}
    if D2_PLAYERS.exists():
        d2 |= {norm_team(t) for t in pd.read_csv(D2_PLAYERS, usecols=["team"]).team.dropna().unique()}
    # A school in both lists is D-I (the D-II seed can carry reclassified schools).
    d2 -= d1
    return d1, d2


def label(df: pd.DataFrame, d1: set[str], d2: set[str]) -> pd.DataFrame:
    def div(series):
        norm = series.map(norm_team)

        def resolve(team: str) -> str:
            # A blank destination means "in the portal, not yet committed" --
            # that is a known state, not a failed name match, so keep it out of
            # the unmatched audit.
            if not team or team in AVAILABLE_VALUES:
                return "uncommitted"
            if team in d1:
                return "D1"
            if team in d2:
                return "D2"
            return "unknown"

        return norm.map(resolve)

    df = df.copy()
    df["from_div"] = div(df["from"])
    df["to_div"] = div(df["to"])
    df["move_type"] = df.from_div + "->" + df.to_div
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="2021:2026", help="inclusive range 'A:B' or comma list")
    ap.add_argument("--source", choices=["both", "api", "barttorvik"], default="both")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--unmatched-out", default=str(UNMATCHED_OUT))
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args(argv)

    if ":" in args.years:
        lo, hi = (int(v) for v in args.years.split(":", 1))
        years = list(range(lo, hi + 1))
    else:
        years = [int(v) for v in args.years.split(",") if v.strip()]

    frames = []
    for year in years:
        for name, fn in (("api", fetch_api), ("barttorvik", fetch_barttorvik)):
            if args.source not in ("both", name):
                continue
            try:
                got = fn(year)
                if not got.empty:
                    frames.append(got)
                print(f"  {year} {name}: {len(got)} rows", file=sys.stderr)
            except Exception as exc:
                print(f"  {year} {name}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
            time.sleep(args.delay)

    if not frames:
        print("No transfer rows fetched -- check network access to the sources.",
              file=sys.stderr)
        return 1

    df = pd.concat(frames, ignore_index=True)
    for col in ("id", "player", "from", "to", "exp"):
        if col not in df.columns:
            df[col] = ""
    to_text = df["to"].fillna("").astype(str).str.strip().str.lower()
    df["available"] = to_text.isin(AVAILABLE_VALUES)
    df["status"] = df["available"].map({True: "Available transfer", False: "Portal committed"})
    df["player_key"] = df["player"].map(norm_player)
    df["updated_at"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")

    d1, d2 = load_division_rosters()
    print(f"roster sizes -> D1 {len(d1)} teams, D2 {len(d2)} teams", file=sys.stderr)
    df = label(df, d1, d2)

    # cbbstat and barttorvik overlap; keep one row per player-year-origin.
    df = df.drop_duplicates(subset=["year", "player_key", "from"], keep="first")
    df = df[OUTPUT_COLUMNS].sort_values(["year", "move_type", "player"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    unmatched = pd.concat([
        df.loc[df.from_div == "unknown", ["from", "year"]].rename(columns={"from": "school"}),
        df.loc[df.to_div == "unknown", ["to", "year"]].rename(columns={"to": "school"}),
    ]).value_counts().reset_index(name="n")
    unmatched.to_csv(args.unmatched_out, index=False)

    counts = df.move_type.value_counts()
    print(json.dumps({
        "years": years,
        "total_rows": len(df),
        "d2_to_d1": int(counts.get("D2->D1", 0)),
        "d1_to_d2": int(counts.get("D1->D2", 0)),
        "d1_to_d1": int(counts.get("D1->D1", 0)),
        "unknown_school_names": int(len(unmatched)),
        "out": args.out,
        "unmatched_out": args.unmatched_out,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
