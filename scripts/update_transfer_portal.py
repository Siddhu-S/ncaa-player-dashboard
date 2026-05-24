#!/usr/bin/env python3
"""Refresh local BartTorvik transfer portal cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

import pandas as pd


API_URL = "https://api.cbbstat.com/players/transfers"
BARTTORVIK_URL = "https://barttorvik.com/playerstat.php"
AVAILABLE_VALUES = {"", "na", "nan", "none", "uncommitted", "undecided", "tbd"}
OUTPUT_COLUMNS = [
    "id",
    "player",
    "from",
    "to",
    "exp",
    "year",
    "available",
    "status",
    "updated_at",
]
CHANGE_LOG_COLUMNS = [
    "detected_at",
    "change_type",
    "player",
    "from",
    "previous_to",
    "current_to",
    "year",
    "previous_status",
    "current_status",
]
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
)


def fetch_transfers(year: int | None = None) -> pd.DataFrame:
    query = urlencode({"year": year}) if year else ""
    url = f"{API_URL}?{query}" if query else API_URL
    req = Request(url, headers={"User-Agent": "ncaa-player-dashboard/1.0"})
    with urlopen(req, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return pd.DataFrame(payload)


def fetch_barttorvik_html(year: int, start: str, end: str) -> str:
    query = urlencode({
        "link": "y",
        "xvalue": "trans",
        "year": year,
        "start": start,
        "end": end,
    })
    url = f"{BARTTORVIK_URL}?{query}"
    opener = build_opener(HTTPCookieProcessor())
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with opener.open(req, timeout=45) as response:
        html = response.read().decode("utf-8", errors="replace")

    if "js_test_submitted" in html and "Verifying Browser" in html:
        verify = Request(
            url,
            data=urlencode({"js_test_submitted": "1"}).encode("utf-8"),
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with opener.open(verify, timeout=45) as response:
            html = response.read().decode("utf-8", errors="replace")

    return html


def fetch_barttorvik_transfers(year: int, start: str, end: str) -> pd.DataFrame:
    html = fetch_barttorvik_html(year, start, end)
    match = re.search(r"var\s+transfers\s*=\s*(\[.*?\]);", html, flags=re.S)
    if not match:
        raise RuntimeError("Could not find BartTorvik transfer data in page HTML")

    payload = json.loads(match.group(1))
    rows = []
    for idx, item in enumerate(payload, start=1):
        player = item[0] if len(item) > 0 else ""
        from_team = item[1] if len(item) > 1 else ""
        to_team = item[2] if len(item) > 2 and item[2] is not None else ""
        rows.append({
            "id": f"barttorvik-{year}-{idx}",
            "player": player,
            "from": from_team,
            "to": to_team,
            "exp": "",
            "year": year,
        })
    return pd.DataFrame(rows)


def normalize_cache(df: pd.DataFrame) -> pd.DataFrame:
    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    to_text = df["to"].fillna("").astype(str).str.strip().str.lower()
    df["available"] = to_text.isin(AVAILABLE_VALUES)
    df["status"] = df["available"].map({True: "Available transfer", False: "Portal committed"})
    df["updated_at"] = updated_at
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df[OUTPUT_COLUMNS].sort_values(["available", "year", "player"], ascending=[False, False, True])


def match_key(value: str) -> str:
    value = "" if pd.isna(value) else str(value)
    value = value.lower().strip()
    value = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def add_change_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_change_key"] = (
        out["year"].fillna("").astype(str)
        + "|"
        + out["player"].apply(match_key)
        + "|"
        + out["from"].apply(match_key)
    )
    return out


def append_availability_changes(previous: pd.DataFrame, current: pd.DataFrame, log_path: str) -> int:
    if previous.empty:
        return 0

    detected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    previous = add_change_key(previous)
    current = add_change_key(current)
    current_by_key = current.drop_duplicates("_change_key").set_index("_change_key")
    previous_by_key = previous.drop_duplicates("_change_key").set_index("_change_key")
    changes: list[dict] = []

    for key, prev in previous_by_key.iterrows():
        was_available = bool(prev.get("available", False))
        if not was_available:
            continue
        if key not in current_by_key.index:
            changes.append({
                "detected_at": detected_at,
                "change_type": "removed_from_portal_cache",
                "player": prev.get("player", ""),
                "from": prev.get("from", ""),
                "previous_to": prev.get("to", ""),
                "current_to": "",
                "year": prev.get("year", ""),
                "previous_status": prev.get("status", ""),
                "current_status": "Missing from latest cache",
            })
            continue
        cur = current_by_key.loc[key]
        is_available = bool(cur.get("available", False))
        if not is_available:
            changes.append({
                "detected_at": detected_at,
                "change_type": "no_longer_available",
                "player": cur.get("player", prev.get("player", "")),
                "from": cur.get("from", prev.get("from", "")),
                "previous_to": prev.get("to", ""),
                "current_to": cur.get("to", ""),
                "year": cur.get("year", prev.get("year", "")),
                "previous_status": prev.get("status", ""),
                "current_status": cur.get("status", ""),
            })

    for key, cur in current_by_key.iterrows():
        is_available = bool(cur.get("available", False))
        if not is_available:
            continue
        if key not in previous_by_key.index or not bool(previous_by_key.loc[key].get("available", False)):
            prev = previous_by_key.loc[key] if key in previous_by_key.index else pd.Series(dtype=object)
            changes.append({
                "detected_at": detected_at,
                "change_type": "newly_available",
                "player": cur.get("player", ""),
                "from": cur.get("from", ""),
                "previous_to": "" if prev.empty else prev.get("to", ""),
                "current_to": cur.get("to", ""),
                "year": cur.get("year", ""),
                "previous_status": "" if prev.empty else prev.get("status", ""),
                "current_status": cur.get("status", ""),
            })

    if not changes:
        return 0

    path = Path(log_path)
    if path.exists():
        log = pd.read_csv(path)
    else:
        log = pd.DataFrame(columns=CHANGE_LOG_COLUMNS)
    log = pd.concat([log, pd.DataFrame(changes)], ignore_index=True)
    log = log[CHANGE_LOG_COLUMNS].drop_duplicates(
        ["change_type", "player", "from", "year", "current_to"],
        keep="last",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    return len(changes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["barttorvik", "cbbstat"], default="barttorvik")
    parser.add_argument("--year", type=int, default=2026, help="Portal year to fetch.")
    parser.add_argument("--start", default="20251101", help="BartTorvik start date as YYYYMMDD.")
    parser.add_argument("--end", default="20260501", help="BartTorvik end date as YYYYMMDD.")
    parser.add_argument("--output", default="transfer_portal_cache.csv")
    parser.add_argument("--change-log", default="transfer_portal_change_log.csv")
    args = parser.parse_args()

    if args.source == "barttorvik":
        raw = fetch_barttorvik_transfers(args.year, args.start, args.end)
    else:
        raw = fetch_transfers(args.year)
    df = normalize_cache(raw)
    output = Path(args.output)
    previous = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=OUTPUT_COLUMNS)
    changes = append_availability_changes(previous, df, args.change_log)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} transfers to {output}")
    print(f"Available: {int(df['available'].sum())}")
    print(f"Logged availability changes: {changes}")


if __name__ == "__main__":
    main()
