#!/usr/bin/env python3
"""Scrape multi-season Division II men's basketball player stats.

The dashboard's existing D-II file is a single 2025-26 snapshot with no season
column, which makes cross-season work (transfers, translation models)
impossible. This script walks the same school athletics sites season by season
so the D-II side gains the time dimension the D-I historical index already has.

Most D-II athletics sites run Sidearm Sports, which exposes prior seasons at a
predictable path:

    https://<host>/sports/mens-basketball/stats/2023-24

so a season backfill is the same fetch with a different suffix. PrestoSports
and one-off layouts are handled by the same generic table scorer rather than
per-host special cases.

Team and Opponent total rows are deliberately KEPT (see TOTALS_LABELS). They
are what make true rate stats computable -- opponent defensive rebounds are
required for ORB%, possessions for TOV%/usage -- so build_d2_rate_stats.py
depends on them being carried through.

Usage:
    python scripts/scrape_d2_seasons.py --seasons 2021-22:2025-26
    python scripts/scrape_d2_seasons.py --seasons 2023-24 --limit 5 --verbose

Output:
    data/d2/raw/<season>/<host>.html   cached page source (re-runs are free)
    data/d2/d2_players_raw.csv         one row per player-season
    data/d2/scrape_manifest.csv        per school-season fetch/parse outcome
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent.parent
SCHOOLS_PATH = HERE / "data" / "d2" / "d2_schools.csv"
RAW_DIR = HERE / "data" / "d2" / "raw"
PLAYERS_OUT = HERE / "data" / "d2" / "d2_players_raw.csv"
MANIFEST_OUT = HERE / "data" / "d2" / "scrape_manifest.csv"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
)

# Rows that are team aggregates rather than players. Kept, not dropped --
# build_d2_rate_stats.py needs them to compute possessions and rebound rates.
TOTALS_LABELS = {
    "total", "totals", "team", "team totals", "ucsd totals",
    "opponent", "opponents", "opp", "opponent totals", "opponents totals",
}
OPPONENT_LABELS = {"opponent", "opponents", "opp", "opponent totals", "opponents totals"}

# Header aliases -> canonical name. Sidearm, Presto and legacy layouts all
# spell these differently; everything downstream reads the canonical name.
HEADER_ALIASES = {
    "#": "jersey", "no": "jersey", "no.": "jersey", "num": "jersey",
    "player": "player", "name": "player", "athlete": "player",
    "gp": "gp", "g": "gp", "games": "gp", "gms": "gp",
    "gs": "gs", "starts": "gs",
    "min": "min", "mins": "min", "minutes": "min", "tot min": "min",
    "fg": "fg_made_att", "fgm-a": "fg_made_att", "fgm-fga": "fg_made_att",
    "total fg": "fg_made_att", "fg m-a": "fg_made_att",
    "fgm": "fgm", "fga": "fga",
    "fg%": "fg_pct", "fg pct": "fg_pct",
    "3pt": "three_made_att", "3fg": "three_made_att", "3ptm-a": "three_made_att",
    "3-pt": "three_made_att", "3pt m-a": "three_made_att", "3fg m-a": "three_made_att",
    "3ptm": "3ptm", "3pm": "3ptm", "3pta": "3pta", "3pa": "3pta",
    "3pt%": "three_pct", "3fg%": "three_pct", "3p%": "three_pct",
    "ft": "ft_made_att", "ftm-a": "ft_made_att", "ft m-a": "ft_made_att",
    "ftm": "ftm", "fta": "fta",
    "ft%": "ft_pct",
    "off": "orb", "oreb": "orb", "or": "orb", "off reb": "orb",
    "def": "drb", "dreb": "drb", "dr": "drb", "def reb": "drb",
    "tot": "trb", "reb": "trb", "treb": "trb", "total reb": "trb",
    "pf": "pf", "fouls": "pf",
    "a": "ast", "ast": "ast", "assists": "ast",
    "to": "tov", "tov": "tov", "turnovers": "tov",
    "blk": "blk", "b": "blk", "blocks": "blk",
    "stl": "stl", "st": "stl", "s": "stl", "steals": "stl",
    "pts": "pts", "points": "pts",
    "dq": "dq", "rpg": "rpg", "apg": "apg", "ppg": "ppg",
}

# Columns that prove a table is an individual box score rather than a schedule,
# roster or navigation table. Used to score candidate tables.
SIGNAL_COLUMNS = {"gp", "min", "fg_made_att", "fgm", "pts", "trb", "ast", "tov", "orb", "drb"}

NUMERIC_COLUMNS = [
    "gp", "gs", "min", "fgm", "fga", "3ptm", "3pta", "ftm", "fta",
    "orb", "drb", "trb", "pf", "ast", "tov", "blk", "stl", "pts",
]

OUTPUT_COLUMNS = [
    "season", "team", "conference", "host", "row_type", "jersey", "player",
    *NUMERIC_COLUMNS,
    "source_url", "source_method", "parse_status",
]


# ─────────────────────────────────────────────────────────────────────────
# HTML table extraction (stdlib only, matching scripts/update_recruiting_rankings.py)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Table:
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)


class TableParser(HTMLParser):
    """Collects every <table> on the page as headers + rows of plain text.

    Deliberately layout-agnostic: Sidearm, Presto and hand-rolled pages all
    differ in class names and nesting, but all of them put the box score in a
    real <table>, so scoring candidate tables by column names (pick_stats_table)
    generalizes where per-host selectors do not.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[Table] = []
        self._depth = 0
        self._cur: Table | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._cell_is_header = False
        self._row_has_header = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self._cur = Table()
        elif self._cur is not None:
            if tag == "tr":
                self._row, self._row_has_header = [], False
            elif tag in ("td", "th"):
                self._cell, self._cell_is_header = [], (tag == "th")
                if tag == "th":
                    self._row_has_header = True
            elif tag == "br" and self._cell is not None:
                self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag == "table":
            if self._depth == 1 and self._cur is not None:
                self.tables.append(self._cur)
                self._cur = None
            self._depth = max(0, self._depth - 1)
        elif self._cur is not None:
            if tag in ("td", "th") and self._cell is not None and self._row is not None:
                self._row.append(clean_cell("".join(self._cell)))
                self._cell = None
            elif tag == "tr" and self._row is not None:
                if any(v for v in self._row):
                    # First header-bearing row becomes the header; later ones
                    # are repeated headers mid-table and are skipped.
                    if self._row_has_header and not self._cur.headers:
                        self._cur.headers = self._row
                    else:
                        self._cur.rows.append(self._row)
                self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def clean_cell(text: str) -> str:
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_header(value: str) -> str:
    key = clean_cell(value).lower().strip().rstrip(":")
    key = re.sub(r"\s+", " ", key)
    return HEADER_ALIASES.get(key, key)


def score_table(table: Table) -> int:
    """How strongly this table looks like an individual box score."""
    if not table.headers or len(table.rows) < 2:
        return 0
    canon = {normalize_header(h) for h in table.headers}
    score = len(canon & SIGNAL_COLUMNS)
    if "player" in canon:
        score += 2
    return score


def pick_stats_table(tables: list[Table]) -> Table | None:
    best, best_score = None, 0
    for table in tables:
        score = score_table(table)
        if score > best_score:
            best, best_score = table, score
    # 4 signal columns is enough to rule out schedules and roster tables while
    # still accepting older layouts that omit OR/DR splits.
    return best if best_score >= 4 else None


def split_made_att(value: str) -> tuple[str, str]:
    """'12-34' -> ('12','34'). Sidearm packs FG/3PT/FT as made-attempted."""
    if not value:
        return "", ""
    match = re.match(r"^\s*(-?\d+)\s*[-/]\s*(-?\d+)\s*$", value)
    return (match.group(1), match.group(2)) if match else ("", "")


def to_number(value: str) -> str:
    """Numeric text -> plain number string, else ''. Handles '1,234' and '32:15'."""
    if value is None:
        return ""
    text = clean_cell(str(value)).replace(",", "")
    if not text or text in {"-", "--", "/", "n/a"}:
        return ""
    if re.match(r"^\d+:\d{2}$", text):          # mm:ss minutes
        mins, secs = text.split(":")
        return str(round(int(mins) + int(secs) / 60, 2))
    match = re.match(r"^(-?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else ""


def classify_row(player: str) -> str:
    key = clean_cell(player).lower().strip(" .*")
    if key in OPPONENT_LABELS:
        return "opponent"
    if key in TOTALS_LABELS:
        return "team"
    return "player"


def parse_stats_table(table: Table) -> list[dict]:
    canon = [normalize_header(h) for h in table.headers]
    out: list[dict] = []
    for raw in table.rows:
        if len(raw) < len(canon):
            raw = raw + [""] * (len(canon) - len(raw))
        rec: dict[str, str] = {}
        for key, value in zip(canon, raw):
            if key in ("fg_made_att", "three_made_att", "ft_made_att"):
                made, att = split_made_att(value)
                prefix = {"fg_made_att": ("fgm", "fga"),
                          "three_made_att": ("3ptm", "3pta"),
                          "ft_made_att": ("ftm", "fta")}[key]
                rec[prefix[0]], rec[prefix[1]] = made, att
            else:
                rec[key] = value
        player = clean_cell(rec.get("player", ""))
        if not player:
            continue
        row_type = classify_row(player)
        # Sidearm marks the active/total row with trailing symbols.
        rec["player"] = re.sub(r"[*†#]+$", "", player).strip()
        rec["row_type"] = row_type
        for col in NUMERIC_COLUMNS:
            rec[col] = to_number(rec.get(col, ""))
        # Derive totals when only splits are present, and vice versa.
        if not rec.get("trb") and rec.get("orb") and rec.get("drb"):
            rec["trb"] = str(float(rec["orb"]) + float(rec["drb"]))
        out.append(rec)
    return out


def parse_page(html: str) -> tuple[list[dict], str]:
    parser = TableParser()
    try:
        parser.feed(html)
    except Exception as exc:                       # malformed markup, keep going
        return [], f"parse_error:{type(exc).__name__}"
    table = pick_stats_table(parser.tables)
    if table is None:
        return [], "no_stats_table_found"
    rows = parse_stats_table(table)
    if not rows:
        return [], "table_found_no_rows"
    if not any(r["row_type"] == "player" for r in rows):
        return rows, "totals_only"
    return rows, "ok"


# ─────────────────────────────────────────────────────────────────────────
# Fetching
# ─────────────────────────────────────────────────────────────────────────

def season_list(spec: str) -> list[str]:
    """'2021-22:2025-26' -> every season in that inclusive range."""
    if ":" in spec:
        start, end = spec.split(":", 1)
        first, last = int(start[:4]), int(end[:4])
        return [f"{y}-{str(y + 1)[-2:]}" for y in range(first, last + 1)]
    return [s.strip() for s in spec.split(",") if s.strip()]


def season_url(base: str, season: str) -> str:
    base = base.rstrip("/")
    if "?" in base:                                 # e.g. /stats/2025-26?path=mbball
        head, query = base.split("?", 1)
        head = re.sub(r"/\d{4}-\d{2}$", "", head.rstrip("/"))
        return f"{head}/{season}?{query}"
    return f"{base}/{season}"


def fetch(url: str, timeout: int, retries: int = 3) -> tuple[str | None, str]:
    last = "unknown_error"
    for attempt in range(retries):
        try:
            req = Request(url, headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            })
            with urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace"), "ok"
        except HTTPError as exc:
            last = f"http_{exc.code}"
            if exc.code in (404, 410):             # season genuinely absent
                return None, last
        except (URLError, TimeoutError) as exc:
            last = f"neterr_{type(exc).__name__}"
        except Exception as exc:
            last = f"error_{type(exc).__name__}"
        if attempt < retries - 1:
            time.sleep(2 ** attempt + random.random())
    return None, last


def cache_path(season: str, host: str, root: Path | None = None) -> Path:
    safe = re.sub(r"[^a-z0-9.-]+", "_", host.lower()) or "unknown"
    return (root or RAW_DIR) / season / f"{safe}.html"


def load_schools(path: Path, limit: int | None) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("stats_base_url") or "").strip()]
    return rows[:limit] if limit else rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default="2021-22:2025-26",
                    help="range 'A:B' or comma list (default: 2021-22:2025-26)")
    ap.add_argument("--schools", default=str(SCHOOLS_PATH))
    ap.add_argument("--players-out", default=str(PLAYERS_OUT))
    ap.add_argument("--manifest-out", default=str(MANIFEST_OUT))
    ap.add_argument("--cache-dir", default=str(RAW_DIR))
    ap.add_argument("--delay", type=float, default=1.5,
                    help="seconds between requests to the same host (default 1.5)")
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--limit", type=int, help="only first N schools (smoke test)")
    ap.add_argument("--refetch", action="store_true", help="ignore cached HTML")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    seasons = season_list(args.seasons)
    schools = load_schools(Path(args.schools), args.limit)
    if not schools:
        print(f"No schools with a stats_base_url in {args.schools}", file=sys.stderr)
        return 1

    cache_root = Path(args.cache_dir)
    print(f"{len(schools)} schools x {len(seasons)} seasons = "
          f"{len(schools) * len(seasons)} pages", file=sys.stderr)

    players: list[dict] = []
    manifest: list[dict] = []

    for season in seasons:
        for i, school in enumerate(schools, 1):
            host = school.get("host") or ""
            url = season_url(school["stats_base_url"], season)
            path = cache_path(season, host, cache_root)
            html, status = None, "cached"

            if path.exists() and not args.refetch:
                html = path.read_text(encoding="utf-8", errors="replace")
            else:
                html, status = fetch(url, args.timeout)
                if html is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(html, encoding="utf-8")
                time.sleep(args.delay + random.random() * 0.5)

            if html is None:
                manifest.append({"season": season, "team": school["team"], "host": host,
                                 "url": url, "fetch_status": status,
                                 "parse_status": "not_fetched", "n_players": 0})
                if args.verbose:
                    print(f"  [{season}] {school['team']}: FETCH {status}", file=sys.stderr)
                continue

            rows, parse_status = parse_page(html)
            n_players = sum(1 for r in rows if r["row_type"] == "player")
            for rec in rows:
                players.append({
                    "season": season,
                    "team": school["team"],
                    "conference": school.get("conference", ""),
                    "host": host,
                    "row_type": rec["row_type"],
                    "jersey": rec.get("jersey", ""),
                    "player": rec.get("player", ""),
                    **{c: rec.get(c, "") for c in NUMERIC_COLUMNS},
                    "source_url": url,
                    "source_method": school.get("platform", ""),
                    "parse_status": parse_status,
                })
            manifest.append({"season": season, "team": school["team"], "host": host,
                             "url": url, "fetch_status": status,
                             "parse_status": parse_status, "n_players": n_players})
            if args.verbose:
                print(f"  [{season}] {i}/{len(schools)} {school['team']}: "
                      f"{parse_status} ({n_players} players)", file=sys.stderr)

    for out_path, rows, cols in (
        (Path(args.players_out), players, OUTPUT_COLUMNS),
        (Path(args.manifest_out), manifest,
         ["season", "team", "host", "url", "fetch_status", "parse_status", "n_players"]),
    ):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)

    ok = sum(1 for m in manifest if m["parse_status"] == "ok")
    print(json.dumps({
        "pages_attempted": len(manifest),
        "pages_parsed_ok": ok,
        "player_rows": sum(1 for p in players if p["row_type"] == "player"),
        "team_total_rows": sum(1 for p in players if p["row_type"] == "team"),
        "opponent_total_rows": sum(1 for p in players if p["row_type"] == "opponent"),
        "players_out": str(args.players_out),
        "manifest_out": str(args.manifest_out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
