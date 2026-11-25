#!/usr/bin/env python3
"""Refresh cached 247 and Rivals Industry basketball recruiting rankings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
from html.parser import HTMLParser
import json
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


OUTPUT_COLUMNS = [
    "source",
    "class_year",
    "rank",
    "player",
    "position",
    "rating",
    "stars",
    "high_school",
    "commit_team",
    "profile_url",
    "source_url",
    "updated_at",
]

AUDIT_COLUMNS = [
    "class_year",
    "source",
    "rank",
    "player",
    "issue",
    "nearest_other_source_player",
    "nearest_other_source_rank",
    "source_url",
]


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tokens: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._href = ""

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self._href = dict(attrs).get("href", "")

    def handle_endtag(self, tag):
        if tag == "a":
            self._href = ""

    def handle_data(self, data):
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        self.tokens.append(text)
        if self._href:
            self.links.append((text, self._href))


def fetch_html(url: str) -> str:
    req = Request(url, headers={
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
        )
    })
    with urlopen(req, timeout=45) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_tokens(url: str) -> tuple[list[str], list[tuple[str, str]]]:
    html = fetch_html(url)
    parser = TextExtractor()
    parser.feed(html)
    return parser.tokens, parser.links


def absolute_url(url: str, base: str) -> str:
    if not url:
        return ""
    if url.startswith("http"):
        return url
    if url.startswith("/"):
        root = re.match(r"https?://[^/]+", base)
        return (root.group(0) if root else "") + url
    return url


def stars_from_rating_text(text: str) -> int | None:
    stars = text.count("★")
    return stars or None


def clean_rating(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return float(match.group(1))


def match_key(value: str) -> str:
    value = "" if pd.isna(value) else str(value)
    value = value.lower().strip()
    value = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def next_data(html: str) -> dict:
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
    if not match:
        return {}
    return json.loads(match.group(1))


def parse_on3_rivals(year: int, max_rank: int) -> list[dict]:
    base_url = f"https://www.on3.com/rivals/rankings/industry-player/basketball/{year}/"
    rows: list[dict] = []

    first = next_data(fetch_html(base_url))
    player_data = first.get("props", {}).get("pageProps", {}).get("playerData", {})
    page_count = int(player_data.get("pagination", {}).get("pageCount", 1) or 1)

    def add_page(page_data: dict, source_url: str, page: int) -> None:
        players = page_data.get("list", [])
        if not players or page_data.get("pagination", {}).get("currentPage") != page:
            raise ValueError(f"Missing or repeated Rivals ranking page {page}")
        for item in players:
            person = item.get("person", {})
            rating = person.get("rating", {})
            rank = item.get("consensusOverallRank") or rating.get("consensusNationalRank")
            if rank is None:
                continue
            rank = int(rank)
            if rank > max_rank:
                continue
            status = person.get("status") or {}
            commit_team = ""
            asset = status.get("committedOrganizationAsset") or {}
            if status.get("committedOrganizationSlug"):
                commit_team = str(status["committedOrganizationSlug"]).replace("-", " ").title()
            high_school = person.get("highSchoolName") or ""
            if person.get("homeTownName"):
                high_school = f"{high_school} ({person['homeTownName']})".strip()
            slug = person.get("slug", "")
            profile_url = f"https://www.on3.com/rivals/{slug}/" if slug else ""
            rows.append({
                "source": "rivals_industry",
                "class_year": year,
                "rank": rank,
                "player": person.get("name", ""),
                "position": item.get("positionAbbreviation") or person.get("positionAbbreviation", ""),
                "rating": rating.get("consensusRating", ""),
                "stars": rating.get("consensusStars", ""),
                "high_school": high_school,
                "commit_team": commit_team or asset.get("title", ""),
                "profile_url": profile_url,
                "source_url": source_url,
            })

    add_page(player_data, base_url, 1)
    for page in range(2, page_count + 1):
        url = f"{base_url}?page={page}"
        add_page(
            next_data(fetch_html(url)).get("props", {}).get("pageProps", {}).get("playerData", {}),
            url,
            page,
        )

    rows = sorted(rows, key=lambda x: (x["class_year"], x["rank"], x["player"]))
    return rows


def parse_247(year: int, max_rank: int) -> list[dict]:
    base_url = f"https://247sports.com/season/{year}-basketball/compositerecruitrankings/?InstitutionGroup=HighSchool"
    rows: list[dict] = []
    seen_profiles: set[str] = set()

    def page_url(page: int) -> str:
        if page == 1:
            return base_url
        return (
            f"https://247sports.com/season/{year}-basketball/compositerecruitrankings/"
            f"?ViewPath=~%2FViews%2FSkyNet%2FPlayerSportRanking%2F_SimpleSetForSeason.ascx&Page={page}"
        )

    expected = 0
    for page in range(1, 30):
        url = page_url(page)
        tokens, links = fetch_tokens(url)
        if page == 1:
            heading = re.search(rf"\b{year} Top Basketball Recruits\s*\(([\d,]+)\)", " ".join(tokens))
            if heading is None:
                raise ValueError("247Sports did not return a rankings page")
            expected = min(max_rank, int(heading[1].replace(",", "")))
        page_players = 0
        for player, href in links:
            href = html.unescape(href)
            raw_href = href.lower()
            if not (raw_href.startswith("/player/") or raw_href.startswith("https://247sports.com/player/")):
                continue
            href = absolute_url(href, url)
            if href in seen_profiles:
                continue
            if len(rows) == expected:
                return rows
            rows.append({
                "source": "247",
                "class_year": year,
                "rank": len(rows) + 1,
                "player": player,
                "position": "",
                "rating": "",
                "stars": "",
                "high_school": "",
                "commit_team": "",
                "profile_url": href,
                "source_url": url,
            })
            seen_profiles.add(href)
            page_players += 1
        if len(rows) == expected or page_players == 0:
            break
    if len(rows) != expected:
        raise ValueError(f"247Sports returned {len(rows)} of {expected} ranked players")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[datetime.now(timezone.utc).year])
    parser.add_argument("--output", default="recruiting_rankings_cache.csv")
    parser.add_argument("--audit-output", default="recruiting_rankings_audit.csv")
    parser.add_argument("--max-247-rank", type=int, default=500)
    parser.add_argument("--max-rivals-rank", type=int, default=300)
    args = parser.parse_args()

    output = Path(args.output)
    df = pd.read_csv(output, keep_default_na=False) if output.exists() else pd.DataFrame(columns=OUTPUT_COLUMNS)
    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    refreshed = []
    sources = [
        ("247", parse_247, args.max_247_rank),
        ("rivals_industry", parse_on3_rivals, args.max_rivals_rank),
    ]
    for year in dict.fromkeys(args.years):
        for source, fetch, max_rank in sources:
            try:
                rows = fetch(year, max_rank)
                if not rows or any(not row["player"].strip() for row in rows):
                    raise ValueError("Source returned no valid ranking list")
            except (OSError, ValueError) as exc:
                print(f"Warning: {source} class {year}: {exc}. Kept any cached rankings.", file=sys.stderr)
                continue
            previous = df["class_year"].eq(year) & df["source"].eq(source)
            df = df.loc[~previous]
            refreshed.append(pd.DataFrame(rows).assign(updated_at=updated_at))
            print(f"Refreshed {len(rows)} {source} rankings for the class of {year}")
    if not refreshed:
        raise SystemExit("No recruiting rankings were refreshed")

    df = pd.concat([df, *refreshed], ignore_index=True)[OUTPUT_COLUMNS]
    df = df.drop_duplicates(["source", "class_year", "rank", "player"])
    df = df.sort_values(["class_year", "source", "rank", "player"], kind="stable")
    audit = build_source_audit(df)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    audit_output = Path(args.audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_output, index=False)
    print(f"Wrote {len(df)} recruiting ranking rows to {output}")
    print(f"Wrote {len(audit)} recruiting source audit rows to {audit_output}")
    if not df.empty:
        print(df.groupby("source").size().to_string())


def build_source_audit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)

    audit_rows: list[dict] = []
    tmp = df.copy()
    tmp["rank"] = pd.to_numeric(tmp["rank"], errors="coerce")
    tmp["_player_key"] = tmp["player"].apply(match_key)

    for year, group in tmp.groupby("class_year"):
        by_source = {source: src_df for source, src_df in group.groupby("source")}
        checks = [
            ("247", "rivals_industry", "247 Composite top 50 missing from Rivals Industry cache"),
            ("rivals_industry", "247", "Rivals Industry top 50 missing from 247 Composite cache"),
        ]
        for source, other_source, issue in checks:
            source_df = by_source.get(source, pd.DataFrame())
            other_base = by_source.get(other_source, pd.DataFrame())
            other_keys = set(other_base["_player_key"]) if not other_base.empty else set()
            top = source_df[source_df["rank"].between(1, 50, inclusive="both")]
            for _, row in top.iterrows():
                row_key = row["_player_key"]
                if row_key in other_keys:
                    continue
                nearest = pd.Series(dtype=object)
                if not other_base.empty:
                    nearest_df = other_base.assign(
                        _name_overlap=other_base["_player_key"].apply(
                            lambda key: len(set(key) & set(row_key))
                        )
                    )
                    nearest = nearest_df.sort_values(["_name_overlap", "rank"], ascending=[False, True]).iloc[0]
                audit_rows.append({
                    "class_year": year,
                    "source": source,
                    "rank": int(row["rank"]),
                    "player": row["player"],
                    "issue": issue,
                    "nearest_other_source_player": "" if nearest.empty else nearest.get("player", ""),
                    "nearest_other_source_rank": "" if nearest.empty else nearest.get("rank", ""),
                    "source_url": row["source_url"],
                })

    return pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS).sort_values(
        ["class_year", "source", "rank"]
    ).reset_index(drop=True)


if __name__ == "__main__":
    main()
