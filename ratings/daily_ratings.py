"""Ingest game data, recheck gaps, and publish BPM/RAPM from stored inputs."""
import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from box_plus_minus import BOX_COLUMNS, complete_team_box, season_bpm
from daily_rapm import ESPN, PACIFIC, espn_scoreboards, fetch_json, recover_espn
from player_ratings import (FEEDS, RELEASES, ROOT, download, feed_rows,
                            key, lineup_ratings, ncaa_date, solve_rapm, unique)
from rapm_repairs import expected_score
import ratings_store as store

NCAA = "https://stats.ncaa.org/contests"
VERIFIED_FINALS = {
    "401909694": ({"627521": 69, "627524": 113},
                  "https://seminoles.com/sports/mens-basketball/stats/2026/florida-am/boxscore/16827"),
    "401920378": ({"627235": 88, "627342": 93},
                  "https://hawkeyesports.com/sports/mbball/schedule"),
}
BOX_FIELDS = dict(two_made="fgm", two_attempts="fga", threes="tpm", three_attempts="tpa",
                  free_throws="ftm", fta="fta", points="pts", orb="orb", drb="drb",
                  assists="ast", turnovers="tov", steals="stl", blocks="blk", fouls="pf")


def matchup(played, home, away):
    return played, frozenset((home, away))


def catalog(inputs, scratch, season, cutoff, dates, previous):
    teams = {r["id"]: r["team"] for r in feed_rows(inputs, "team_ids", season)}
    by_name = unique((key(name), tid) for tid, name in teams.items())
    raw = list(feed_rows(inputs, "player_box", season))
    events = unique((r["contest_id"], r["espn_game_id"]) for r in raw if r["espn_game_id"])
    aliases = unique((r[f"{side}_espn_team_id"], r[f"{side}_ncaa_team_id"]) for r in raw
                     for side in ("home", "away") if r[f"{side}_espn_team_id"]
                     and r[f"{side}_ncaa_team_id"] in teams)
    games, sources, evidence = {}, defaultdict(dict), defaultdict(dict)
    known = {matchup(date.fromisoformat(g["game_date"]), g["home_team_id"], g["away_team_id"]): k
             for k, g in previous.items()}
    ids = {}

    def remember(played, home, away, score, provider, provider_id, url, event=None):
        pair = matchup(played, home, away)
        # Provider IDs identify a game even if an overnight date differs.
        prior = previous.get(f"espn:{event}") if event else None
        if prior:
            pair = matchup(date.fromisoformat(prior["game_date"]), home, away)
        elif event and f"espn:{event}" in ids:
            pair = ids[f"espn:{event}"]
        gid = known.setdefault(pair, f"espn:{event}" if event else f"ncaa:{provider_id}")
        games.setdefault(gid, dict(season=season, game_key=gid, game_date=pair[0].isoformat(),
                                   home_team_id=home, away_team_id=away,
                                   home_score=score[0], away_score=score[1]))
        evidence[gid][provider] = dict(zip((home, away), score))
        sources[gid][provider, provider_id] = dict(provider=provider, provider_game_id=provider_id, url=url)
        ids[f"{provider}:{provider_id}"] = pair
        if event:
            ids[f"espn:{event}"] = pair

    for row in feed_rows(inputs, "schedule", season):
        played = ncaa_date(row["game_date"])
        if played not in dates or not all(row[f"{side}_score"].isdigit() for side in ("home", "away")):
            continue
        home, away = (by_name.get(key(row[side])) for side in ("home", "away"))
        if not home or not away or home == away:
            continue
        ncaa = row["contest_id"]
        remember(played, home, away, expected_score(ncaa, row), "ncaa", ncaa,
                 f"{NCAA}/{ncaa}/box_score", events.get(ncaa))

    for board in espn_scoreboards(sorted(dates | {d + timedelta(days=1) for d in dates}), scratch):
        for event in board:
            if not event["status"]["type"]["completed"]:
                continue
            game = event["competitions"][0]
            played = datetime.fromisoformat(event["date"]).astimezone(PACIFIC).date()
            if played > cutoff:
                continue
            if played not in dates and f"espn:{event['id']}" not in ids:
                continue
            sides = {c["homeAway"]: (aliases.get(c["team"]["id"]) or
                                     by_name.get(key(c["team"]["location"])), int(c["score"]))
                     for c in game["competitors"]}
            if set(sides) != {"home", "away"}:
                continue
            (home, hs), (away, aws) = sides["home"], sides["away"]
            if not home or not away or home == away:
                continue
            eid = event["id"]
            remember(played, home, away, (hs, aws), "espn", eid,
                     f"https://www.espn.com/mens-college-basketball/boxscore/_/gameId/{eid}", eid)
    for game_id, game in games.items():
        correction = VERIFIED_FINALS.get(game_id.removeprefix("espn:"))
        if correction and set(correction[0]) == {game["home_team_id"], game["away_team_id"]}:
            scores, url = correction
            for side in ("home", "away"):
                game[f"{side}_score"] = scores[game[f"{side}_team_id"]]
            evidence[game_id]["official_school"] = scores
            sources[game_id]["official_school", game_id] = dict(
                provider="official_school", provider_game_id=game_id, url=url)
    return teams, raw, games, {g: list(v.values()) for g, v in sources.items()}, evidence, known, ids, aliases


def box_row(game, tid, name, player, minutes, stats, neutral, teams):
    home = tid == game["home_team_id"]
    opponent = game["away_team_id"] if home else game["home_team_id"]
    row = [None] * 53
    row[0], row[5], row[6], row[8] = game["game_date"].replace("-", ""), teams[opponent], game["game_key"], minutes
    row[46], row[47], row[48], row[51], row[52] = (
        "N" if neutral else ("H" if home else "A"), teams[tid], name, player, game["season"])
    for stat, column in BOX_COLUMNS.items():
        row[column] = stats[stat]
    return row


def ncaa_box(game, rows, teams, neutral):
    result = []
    for raw in rows:
        team = raw["team_ncaa_team_id"]
        if team not in (game["home_team_id"], game["away_team_id"]):
            raise ValueError("NCAA box has an unresolved team")
        if not raw["player_id"]:
            raise ValueError("NCAA box has an unresolved player")
        stats = {name: float(raw[field]) for name, field in BOX_FIELDS.items()}
        stats["two_made"] -= stats["threes"]
        stats["two_attempts"] -= stats["three_attempts"]
        result.append(box_row(game, team, raw["clean_name"] or raw["player"], f"ncaa:{raw['player_id']}",
                              float(raw["mins"]), stats, neutral, teams))
    return result


def espn_box(game, summary, teams, aliases, player_ids):
    result = []
    neutral = summary["header"]["competitions"][0]["neutralSite"]
    for block in summary["boxscore"]["players"]:
        team = aliases.get(block["team"]["id"])
        if team not in (game["home_team_id"], game["away_team_id"]):
            raise ValueError("ESPN box has an unresolved team")
        for group in block["statistics"]:
            for raw in group["athletes"]:
                if raw.get("didNotPlay") or not raw["stats"]:
                    continue
                data = dict(zip(group["labels"], raw["stats"]))
                name = raw["athlete"]["displayName"]
                player = player_ids.get((team, key(name)))
                if not player:
                    raise ValueError(f"ESPN box player {name} has no unique NCAA roster ID")
                fg, fga = map(int, data["FG"].split("-"))
                threes, tpa = map(int, data["3PT"].split("-"))
                ft, fta = map(int, data["FT"].split("-"))
                stats = dict(two_made=fg-threes, two_attempts=fga-tpa, threes=threes,
                             three_attempts=tpa, free_throws=ft, fta=fta)
                for name_key, label in zip(("points", "orb", "drb", "assists", "turnovers", "steals",
                                            "blocks", "fouls"),
                                           ("PTS", "OREB", "DREB", "AST", "TO", "STL", "BLK", "PF")):
                    stats[name_key] = int(data[label])
                parts = data["MIN"].split(":")
                minutes = float(parts[0]) + (float(parts[1]) / 60 if len(parts) == 2 else 0)
                result.append(box_row(game, team, name, f"ncaa:{player}", minutes, stats, neutral, teams))
    return result


def valid_box(rows, game, teams):
    for side in ("home", "away"):
        box = [r for r in rows if r[47] == teams[game[f"{side}_team_id"]]]
        if not box:
            raise ValueError(f"Missing {side} box score")
        _, stats = complete_team_box(box)
        points = int(sum(stats["points"]))
        expected = game[f"{side}_score"]
        # The official game includes an own basket with no individual scorer.
        if game["game_date"] == "2026-11-02" and box[0][47] == "Colgate" and (points, expected) == (74, 76):
            continue
        if points != expected:
            raise ValueError(f"{side} player points {points} differ from final {expected}")
    return sorted(rows, key=lambda r: (r[47], str(r[51])))


def box_conflicts(first, second):
    a, b = ({(r[47], r[51]): r for r in rows if r[8] > 0} for rows in (first, second))
    conflicts = []
    for player in sorted(a.keys() | b.keys()):
        if player not in a or player not in b:
            row = a.get(player) or b[player]
            if row[8] <= 1 and all(row[c] == 0 for c in BOX_COLUMNS.values()):
                continue
            conflicts.append(f"{row[48]} is missing from one box")
            continue
        differences = [stat for stat, column in BOX_COLUMNS.items() if a[player][column] != b[player][column]]
        if abs(a[player][8] - b[player][8]) > 1:
            differences.append("minutes")
        if differences:
            conflicts.append(f"{a[player][48]} differs in {', '.join(differences)}")
    return conflicts


def decision(rows, evidence, stamp, *, reason=None, excluded=0, review=False):
    if rows:
        status = "accepted_with_exclusions" if excluded or review else "accepted"
        revision = hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()
    else:
        status = "needs_review" if review else "retry"
        revision = None
    return dict(rows=rows, evidence=dict(evidence, reason=reason), status=status, revision=revision,
                excluded=excluded, reason=reason, retry_at=datetime.combine(stamp.astimezone(PACIFIC).date() + timedelta(days=1),
                                             datetime.min.time(), PACIFIC).isoformat()
                if status != "accepted" else None)


def collect(inputs, scratch, season, cutoff, dates, previous, stamp):
    teams, raw, games, sources, scores, known, ids, aliases = catalog(
        inputs, scratch, season, cutoff, dates, previous)
    ncaa_rows = defaultdict(list)
    for row in raw:
        ncaa_rows[row["contest_id"]].append(row)
    venue_by_event = {}
    for day in dates | {d + timedelta(days=1) for d in dates}:
        board = scratch / f"scoreboard-{day}.json"
        if board.exists():
            for event in json.loads(board.read_text())["events"]:
                venue_by_event[event["id"]] = event["competitions"][0].get("neutralSite")

    observations, rejected = {}, {}

    def remember(event, played, home, away, score, source, rows, excluded):
        pair = ids.get(f"espn:{event}") or ids.get(event) or matchup(played, home, away)
        gid = known.get(pair)
        if gid not in games:
            return
        if dict(zip((home, away), score)) != dict(zip(
            (games[gid]["home_team_id"], games[gid]["away_team_id"]),
            (games[gid]["home_score"], games[gid]["away_score"])
        )):
            rejected[gid] = "Possession feed final score differs from catalog"
            return
        observations[gid] = source, rows, excluded

    status = {}
    roster, _ = lineup_ratings(inputs, season, cutoff, status, game_observer=remember, fit=False)
    player_ids = unique(((team, key(name)), pid) for (team, pid), name in roster.items())
    def score_conflict(gid):
        finals = scores[gid]
        return "official_school" not in finals and len({tuple(sorted(s.items())) for s in finals.values()}) > 1

    missing = {}
    for gid, game in games.items():
        if gid not in observations and not score_conflict(gid):
            played = date.fromisoformat(game["game_date"])
            sides = {teams[game[f"{s}_team_id"]]: game[f"{s}_score"] for s in ("home", "away")}
            missing[played, frozenset(sides.items())] = sides
    fallback, gaps = recover_espn(inputs, scratch, season, missing, teams, {})
    if fallback:
        lineup_ratings(inputs, season, cutoff, game_observer=remember, extra_backfill=fallback, fit=False)
    gap_index = {(g["date"], frozenset(g["teams"])): g["reason"] for g in gaps}
    exclusions, historical_sources = {}, {}
    for path in (ROOT / "rapm_backfill_audit_through_2026-11-11.csv", scratch / "fallback-audit.csv"):
        if path.exists():
            with path.open(newline="") as f:
                for row in csv.DictReader(f):
                    event = row["espn_event_id"]
                    if row["excluded_lineup_possessions"]:
                        exclusions[event] = int(row["excluded_lineup_possessions"])
                    historical_sources[event] = {field: row[field] for field in
                        ("source", "play_by_play_url", "source_review", "review_source_url") if row.get(field)}

    def fetch_box(eid):
        path = scratch / f"{eid}.json"
        try:
            return eid, json.loads(path.read_text()) if path.exists() else fetch_json(f"{ESPN}/summary?event={eid}", path), None
        except (OSError, ValueError) as error:
            return eid, None, str(error)

    event_ids = {s["provider_game_id"] for rows in sources.values() for s in rows if s["provider"] == "espn"}
    with ThreadPoolExecutor(max_workers=6) as pool:
        summaries = {eid: (data, error) for eid, data, error in pool.map(fetch_box, sorted(event_ids))}

    for gid, game in games.items():
        urls = list({s["url"] for s in sources[gid]})
        evidence = {"source_urls": sorted(urls), "final_scores": scores[gid]}
        if score_conflict(gid):
            yield game, sources[gid], {metric: decision([], evidence, stamp,
                    reason="NCAA and ESPN final scores disagree", review=True) for metric in ("bpm", "rapm")}
            continue
        source_ids = {s["provider"]: s["provider_game_id"] for s in sources[gid]}
        event = source_ids.get("espn")
        ncaa = source_ids.get("ncaa")
        summary, download_error = summaries.get(event, (None, None))
        candidates, errors = {}, {}
        neutral = venue_by_event.get(event)
        if summary:
            neutral = summary["header"]["competitions"][0].get("neutralSite", neutral)
            try:
                candidates["espn"] = valid_box(espn_box(game, summary, teams, aliases, player_ids), game, teams)
            except (KeyError, TypeError, ValueError) as error:
                errors["espn"] = str(error)
        elif download_error:
            errors["espn"] = download_error
        if ncaa_rows.get(ncaa) and neutral is not None:
            try:
                candidates["ncaa"] = valid_box(ncaa_box(game, ncaa_rows[ncaa], teams, neutral), game, teams)
            except (KeyError, TypeError, ValueError) as error:
                errors["ncaa"] = str(error)
        conflicts = box_conflicts(*candidates.values()) if len(candidates) == 2 else []
        chosen = "ncaa" if "ncaa" in candidates else "espn"
        box_evidence = dict(evidence, box_sources=sorted(candidates), source_errors=errors, conflicts=conflicts)
        if conflicts:
            bpm = decision([], box_evidence, stamp, reason="Player box scores disagree", review=True)
        elif chosen in candidates:
            bpm = decision(candidates[chosen], dict(box_evidence, selected=chosen), stamp)
        else:
            bpm = decision([], box_evidence, stamp, reason="No validated player box score and venue",
                           review=bool(errors) and not download_error)
        if gid in observations:
            source, rows, excluded = observations[gid]
            excluded = excluded or (exclusions.get(event, 0) if source == "backfill" else 0)
            encoded = [[period, clock, lineup, pts, [roster[p] for p in lineup]]
                       for period, clock, lineup, pts in rows]
            rapm = decision(encoded, dict(evidence, selected=source,
                                          reconstruction=historical_sources.get(event, {}),
                                          possession_exclusions=excluded), stamp, excluded=excluded,
                            reason="Some possessions have unresolved lineups or control" if excluded else None)
        else:
            problem = rejected.get(gid) or status.get(ncaa)
            names = frozenset(teams[game[f"{s}_team_id"]] for s in ("home", "away"))
            reason = gap_index.get((game["game_date"], names)) or problem or "No validated possession lineup"
            rapm = decision([], evidence, stamp, reason=reason,
                            review=bool(problem) or "Identity review" in reason or "mismatch" in reason)
        yield game, sources[gid], dict(bpm=bpm, rapm=rapm)


def fit_and_publish(season, cutoff, games, statuses, teams, stamp, *, dry_run=False):
    dates = {g["game_key"]: g["game_date"] for g in games}
    expected = len(games)
    coverage = {m: dict(expected_games=expected, included_games=0, review_or_retry=0,
                        games_with_exclusions=0, excluded_input_rows=0)
                for m in ("bpm", "rapm")}
    manifest, boxes = [], []
    groups = defaultdict(lambda: [0, 0])
    exposures, appearances, latest, roster = Counter(), defaultdict(set), {}, {}
    for status in statuses:
        metric, gid, rows = status["metric"], status["game_key"], status["rows"]
        if status["status"] != "accepted":
            coverage[metric]["review_or_retry"] += 1
        if not rows:
            continue
        coverage[metric]["included_games"] += 1
        excluded = status.get("excluded_rows", status.get("excluded", 0))
        if excluded:
            coverage[metric]["games_with_exclusions"] += 1
            coverage[metric]["excluded_input_rows"] += excluded
        manifest.append([gid, metric, status["revision"]])
        if metric == "bpm":
            boxes.extend(rows)
            continue
        for _, _, lineup, points, names in rows:
            lineup = tuple(tuple(p) for p in lineup)
            groups[lineup][0] += 1
            groups[lineup][1] += points
            for player, name in zip(lineup, names):
                roster[player] = name
                exposures[player] += 1
                appearances[player].add(gid)
                latest[player] = max(latest.get(player, date.min), date.fromisoformat(dates[gid]))
    if not boxes or not groups:
        raise ValueError("Both metrics need accepted game inputs before publishing")
    bpm, bpm_through = season_bpm(boxes, season, cutoff, include_box_id=True)
    rapm = solve_rapm(groups, exposures, appearances, latest, roster, teams)
    team_ids = unique((name, tid) for tid, name in teams.items())
    combined = {(team_ids[row["team"]], row["box_player_id"].removeprefix("ncaa:")): row
                for row in bpm.values()}
    for player, row in rapm.items():
        combined.setdefault(player, {}).update(row)
    players = []
    for (team_id, player_id), row in sorted(combined.items()):
        player_key = f"{team_id}:{player_id}"
        players.append(dict(season=season, through_date=cutoff.isoformat(), player_key=player_key,
                            player_name=row["player_name"], team=row["team"], bpm=row.get("bpm"),
                            rapm=row.get("rapm"), details=row))
    if len({p["player_key"] for p in players}) != len(players):
        raise ValueError("BPM and RAPM player identity collision")
    coverage["bpm"]["through_date"] = bpm_through
    coverage["rapm"]["through_date"] = max(latest.values()).isoformat()
    for counts in coverage.values():
        counts["missing_games"] = counts["expected_games"] - counts["included_games"]
    if not dry_run:
        store.publish(dict(season=season, through_date=cutoff.isoformat(), published_at=stamp,
                           model_version="bpm-college-v1+rapm-ridge-2000-v1",
                           coverage=coverage, input_manifest=manifest), players)
    print(json.dumps(coverage, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int)
    parser.add_argument("--cutoff", type=date.fromisoformat)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not os.environ.get("RATINGS_DATABASE_URL"):
        parser.error("Set the RATINGS_DATABASE_URL Actions secret before publishing")
    cutoff = args.cutoff or (datetime.now(PACIFIC).date() - timedelta(days=1))
    season = args.season or cutoff.year + (cutoff.month >= 7)
    stamp = datetime.now(timezone.utc)
    run_id = os.getenv("GITHUB_RUN_ID", str(uuid4())) + ":" + os.getenv("GITHUB_RUN_ATTEMPT", "1")
    previous, due, last_cutoff = ([], [], None) if args.dry_run else store.load_queue(
        season, cutoff, stamp.isoformat())
    dates = {cutoff} | {date.fromisoformat(day) for day in due}
    if last_cutoff:
        start = date.fromisoformat(last_cutoff)
        dates.update(start + timedelta(days=i) for i in range(1, (cutoff-start).days+1))
    with tempfile.TemporaryDirectory(prefix="daily-ratings-") as folder:
        scratch = Path(folder)
        inputs = args.inputs or scratch
        if args.inputs is None:
            with ThreadPoolExecutor(max_workers=5) as pool:
                jobs = [pool.submit(download, f"{RELEASES}/ncaa_mbb_{name}/ncaa_mbb_{name}_{season}.csv.gz",
                                    inputs / f"{name}.csv.gz") for name in (*FEEDS, "player_box")]
                for job in jobs:
                    job.result()
        if not last_cutoff:
            dates.update(ncaa_date(r["game_date"]) for r in feed_rows(inputs, "schedule", season)
                         if ncaa_date(r["game_date"]) <= cutoff)
            if dates:
                dates.update(date(season-1, 11, 1)+timedelta(days=i)
                             for i in range(max(0, (cutoff-date(season-1, 11, 1)).days+1)))
        count = Counter()
        pending_writes = []
        dry_games, dry_statuses = [], []
        for game, sources, metrics in collect(inputs, scratch, season, cutoff, dates,
                                              {g["game_key"]: g for g in previous}, stamp):
            count.update(f"{metric}:{result['status']}" for metric, result in metrics.items())
            if args.dry_run:
                dry_games.append(game)
                dry_statuses.extend(dict(value, game_key=game["game_key"], metric=metric)
                                    for metric, value in metrics.items())
            else:
                pending_writes.append(store.game_batch(game, sources, metrics, run_id, stamp.isoformat()))
                if len(pending_writes) == 20:
                    store.query("\n".join(pending_writes))
                    pending_writes.clear()
        if pending_writes:
            store.query("\n".join(pending_writes))
        print(dict(count))
        games, statuses = (dry_games, dry_statuses) if args.dry_run else store.load_season(season, cutoff)
        teams = {r["id"]: r["team"] for r in feed_rows(inputs, "team_ids", season)}
        fit_and_publish(season, cutoff, games, statuses, teams, stamp.isoformat(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
