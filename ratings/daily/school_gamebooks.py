"""Read and validate official school game books for the daily Ratings job."""
import csv
from calendar import month_abbr
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from html import unescape
from html.parser import HTMLParser
import re
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

from player_ratings import ROOT, key, unique
from rebuild_espn_rapm import (espn_game, replay, short_stint_minutes, sidearm_game,
                              sidearm_minutes, sidearm_payload)

STAT_FIELDS = {
    "two_made": "totalFieldGoalsMade", "two_attempts": "totalFieldGoalsAttempted",
    "threes": "threePointFieldGoalsMade", "three_attempts": "threePointFieldGoalsAttempted",
    "free_throws": "freeThrowsMade", "fta": "freeThrowsAttempted", "points": "totalPointsScored",
    "orb": "offensiveRebounds", "drb": "defensiveRebounds", "assists": "assists",
    "turnovers": "turnovers", "steals": "steals", "blocks": "blockedShots", "fouls": "personalFouls",
}

# The official log reconstructs 21:54. Torvik reports 21.9 and ESPN rounds to
# 22. The school's own printed box lists 23 for this player only.
SCHOOL_REPLAY_MINUTES = {
    ("401924721", "627312", "11483213"): (23, 21.9, (
        "https://barttorvik.com/2027_all_advgames.json.gz",
        "https://www.espn.com/mens-college-basketball/boxscore/_/gameId/401924721",
    )),
}


# Dayton's final report credits Haynes with a turnover between the two
# logged swaps. The earlier swap also reproduces the official box minutes.
# https://daytonflyers.com/documents/2026/11/22/2924476.PDF
SCHOOL_DUPLICATE_SWAPS = {
    "401920736": (2, 395, 327, {("out", "RANDALL,GRANT"), ("in", "HAYNES,JALEN")}),
}


def text(value):
    return " ".join(unescape(re.sub(r"<[^>]*>", " ", value)).split())


def table_rows(table):
    return [[text(cell) for cell in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row, re.S)]
            for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, re.S)]


def legacy_gamebook(html, neutral):
    """Normalize the classic SIDEARM box and play tables to the structured format."""
    title = re.search(r"<title>\s*Men(?:'|&#39;|&apos;)s Basketball .*? on (\d+/\d+/\d{4})", html, re.S)
    if not title or neutral is None:
        raise ValueError("No supported school box score with a confirmed venue")
    tables = re.findall(r"<table\b[^>]*>.*?</table>", html, re.S)
    boxes = [table for table in tables if "sortable-table" in table.split(">", 1)[0]]
    if len(boxes) != 2:
        raise ValueError("School game book does not contain exactly two player boxes")
    book = {"itsWomens": False, "isNeutral": neutral, "periodsWithPlays": [], "legacy": True,
            "gameDate": datetime.strptime(title[1], "%m/%d/%Y").strftime("%m/%d/%y")}
    for side, table in zip(("visiting", "home"), boxes, strict=True):
        caption = text(re.search(r"<caption\b[^>]*>(.*?)</caption>", table, re.S)[1])
        name, score = caption.rsplit(" ", 1)
        if not score.isdigit():
            raise ValueError("School player box has no final score")
        block = {"id": side, "scoreSummary": {"score": score}, "players": []}
        book.update({f"{side}Team": block, f"{side}TeamId": side,
                     f"{side}TeamName": name, f"{side}TeamScore": score})
        for cells in table_rows(table):
            if len(cells) != 15 or not cells[0].isdigit() or cells[3] in {"-", "--"}:
                continue
            jersey, player, starter, mins, fg, threes, ft, reb, _, pf, ast, tov, blk, stl, pts = cells
            player = re.sub(r"^\d+\s+", "", player).upper().replace(", ", ",")
            fgm, fga = fg.split("-")
            tpm, tpa = threes.split("-")
            ftm, fta = ft.split("-")
            orb, drb = reb.split("-")
            stats = dict(zip(STAT_FIELDS.values(), (fgm, fga, tpm, tpa, ftm, fta, pts,
                                                    orb, drb, ast, tov, stl, blk, pf), strict=True))
            minutes, _, seconds = mins.rstrip("+").partition(":")
            stats.update(minutesPlayed=minutes, secondsPlayed=seconds or "0")
            block["players"].append(dict(checkName=player, uniform=jersey, gamePlayed=1,
                                         gameStarted=int(starter == "*"), stats=stats))
    for period, table in enumerate((t for t in tables if 'class="sidearm-table play-by-play"' in t), 1):
        clock = "20:00" if period <= 2 else "5:00"
        plays = []
        for cells in table_rows(table):
            if len(cells) != 9:
                continue
            if re.fullmatch(r"\d{1,2}:\d{2}(?:\.\d+)?", cells[0]):
                clock = cells[0]
            elif cells[0] not in ("--", ""):
                continue
            sides = [(side, cells[index]) for side, index in (("visiting", 1), ("home", 5)) if cells[index]]
            if len(sides) != 1:
                continue
            side, value = sides[0]
            description, _, actor = value.partition(" by ")
            action, _, detail = description.partition(" ")
            actor = actor.split("(", 1)[0].strip().upper().replace(", ", ",")
            plays.append(dict(action=action, type=detail.strip(), teamId=side, time=clock, checkname=actor))
        book["periodsWithPlays"].append({"number": period, "plays": plays})
    return book


def parse_gamebook(html, neutral):
    try:
        return sidearm_payload(html)
    except ValueError:
        return legacy_gamebook(html, neutral)


def gamebook_urls(season):
    path = ROOT / f"school_gamebooks_{season}.csv"
    if not path.exists():
        return {}
    urls = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            urls[row["event"]].append(row["url"])
    return urls


def fetch_html(url):
    request = Request(url, headers={"User-Agent": "UCSD-Recruiting-Player-Ratings/1.0"})
    with urlopen(request, timeout=15) as response:
        html = response.read(8_000_001)
    if len(html) > 8_000_000:
        raise ValueError("School game book exceeds size limit")
    return html.decode("utf-8")


def school_schedules(season):
    with (ROOT / "school_gamebook_sites.csv").open(newline="", encoding="utf-8") as source:
        return {
            row["team"]: row["schedule"].format(season=f"{season - 1}-{season % 100:02}",
                                                 start_year=season - 1)
            for row in csv.DictReader(source)
        }


def school_aliases(season, roster_rows):
    path = ROOT / f"school_player_aliases_{season}.csv"
    if not path.exists():
        return {}
    jerseys = {(row["team_id"], row["player_id"]): row["jersey"] for row in roster_rows}
    with path.open(newline="", encoding="utf-8") as source:
        aliases = {}
        for row in csv.DictReader(source):
            team, player, jersey = row["team_id"], row["player_id"], row["jersey"]
            if jerseys.get((team, player), "").lstrip("0") == jersey.lstrip("0"):
                aliases[team, key(row["name"]), str(int(jersey))] = player
        return aliases


MONTH_NUMBER = {month.casefold(): number for number, month in enumerate(month_abbr) if month}
SCHEDULE_DATE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+(\d{1,2})\b",
    re.I,
)
BOX_SCORE_PATH = re.compile(r"/(?:box[-_]?scores?)(?:/|\.aspx$|$)", re.I)


def month_day(label):
    found = SCHEDULE_DATE.search(label)
    if found:
        return MONTH_NUMBER[found[1][:3].casefold()], int(found[2])
    return None


class ScheduleLinks(HTMLParser):
    """Read Sidearm, WMT and Presto box-score links from official schedules."""

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule
        self.host = urlsplit(schedule).netloc.casefold().removeprefix("www.")
        self.links = {}
        self.reading_date = False
        self.event_date = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = (attrs.get("class") or "").split()
        if "schedule-event-item" in classes or "sidearm-schedule-game" in classes:
            self.event_date = None
        if tag == "time" and "schedule-event-date__month-day" in classes:
            self.reading_date = True
        if tag != "a" or not attrs.get("href"):
            return
        url = urljoin(self.schedule, attrs["href"]).split("#", 1)[0]
        link = urlsplit(url)
        if (link.scheme not in {"http", "https"}
                or link.netloc.casefold().removeprefix("www.") != self.host
                or not BOX_SCORE_PATH.search(link.path)
                or link.path.lower().endswith(".pdf")):
            return
        label = attrs.get("aria-label") or attrs.get("title") or ""
        played = month_day(label) or self.event_date
        if not played:
            encoded = re.search(r"/(20\d{2})(\d{2})(\d{2})(?:_|\.)", link.path)
            if encoded:
                played = int(encoded[2]), int(encoded[3])
        if played or url not in self.links:
            self.links[url] = played

    def handle_data(self, data):
        if self.reading_date:
            self.event_date = month_day(data) or self.event_date

    def handle_endtag(self, tag):
        if tag == "time":
            self.reading_date = False


@lru_cache(maxsize=512)
def schedule_gamebooks(schedule):
    try:
        html = fetch_html(schedule)
        parser = ScheduleLinks(schedule)
        parser.feed(html)
    except (OSError, ValueError):
        return ()
    return tuple(parser.links.items())


def candidate_urls(game, teams, exact_urls, schedules):
    seen = set(exact_urls)
    yield from exact_urls
    played = datetime.fromisoformat(game["game_date"])
    for side, other in (("home", "away"), ("away", "home")):
        schedule = schedules.get(teams[game[f"{side}_team_id"]])
        if not schedule:
            continue
        opponent = key(teams[game[f"{other}_team_id"]])
        date_key = played.month, played.day
        links = sorted(reversed(schedule_gamebooks(schedule)),
                       key=lambda entry: entry[1] == date_key, reverse=True)
        for url, day in links:
            pieces = urlsplit(url).path.rstrip("/").split("/")
            opponent_slug = pieces[-3] if len(pieces) >= 4 and pieces[-2] == "boxscore" else ""
            if url not in seen and (day == date_key or key(opponent_slug) == opponent):
                seen.add(url)
                yield url


def school_gamebook(book, game, teams, player_ids, jersey_ids, roster, aliases):
    if book["itsWomens"] or datetime.strptime(book["gameDate"], "%m/%d/%y").date().isoformat() != game["game_date"]:
        raise ValueError("School game book has the wrong sport or game date")
    scheduled = {game[f"{side}_team_id"]: game[f"{side}_score"] for side in ("home", "away")}
    by_name = {key(teams[team]): team for team in scheduled}
    school_names = {"home": book["homeTeamName"], "away": book["visitingTeamName"]}
    clubs = {side: by_name.get(key(name)) for side, name in school_names.items()}
    if set(clubs.values()) != set(scheduled):
        raise ValueError("School game book does not match the two scheduled teams")
    scores = {"home": book["homeTeamScore"], "away": book["visitingTeamScore"]}
    if any(int(scores[side]) != scheduled[team] for side, team in clubs.items()):
        raise ValueError("School game book does not match the independent final score")

    box, mapped, short_stints = [], {}, {}
    for side, group in (("home", "homeTeam"), ("away", "visitingTeam")):
        team = clubs[side]
        for row in book[group]["players"]:
            check = row["checkName"]
            if check == "TEAM" or int(row["gamePlayed"]) != 1:
                continue
            last, first = check.rsplit(",", 1)
            name = row.get("playerFirstLastName") or f"{first} {last}"
            jersey = row["uniform"]
            player = player_ids.get((team, key(name)))
            if player is None and jersey.isdigit() and key(first):
                uniform = str(int(jersey))
                player = aliases.get((team, key(name), uniform))
                player = player or jersey_ids.get((team, uniform, key(last), key(first)[0]))
            if player is None:
                raise ValueError(f"School player {name} #{jersey} has no unique NCAA roster ID")
            if (team, check) in mapped or (team, player) in mapped.values():
                raise ValueError(f"School player {name} is duplicated")
            mapped[team, check] = team, player
            stats = {label: int(row["stats"][field]) for label, field in STAT_FIELDS.items()}
            stats["two_made"] -= stats["threes"]
            stats["two_attempts"] -= stats["three_attempts"]
            mins = sidearm_minutes(row["stats"])
            if mins == 0 and any(stats.values()) and not int(row["gameStarted"]):
                short_stints[len(box)] = team, check
            box.append((team, roster[team, player], player, mins, stats))
    if short_stints:
        _, _, plays = sidearm_game(book, clubs)
        for index, (team, check) in short_stints.items():
            label, name, player, _, stats = box[index]
            box[index] = label, name, player, short_stint_minutes(plays, team, check), stats
    return box, mapped, clubs, scheduled


def corroborated_team_baskets(book, clubs, summary, event):
    if not summary:
        return ()
    try:
        _, _, school_plays = sidearm_game(book, clubs)
        if not any(play.kind == "team_basket" for play in school_plays):
            return ()
        espn_clubs, _, _, espn_plays = espn_game(summary, event)
    except (KeyError, TypeError, ValueError):
        return ()
    identifiers = {team: clubs[side] for side, team in espn_clubs.items()}
    recorded = Counter((p.period, p.clock, identifiers[p.team], p.points)
                       for p in espn_plays if p.kind == "team_basket" and p.team in identifiers)
    official = Counter((p.period, p.clock, p.team, p.points)
                       for p in school_plays if p.kind == "team_basket")
    return tuple((recorded & official).elements())


def school_possessions(book, mapped, clubs, final, roster, player_ids, *, event="",
                       team_baskets=(), espn_minutes=None):
    def actor_key(name):
        last, _, first = name.rpartition(",")
        return key(f"{first} {last}" if last else first)

    names = unique(((team, actor_key(name)), name) for team, name in mapped)
    box_by_id = {player: name for (_, name), player in mapped.items()}

    def resolve_actor(team, name):
        normalized = actor_key(name)
        match = names.get((team, normalized))
        if match is None:
            match = box_by_id.get((team, player_ids.get((team, normalized))))
        if match is None and book.get("legacy") and len(name) == 20:
            candidates = [full for club, full in mapped if club == team and full.startswith(name)]
            match = candidates[0] if len(candidates) == 1 else None
        return match or name

    school_final, players, plays = sidearm_game(book, clubs, resolve_actor=resolve_actor)
    periods = [int(period["number"]) for period in book["periodsWithPlays"] if period["plays"]]
    if school_final != final or len(periods) < 2 or periods != list(range(1, len(periods) + 1)):
        raise ValueError("School play-by-play is missing or has the wrong final score")
    if swap := SCHOOL_DUPLICATE_SWAPS.get(event):
        period, first, repeated, actors = swap
        away = clubs["away"]
        earlier = {(p.kind, p.player) for p in plays if (p.period, p.clock, p.team) == (period, first, away)}
        if actors <= earlier:
            plays = [p for p in plays if (p.period, p.clock, p.team) != (period, repeated, away)
                     or (p.kind, p.player) not in actors]
    observations, minutes, bad_seconds, _, _ = replay(plays, players, final, team_baskets=team_baskets)
    if bad_seconds:
        raise ValueError("School play-by-play does not reproduce box-score playing times")
    minute_review = []
    for actor, row in players.items():
        played = minutes[actor]
        if abs(played - row[3]) <= 1:
            continue
        team, player = mapped[actor]
        verified = SCHOOL_REPLAY_MINUTES.get((event, team, player))
        espn_time = (espn_minutes or {}).get((team, player, row[4]))
        if espn_time is not None and abs(played - espn_time) <= 5 / 60:
            minute_review.append(dict(player_id=player, school_minutes=row[3],
                                      replay_minutes=round(played, 3), espn_replay_minutes=round(espn_time, 3),
                                      source_urls=[f"https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/{event}"]))
            continue
        if verified is None or row[3] != verified[0] or abs(played - verified[1]) > 0.05:
            raise ValueError("School play-by-play does not reproduce box-score playing times")
        minute_review.append(dict(player_id=player, school_minutes=row[3], replay_minutes=round(played, 3),
                                  verified_minutes=verified[1], source_urls=verified[2]))
    possessions, counts, excluded = [], Counter(), 0
    for action, offense, points, lineup, disputed in observations:
        if disputed:
            excluded += 1
            continue
        lineup_by_team = dict(lineup)
        defense = next(team for team in final if team != offense)
        ordered = tuple(player for team in (offense, defense)
                        for player in sorted(mapped[team, actor] for actor in lineup_by_team[team]))
        if len({player for _, player in ordered}) != 10:
            raise ValueError("School play-by-play does not resolve ten distinct players")
        counts[offense] += 1
        possessions.append([action.period, action.clock, ordered, points, [roster[p] for p in ordered]])
    if min(counts[team] for team in final) < 40 or abs(counts[clubs["home"]] - counts[clubs["away"]]) > 3:
        raise ValueError("School play-by-play has unbalanced possession counts")
    return possessions, excluded, minute_review
