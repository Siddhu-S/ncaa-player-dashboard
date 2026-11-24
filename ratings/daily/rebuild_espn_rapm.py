"""Rebuild the audited November 2026 RAPM backfill from cached game sources."""
import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import groupby, pairwise
from pathlib import Path

from player_ratings import ROOT, feed_rows, game_logs, key, roster_jersey_ids, roster_jersey_player, unique
from rapm_repairs import MISSING_ROSTER_PLAYERS

SOURCE_NOTES = {
    "401911534": ("Broadcast review confirms Moeller replaced Criss at 2H 12:48 after Criss's foul. "
                  "The official game book confirms the order of three free-throw trips.",
                  "https://utepminers.com/sports/mens-basketball/stats/2026-27/nm-state/boxscore/12558"),
    "401925941": ("The school box has no Dallin Grant appearance. The substitution log records "
                  "his same-clock entry and exit before Vandermyde enters, then repeats the swap.",
                  "https://gohuskies.com/sports/mens-basketball/stats/2026-27/utah-tech/boxscore/25436"),
    "401928354": ("UC Santa Barbara's log records Vartiainen entering at 7:35 and leaving at 7:17. "
                  "The redundant Little in/out at 7:35 contributes no playing time.",
                  "https://ucsbgauchos.com/sports/mens-basketball/stats/2026-27/pepperdine/boxscore/12896"),
    "401925940": ("Utah State's game log records Brown's layup before the "
                  "substitutions at 18:19 in the second half.",
                  "https://utahstateaggies.com/sports/mens-basketball/stats/2026-27/utah-tech/boxscore/16847"),
    "401929003": ("Hawai‘i's game log records Nance's offensive-foul turnover before "
                  "the substitutions at 10:21.",
                  "https://hawaiiathletics.com/boxscore.aspx?id=31833&path=mbball"),
    "401919759": ("School play-by-play orders Jevtic's layup before the substitutions at "
                  "15:54 and Duchamp's entry after the foul at 7:01.",
                  "https://lionathletics.com/boxscore.aspx?id=11083&path=mbball"),
    "401924911": ("UNLV's official play-by-play credits the 1H 3:43 three to Max Majerle.",
                  "https://unlvrebels.com/sports/mens-basketball/stats/2026-27/northern-arizona/boxscore/17950"),
    "401921747": ("Green Bay and ESPN agree on the three scorer credits. Cincinnati and Torvik "
                  "transfer one three from Collins to Colvin. The disputed candidates with both "
                  "players on court have identical RAPM lineups and points.",
                  "https://greenbayphoenix.com/sports/mens-basketball/stats/2026-27/cincinnati/boxscore/10693"),
    "401920371": ("UC San Diego credits the 2H 8:39 three to Leo Beath and records the "
                  "2H 14:48 offensive rebound before substitutions.",
                  "https://ucsdtritons.com/boxscore.aspx?id=12384&path=mbball"),
    "401924951": ("The official log also records an 11-second gap between Houston Christian's "
                  "exit and entry. The possession spanning it is excluded.",
                  "https://goleathernecks.com/boxscore.aspx?id=14882&path=mbball"),
    "401920306": ("The official log records a Georgia Tech defensive rebound and steal at the "
                  "same stoppage. Excluded the intervening Eastern Kentucky turnover observation "
                  "because the control sequence is ambiguous.",
                  "https://ekusports.com/boxscore.aspx?id=9906&path=mbball"),
    "401921809": ("The LSU game book identifies Ahmed Nedal Abdelrahman as #7. ESPN lists "
                  "him as Ahmed Nedal #9.",
                  "https://static.lsusports.net/assets/docs/mb/27stats/ls02.pdf"),
    "401929002": ("Hawai‘i's official box lists Kajus Mikalauskas, who is absent from the Torvik "
                  "game box. He is included in the replay based on the substitutions.",
                  "https://hawaiiathletics.com/boxscore.aspx?id=31831&path=mbball"),
    "401921141": ("Elon's game book shows Caleb Zhu's redundant out/in at the same first-half "
                  "7:03 dead ball, after both free throws.",
                  "https://elonphoenix.com/sports/mens-basketball/stats/2026-27/south-carolina/boxscore/10017"),
    "401929176": ("The school log shows Booth's dunk at 17:01 followed by Tate's foul and "
                  "Booth's single free throw at 17:00, with no intervening possession.",
                  "https://gobison.com/sports/mens-basketball/stats/2026-27/colorado-state/boxscore/17736"),
    "401922047": ("User review of the broadcast confirms Longwood controlled the ball before its "
                  "timeout and inbound, then lost possession on the 1H 2:25 loose-ball foul. "
                  "Control around the 2H 3:37 foul remains unconfirmed.",
                  "https://www.espn.com/watch/player/_/id/ac001f9d-e2d2-4ab7-8006-714c7b5991ac"),
}
# Match the exact provider events so a corrected upstream log can pass as-is.
REDUNDANT_SUBSTITUTIONS = {
    "401925941": {
        (2, 629, "in", "5176272"), (2, 629, "out", "5176272"),
        (2, 554, "out", "5176272"), (2, 554, "in", "5241401"),
    },
}
SUBSTITUTION_ACTORS = {
    "401928354": {(2, 455, "in", "5033017"): "5175397"},
}
# Independent school boxes and play-by-play corroborate these replay times.
VERIFIED_REPLAY_MINUTES = {
    # Both school play logs and ESPN reconstruct 9:07. The box lists 8.
    "401928830": ("https://gozips.com/sports/mens-basketball/stats/2026-27/wright-state/boxscore/11635",
                  {"Marvin Musiime-Kamali": 547 / 60}),
    "401925740": ("https://arkansasrazorbacks.com/wp-content/uploads/2026/11/Gm-6-Stats-vs-Miss-Valley-State.pdf",
                  {"Ilia Frolov": (3 * 60 + 29) / 60}),
    "401920570": ("https://ecupirates.com/sports/mens-basketball/stats/2026/southern-ill-/boxscore/14335",
                  {"Dreylan Creasy-Quirindoongo": (60 + 50) / 60}),
    "401911239": ("https://richmondspiders.com/sports/mens-basketball/stats/2026-27/southern-illinois/boxscore/12124",
                  {"Sam Simmons Jr.": 9, "Brock McClure": 2}),
    # The official substitution log gives stints of 2:20, 1:42 and 1:29.
    "401918994": ("https://tsutigers.com/sports/mens-basketball/stats/2026-27/south-florida/boxscore/6349",
                  {"Ajayi Simmons": 331 / 60}),
    "401921086": ("https://umbcretrievers.com/sports/mens-basketball/stats/2026-27/morgan-state-university/boxscore/9692",
                  {"Manok Lual": 3, "Matar Wade": 2}),
}
ESPN_NAME_VARIANTS = {
    # Akron lists the NCAA roster's Halil Barre as Chabi Barre, #5.
    # https://gozips.com/sports/mens-basketball/roster/chabi-barre/10210
    ("2006", "Chabi Barre"): ("11710410", "5", 76774),
    # South Alabama's official roster lists the same #20 as Keyne Costa.
    ("6", "Keyne Costa"): ("11644899", "20", None),
    # San Diego State lists its #45, Jeremiah Cherry, by his nickname.
    # https://goaztecs.com/sports/mens-basketball/roster/player/bear-cherry
    ("21", "Bear Cherry"): ("11954181", "45", None),
    # The official Alabama State box lists Terrence Hampton at #30.
    # https://bamastatesports.com/boxscore.aspx?id=11641&path=mbball
    ("2011", "Terrance Hampton"): ("12154306", "33", None),
    # Same-team aliases checked against roster jerseys and Torvik game boxes.
    # November 18 aliases also verified against official school game books or rosters.
    ("236", "Matt Hampton"): ("11956451", "1", 144475),
    ("119", "Jon Barton"): ("11481891", "5", 134457),
    ("2344", "Kany Musungay"): ("11955697", "23", 126855),
    ("71", "Ben Thornbrue"): ("11956454", "4", 144476),
    ("2473", "Nate Ahner"): ("11637451", "22", 143948),
    ("21", "Bear Cherry"): ("11954181", "45", 128446),
    ("309", "Derrick Florence III"): ("11593023", "12", 143893),
    ("2539", "Abdul Bashir"): ("12154001", "5", 135357),
    ("2210", "Josh Dalton"): ("12145119", "1", 135123),
    ("2065", "Daniel Perez-Rouzan"): ("12156694", "23", 76244),
    ("2514", "Mike Parrotta"): ("12157957", "24", 145562),
    ("2115", "Daniel James"): ("11485090", "22", None),
    ("2115", "Matt Goodwin"): ("11938443", "6", 144161),
    ("2916", "Mekhi Conner"): ("12148839", "3", 144871),
    ("324", "Q Williams IV"): ("12146992", "4", 144738),
    ("2378", "JP Ndong"): ("11956726", "2", 144524),
    ("2771", "AJ Lesburt Jr."): ("11634490", "33", 77331),
    ("2771", "Elijah Wilborn"): ("11634485", "5", 78112),
    ("2771", "JP Skoric"): ("11634487", "10", 143923),
    ("2771", "Jacob Moore"): ("11484845", "14", 134306),
    ("2350", "Chris Jeffrey"): ("12151259", "6", 134243),
    ("2350", "Nic Anderson"): ("11482008", "8", 134120),
    ("2698", "Lou Hutchinson III"): ("12148075", "24", 76872),
    ("36", "Pops Dunson"): ("11565248", "2", 143749),
    ("70", "Mo Diao"): ("12149782", "34", 144934),
    ("70", "Will Jenson"): ("11476748", "32", 134466),
    ("2441", "Andy Ramirez"): ("11591223", "22", 143849),
    ("2441", "Brad Ballinger"): ("11591219", "25", 143846),
    ("2569", "Chris Parker"): ("11481808", "55", 135166),
    ("140", "Josh Kori"): ("12154180", "7", 145372),
    ("2572", "Max Blank"): ("11949413", "49", 144251),
    ("2572", "Mike Montano"): ("11949419", "6", 144256),
    ("2640", "Ahmed Nedal"): ("11474386", "9", 134870),
    ("2640", "Ben Vungo"): ("12158490", "5", 79465),
    ("2640", "Trey Lewis"): ("12340852", "12", 145638),
    ("2710", "Tayelor Moore"): ("12158056", "5", 145579),
    ("2320", "Mike Collins"): ("12148395", "7", 134643),
    ("2166", "JaQualon Roberts"): ("11482014", "24", 77890),
    ("2415", "Khali Horton"): ("11956116", "7", 79085),
    ("315", "Tae Blackshear"): ("11638289", "77", 133820),
    ("25", "Semetri Carr"): ("11471986", "3", 135006),
    ("399", "Zacharie Matulu"): ("11481537", "14", 79205),
    # Virginia Tech identifies Efrem “Butta” Johnson and lists him at #33.
    # https://hokiesports.com/news/2026/10/1/butta-johnson-set-to-join-hokies-mens-basketball-roster
    ("259", "Butta Johnson"): ("12155589", "33", None),
    ("257", "Sam Simmons Jr."): ("12151923", "2", 145134),
    ("2400", "Tianhao Wu"): ("12151613", "9", 145098),
    ("264", "BJ Roy"): ("11471842", "3", 135113),
    ("140", "DJ Jackson Jr."): ("12154162", "1", 145366),
    ("294", "Sam Mbingazo"): ("11592282", "35", 143865),
    ("127", "Cam Ward"): ("11471652", "3", 134374),
    ("339", "JR Kemm"): ("12151551", "5", 145087),
    ("94", "Juanma Ruiz"): ("11480811", "17", 134669),
    ("2737", "Anthony Smith III"): ("11939132", "3", 134952),
    ("2737", "Eesher Singh Sarai"): ("12153191", "6", 145291),
    ("2534", "Cheikh Sadibou Ndiaye"): ("11478497", "32", 128265),
    ("2502", "Gantas Krianauskas"): ("12149647", "22", 144919),
    ("2502", "Emil Skytt"): ("12149653", "8", 76032),
    ("249", "Cash Chavis"): ("12151866", "1", 126524),
    ("2751", "Mike Mora"): ("12153860", "1", 145340),
    ("2097", "AJ Wright Jr."): ("12153373", "1", 145300),
    ("2172", "Baba Mbengue"): ("11954700", "21", 126840),
    ("71", "Gustavo Guimares"): ("11477609", "17", 134893),
    # https://lemoynedolphins.com/sports/mens-basketball/stats/2026-27/manhattan-university/boxscore/15877
    ("2363", "Asier Miguel"): ("11480682", "11", None),
    # https://towsontigers.com/sports/mens-basketball/stats/2026-27/umbc/boxscore/29798
    ("119", "Stephon Ashley Wright"): ("12195893", "0", None),
    # https://goislanders.com/sports/mens-basketball/stats/2026-27/rice/boxscore/8574
    ("357", "Khi Wallace"): ("11590503", "7", None),
    # https://fgcuathletics.com/sports/mens-basketball/roster/2026-27
    ("526", "BB Washington"): ("11479253", "8", None),
    ("526", "CJ Outlaw"): ("11593006", "45", None),
    # https://ucasports.com/roster.aspx?path=mbball&season=2026-27
    ("2110", "Harry Beauchamp"): ("11475344", "9", None),
    ("2110", "J.C. Shulman"): ("11475506", "4", None),
    ("2110", "R.J. Young"): ("12148184", "0", None),
    ("2110", "Simm-Marten Saadi"): ("12148179", "35", None),
}
ESPN_ROSTER_JERSEYS = {"11474386": "7", "11478497": "23", "12154306": "30"}
SHOT_CORRECTIONS = {
    "401924911": (1, 223, "5308449", "5108047"),
    "401920371": (2, 519, "5107287", "5106867"),
}


@dataclass(frozen=True)
class Play:
    period: int
    clock: float
    kind: str
    team: str = ""
    player: str = ""
    points: int = 0
    last_free_throw: bool = False
    single_free_throw: bool = False
    and_one: bool = False


def seconds(value):
    minutes, _, rest = value.rpartition(":")
    return int(minutes or 0) * 60 + float(rest)


def repair_utep_replay(plays):
    utep, criss, moeller = "2638", "5210813", "5313847"
    stoppage = [i for i, play in enumerate(plays) if (play.period, play.clock) == (2, 768)]
    swap = [plays[i] for i in stoppage if (plays[i].team, plays[i].player) in
            {(utep, criss), (utep, moeller)} and plays[i].kind in {"in", "out"}]
    fouls = [i for i in stoppage if (plays[i].team, plays[i].player, plays[i].kind) == (utep, criss, "foul")]
    if not swap and len(fouls) == 1:
        i = fouls[0] + 1
        plays[i:i] = [Play(2, 768, "out", utep, criss), Play(2, 768, "in", utep, moeller)]

    # Apply the school order only to the observed ESPN blocks. If the feed
    # has already corrected a block, use its order as supplied.
    free_throw_trips = (
        (1024, "5312326", "foul free_throw out in free_throw free_throw dead_rebound",
         [(1, False), (1, True), (0, False)], (0, 5, 6, 1, 2, 3, 4)),
        (567, "5175354", "foul out in free_throw def_rebound free_throw",
         [(0, True), (1, False)], (0, 5, 1, 2, 3, 4)),
        (123, criss, "shot def_rebound foul out out in in out in free_throw free_throw dead_rebound",
         [(1, True), (0, False)], (0, 1, 2, 10, 11, 3, 4, 5, 6, 7, 8, 9)),
    )
    for clock, shooter, kinds, attempts, order in free_throw_trips:
        indices = [i for i, play in enumerate(plays) if (play.period, play.clock) == (2, clock)]
        block = [plays[i] for i in indices]
        if [play.kind for play in block] != kinds.split():
            continue
        free_throws = [play for play in block if play.kind == "free_throw"]
        if [(play.team, play.player, play.points, play.last_free_throw) for play in free_throws] != [
            (utep, shooter, points, last) for points, last in attempts
        ]:
            continue
        for i, source in zip(indices, order):
            plays[i] = block[source]


def espn_game(summary, event=""):
    game = summary["header"]["competitions"][0]
    teams = {club["homeAway"]: club["id"] for club in game["competitors"]}
    final = {club["id"]: int(club["score"]) for club in game["competitors"]}
    players = {}
    for block in summary["boxscore"]["players"]:
        team = block["team"]["id"]
        for stats in block["statistics"]:
            if "MIN" not in stats["labels"]:
                continue
            for row in stats["athletes"]:
                if row.get("didNotPlay") or not row["stats"]:
                    continue
                athlete = row["athlete"]
                box = dict(zip(stats["labels"], row["stats"]))
                players[team, athlete["id"]] = (
                    athlete["displayName"], athlete.get("jersey", ""), row["starter"],
                    float(box["MIN"]), int(box["PTS"]),
                )
    plays = []
    kinds = {"586": "off_rebound", "587": "def_rebound", "598": "turnover",
             "521": "technical", "412": "end", "449": "dead_rebound", "607": "steal",
             "618": "block", "519": "foul"}
    for row in summary.get("plays", []):
        play_type = row["type"]["id"]
        text = row["text"]
        kind = kinds.get(play_type, "")
        last = single = False
        if play_type == "584":
            kind = "in" if "subbing in" in text else "out" if "subbing out" in text else "bad_sub"
        elif play_type == "540":
            kind = "free_throw"
            attempt = re.search(r"free throw (\d+) of (\d+)", text)
            if attempt is None:
                raise ValueError(f"unrecognized free throw: {text}")
            last = attempt[1] == attempt[2]
            single = attempt[1] == attempt[2] == "1"
        elif row.get("shootingPlay"):
            kind = "shot"
        elif play_type == "615" and "Jump Ball won" in text:
            kind = "jump"
        if kind:
            actors = row.get("participants", [])
            player = actors[0]["athlete"]["id"] if actors else ""
            if kind == "shot" and not player and re.match(r"team\s+makes\b", text, re.I):
                kind = "team_basket"
            plays.append(Play(row["period"]["number"], seconds(row["clock"]["displayValue"]),
                              kind, row.get("team", {}).get("id", ""), player,
                              int(row["scoreValue"]) if row.get("scoringPlay") else 0, last, single))
        elif row.get("scoringPlay"):
            raise ValueError(f"unrecognized scoring play: {text}")
    if event == "401911534":
        repair_utep_replay(plays)
    redundant = REDUNDANT_SUBSTITUTIONS.get(event, ())
    actors = SUBSTITUTION_ACTORS.get(event, {})
    if redundant or actors:
        plays = [replace(play, player=actors.get((play.period, play.clock, play.kind, play.player), play.player))
                 for play in plays if (play.period, play.clock, play.kind, play.player) not in redundant]
    if event in SHOT_CORRECTIONS:
        period, clock, original, corrected = SHOT_CORRECTIONS[event]
        index, = (i for i, p in enumerate(plays) if
                  (p.period, p.clock, p.player, p.points) == (period, clock, original, 3))
        team = plays[index].team
        plays[index] = replace(plays[index], player=corrected)
        for player, delta in ((original, -3), (corrected, 3)):
            name, jersey, starter, minutes, points = players[team, player]
            players[team, player] = name, jersey, starter, minutes, points + delta
    if event == "401920371":
        index, = (i for i, p in enumerate(plays) if (p.period, p.clock, p.kind, p.player) ==
                  (2, 888, "out", "5243038"))
        if (plays[index + 1].kind, plays[index + 1].clock) != ("off_rebound", 888):
            raise ValueError("UC San Diego substitution sequence changed")
        plays[index:index + 2] = reversed(plays[index:index + 2])
    if event == "401921141":
        out, sub_in = (i for i, play in enumerate(plays)
                       if (play.period, play.clock, play.player) == (1, 423, "5458757")
                       and play.kind in {"in", "out"})
        if (plays[out].kind != "out" or plays[sub_in].kind != "in"
                or any(play.clock != 423 or play.kind not in {"in", "out"}
                       for play in plays[out:sub_in + 1])):
            raise ValueError("Elon same-clock substitution sequence changed")
        del plays[sub_in]
        del plays[out]
    if event == "401929176":
        shot, = (i for i, play in enumerate(plays)
                 if (play.period, play.clock, play.kind, play.team, play.points) == (2, 1021, "shot", "36", 2))
        basket = plays[shot]
        following = [play for play in plays[shot + 1:] if (play.period, play.clock) == (2, 1020)]
        free_throws = [play for play in following if play.kind == "free_throw"]
        if (len(free_throws) != 1 or free_throws[0].player != basket.player
                or free_throws[0].team != basket.team or not free_throws[0].last_free_throw
                or not any(play.kind == "foul" and play.team == "2449" for play in following)
                or any(play.kind not in {"foul", "in", "out", "free_throw"} for play in following)):
            raise ValueError("Colorado State and-one sequence changed")
        plays[shot] = replace(basket, clock=1020)
    if event == "401922047":
        foul, = (i for i, play in enumerate(plays)
                 if (play.period, play.clock, play.kind, play.team, play.player) ==
                 (1, 145, "foul", "2344", "5061602"))
        basket, = (i for i, play in enumerate(plays)
                   if (play.period, play.clock, play.kind, play.team, play.player, play.points) ==
                   (1, 157, "shot", "2678", "5107351", 2))
        free_throws = [play for play in plays[foul + 1:] if (play.period, play.clock) == (1, 145)
                       and play.kind == "free_throw"]
        if (basket >= foul or any(play.kind not in {"in", "out"} for play in plays[basket + 1:foul])
                or [(p.team, p.player, p.last_free_throw) for p in free_throws] !=
                [("2678", "5314582", False), ("2678", "5314582", True)]):
            raise ValueError("VMI–Longwood first-half loose-ball foul sequence changed")
        plays[foul] = replace(plays[foul], kind="possession_lost_on_foul")
    moves = {
        "401919759": ((1, 954, "shot", None), (1, 421, "foul", None)),
        "401929003": ((1, 621, "turnover", "foul"),),
        "401925940": ((2, 1099, "shot", None),),
    }
    for period, clock, kind, after in moves.get(event, ()):
        indices = [i for i, p in enumerate(plays) if (p.period, p.clock) == (period, clock)]
        index, = (i for i in indices if plays[i].kind == kind)
        destination = indices[0] if after is None else next(
            i + 1 for i in indices if plays[i].kind == after
        )
        plays.insert(destination, plays.pop(index))
    # ESPN can give the basket and its continuation free throw different clocks.
    # Only use the next live action in the same period, stopping at technicals.
    following = None
    for i in range(len(plays) - 1, -1, -1):
        play = plays[i]
        if (play.kind == "shot" and play.points and following and following.single_free_throw
                and (following.period, following.team, following.player)
                == (play.period, play.team, play.player)):
            plays[i] = replace(play, and_one=True)
        if play.kind not in {"in", "out", "foul", "dead_rebound", "block", "steal"}:
            following = play
    return teams, final, players, plays


def sidearm_payload(html):
    """Read the structured game book embedded in a SIDEARM box-score page."""
    script = re.search(r'id="__NUXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if script is None:
        raise ValueError("No structured SIDEARM game book")
    encoded = json.loads(script[1])

    def decode(index):
        if index < 0:
            return None
        value = encoded[index]
        if isinstance(value, dict):
            return {field: decode(ref) for field, ref in value.items()}
        if isinstance(value, list):
            return [decode(ref) for ref in value]
        return value

    index = next((i for i, row in enumerate(encoded)
                  if isinstance(row, dict) and "periodsWithPlays" in row), None)
    if index is None:
        raise ValueError("No structured SIDEARM game book")
    return decode(index)


def memphis_game(html, teams):
    return sidearm_game(sidearm_payload(html), teams)


def sidearm_minutes(stats):
    seconds_played = stats.get("secondsPlayed") or 0
    return float(stats["minutesPlayed"]) + (0 if seconds_played == "+" else float(seconds_played)) / 60


def sidearm_game(box, teams, *, resolve_actor=None):
    clubs = {box["homeTeamId"]: teams["home"], box["visitingTeamId"]: teams["away"]}
    final, players = {}, {}
    for side in ("homeTeam", "visitingTeam"):
        block = box[side]
        team = clubs[block["id"]]
        final[team] = int(block["scoreSummary"]["score"])
        for row in block["players"]:
            if not int(row["gamePlayed"]) or row["checkName"] == "TEAM":
                continue
            last, first = row["checkName"].rsplit(",", 1)
            players[team, row["checkName"]] = (
                f"{first} {last}", row["uniform"], bool(int(row["gameStarted"])),
                sidearm_minutes(row["stats"]),
                int(row["stats"]["totalPointsScored"]),
            )

    plays = []
    score = (0, 0)
    for period in box["periodsWithPlays"]:
        number = int(period["number"])
        for row in period["plays"]:
            action, detail = row["action"], row["type"]
            points = 0
            if action in {"GOOD", "MISS"}:
                kind = "free_throw" if detail == "FT" else "shot"
                if action == "GOOD":
                    points = 1 if detail == "FT" else 3 if detail == "3PTR" else 2
            elif action == "SUB":
                kind = detail.lower()
            elif action == "REBOUND":
                kind = {"DEF": "def_rebound", "OFF": "off_rebound", "DEADB": "dead_rebound"}[detail]
            else:
                kind = {"TURNOVER": "turnover", "FOUL": "technical" if detail == "TECH" else "foul", "STEAL": "steal",
                        "BLOCK": "block"}.get(action)
            if kind:
                team = clubs.get(row["teamId"], "")
                next_score = score
                if points and row.get("homeScore") is not None and row.get("visitingScore") is not None:
                    next_score = int(row["homeScore"]), int(row["visitingScore"])
                player = row["checkname"]
                if player == "TEAM":
                    player = ""
                    if kind == "shot" and points:
                        kind = "team_basket"
                        delta = tuple(after - before for before, after in zip(score, next_score))
                        if delta == (points, 0):
                            team = teams["home"]
                        elif delta == (0, points):
                            team = teams["away"]
                elif player and resolve_actor and (team, player) not in players:
                    player = resolve_actor(team, player)
                plays.append(Play(number, seconds(row["time"]), kind, team, player, points))
                score = next_score
        plays.append(Play(number, 0, "end"))
    # The school reports individual attempts. At a dead ball, consecutive
    # attempts by the same shooter are one trip.
    for i, play in enumerate(plays):
        if play.kind == "free_throw":
            following = next((p for p in plays[i + 1:] if p.kind not in
                              {"in", "out", "dead_rebound", "foul"}), None)
            same_trip = following and (following.kind, following.period, following.clock,
                                       following.team, following.player) == (
                "free_throw", play.period, play.clock, play.team, play.player)
            plays[i] = replace(play, last_free_throw=not same_trip)
    return final, players, plays


def short_stint_minutes(plays, team, player):
    """Recover a rounded-zero bench appearance from complete substitution pairs."""
    substitutions = list(dict.fromkeys(
        p for p in plays if (p.team, p.player) == (team, player) and p.kind in {"in", "out"}
    ))
    if not substitutions or len(substitutions) % 2:
        return 0
    elapsed = 0
    for entry, exit in zip(substitutions[::2], substitutions[1::2]):
        if entry.kind != "in" or exit.kind != "out" or entry.period != exit.period:
            return 0
        if entry.clock < exit.clock:
            return 0
        elapsed += entry.clock - exit.clock
    return elapsed / 60 if 0 < elapsed < 60 else 0


def remove_zero_time_substitutions(plays, players):
    # Adjacent in/out records for an athlete excluded from the box do not
    # change a lineup if no game time or basketball action separates them.
    appearances = Counter((play.team, play.player) for play in plays if play.player)
    ignored = set()
    for index, (entry, exit) in enumerate(pairwise(plays)):
        actor = entry.team, entry.player
        if (entry.player and entry.kind == "in" and exit.kind == "out"
                and actor not in players and appearances[actor] == 2
                and (entry.period, entry.clock, *actor) == (exit.period, exit.clock, exit.team, exit.player)):
            ignored.update((index, index + 1))
    return [play for index, play in enumerate(plays) if index not in ignored]


def replay(plays, players, final, *, team_baskets=()):
    lineups = {team: {player for (club, player), row in players.items() if club == team and row[2]}
               for team in final}
    if any(len(lineup) != 5 for lineup in lineups.values()):
        raise ValueError("missing starters")
    other = dict(zip(final, reversed(tuple(final))))
    minutes, scoring, player_points, actor_errors = (Counter() for _ in range(4))
    possessions = []
    bad_seconds = 0
    offense, anchor, points, disputed = "", None, 0, False
    technical = Counter()
    technical_clock = None
    technical_actors = set()
    technical_points = Counter()
    verified_team_baskets = Counter(team_baskets)
    previous = None

    def tick(elapsed):
        nonlocal bad_seconds, disputed
        if elapsed < 0:
            raise ValueError("game clock reversed")
        for team, lineup in lineups.items():
            if len(lineup) != 5:
                bad_seconds += elapsed
                disputed |= elapsed > 0
            for player in lineup:
                minutes[team, player] += elapsed / 60

    def finish():
        nonlocal offense, anchor, points, disputed
        if anchor is not None:
            action, lineup = anchor
            possessions.append((action, offense, points, lineup, disputed))
        offense, anchor, points, disputed = "", None, 0, False

    plays = remove_zero_time_substitutions(plays, players)
    next_live = {}
    following = None
    for play in reversed(plays):
        if play.kind == "turnover":
            next_live[play] = following
        if play.kind in {"shot", "team_basket", "turnover", "off_rebound", "def_rebound", "jump", "end"}:
            following = play
    for period, half in groupby(plays, key=lambda p: p.period):
        clock = 1200 if period <= 2 else 300
        if previous is not None and period != previous + 1:
            raise ValueError("missing period")
        previous = period
        for time, at_clock in groupby(half, key=lambda p: p.clock):
            block = list(at_clock)
            tick(clock - time)
            clock = time
            if technical_clock != time:
                technical.clear()
                technical_actors.clear()
            applied_substitutions = set()
            for i, play in enumerate(block):
                kind, team, player = play.kind, play.team, play.player
                if kind in {"in", "out", "bad_sub"}:
                    if (team, player) not in players or kind == "bad_sub":
                        raise ValueError("unknown substitution")
                    if (player in lineups[team]) != (kind == "out"):
                        if play in applied_substitutions:
                            continue
                        raise ValueError("conflicting substitution")
                    applied_substitutions.add(play)
                    if kind == "in":
                        lineups[team].add(player)
                    else:
                        lineups[team].remove(player)
                    continue
                if kind == "team_basket":
                    evidence_key = period, time, team, play.points
                    if verified_team_baskets[evidence_key] <= 0:
                        raise ValueError("Team-only basket is not corroborated")
                    verified_team_baskets[evidence_key] -= 1
                    kind = "shot"
                elif play.points:
                    player_points[team, player] += play.points
                if play.points:
                    scoring[team] += play.points
                if kind == "technical":
                    technical[other[team]] += 2
                    technical_actors.add((team, player))
                    technical_clock = time
                    continue
                if kind == "free_throw" and technical[team]:
                    technical[team] -= 1
                    technical_points[team] += play.points
                    continue
                if kind == "turnover" and (team, player) in technical_actors and technical[other[team]]:
                    following = next_live.get(play)
                    if following and following.period == period and following.team == team and following.kind == "shot":
                        # A bookkeeping turnover attached to a technical does
                        # not end the trip when the charged team retains control.
                        continue
                if kind == "end":
                    finish()
                    continue
                if kind == "jump":
                    if offense and team != offense:
                        finish()
                    continue
                if kind not in {"shot", "free_throw", "turnover", "off_rebound", "def_rebound",
                                "possession_lost_on_foul"}:
                    continue
                if team not in lineups:
                    raise ValueError("live play missing team")
                wrong_actor = player and player not in lineups[team]
                if wrong_actor:
                    actor_errors[period, time, team, kind, player] += 1
                owner = other[team] if kind == "def_rebound" else team
                if offense and owner != offense:
                    finish()
                if kind in {"off_rebound", "def_rebound"} and anchor is None:
                    # A rebound without a recorded live attempt provides no
                    # possession anchor. The next live play establishes control.
                    continue
                offense = owner
                snapshot = tuple((club, tuple(sorted(ids))) for club, ids in lineups.items())
                invalid = bool(wrong_actor) or any(len(ids) != 5 for _, ids in snapshot)
                disputed |= invalid
                if anchor is None:
                    anchor = (play, snapshot)
                points += play.points
                and_one = play.and_one or (kind == "shot" and play.points and any(
                    p.kind == "free_throw" and p.team == team and p.player == player
                    and p.last_free_throw for p in block[i + 1:]
                ))
                if kind in {"turnover", "def_rebound", "possession_lost_on_foul"} or (
                    kind == "shot" and play.points and not and_one
                ) or (
                    kind == "free_throw" and play.last_free_throw and play.points
                ):
                    finish()
        tick(clock)
        finish()
        technical.clear()

    if scoring != final:
        raise ValueError(f"final score mismatch {dict(scoring)} vs {final}")
    if any(player_points[p] != row[4] for p, row in players.items()) or set(player_points) - players.keys():
        raise ValueError("individual play-by-play scoring disagrees with player box")
    recovered_points = Counter(technical_points)
    for _, team, points, _, _ in possessions:
        recovered_points[team] += points
    if recovered_points != final:
        raise ValueError("possession points do not reconcile")
    return possessions, minutes, bad_seconds, actor_errors, technical_points


def reconstruct(inputs, sources, match_file, destination, report_file, season=2027, *,
                use_torvik=True, ncaa_minutes=None):
    team_names = {row["id"]: row["team"] for row in feed_rows(inputs, "team_ids", season)}
    espn_to_ncaa = unique(
        (row[f"{side}_espn_team_id"], row[f"{side}_ncaa_team_id"])
        for row in feed_rows(inputs, "pbp", season) for side in ("home", "away")
        if row[f"{side}_espn_team_id"] and row[f"{side}_ncaa_team_id"] in team_names
    )
    team_by_name = unique((key(name), team) for team, name in team_names.items())
    roster = list(feed_rows(inputs, "team_rosters", season))
    ncaa_players = unique(
        ((row["team_id"], key(name)), row["player_id"])
        for row in roster for name in (row["name"], row["player"], row["clean_name"])
        if name and row["player_id"]
    )
    jersey_ids = roster_jersey_ids(roster)
    roster_by_id = {(row["team_id"], row["player_id"]): row for row in roster}
    for (team, name), (player, _) in MISSING_ROSTER_PLAYERS.items():
        ncaa_players.setdefault((team, key(name)), player)
    box_games = defaultdict(dict)
    for row in game_logs(inputs, season, False) if use_torvik else ():
        box_games[row[0], row[47], row[5]][key(row[48])] = row

    with match_file.open(newline="", encoding="utf-8") as source:
        matches = list(csv.DictReader(source))
    reports, output = [], []
    for match in matches:
        event = match["espn_event_id"]
        report = {field: match[field] for field in
                  ("date", "home_team", "away_team", "espn_event_id", "source", "play_by_play_url")}
        reports.append(report)
        try:
            summary = json.loads((sources / f"{event}.json").read_text())
            teams, final, players, plays = espn_game(summary, event)
            if match["source"] == "University of Memphis":
                final, players, plays = memphis_game((sources / "memphis.html").read_text(), teams)
            if not plays:
                raise ValueError("source play-by-play is empty")
            team_ids = {}
            boxes = {}
            for side, opposite in (("home", "away"), ("away", "home")):
                club = teams[side]
                team = match[f"{side}_team"]
                team_id = espn_to_ncaa.get(club) or team_by_name.get(key(team))
                if not team_id:
                    raise ValueError(f"unmapped team {team} (ESPN {club})")
                team_ids[club] = team_id
                boxes[club] = box_games[match["date"].replace("-", ""), team, match[f"{opposite}_team"]]
                if final[club] != int(match[f"box_{side}"]) or (
                    use_torvik and sum(int(row[33]) for row in boxes[club].values()) != final[club]
                ):
                    raise ValueError(f"independent team score mismatch for {team}")

            observations, minutes, bad_seconds, actor_errors, tech = replay(plays, players, final)
            if event in SOURCE_NOTES:
                report["source_review"], report["review_source_url"] = SOURCE_NOTES[event]
            if event == "401920306":
                index, = (i for i, (p, _, _, _, _) in enumerate(observations)
                          if (p.period, p.clock, p.kind, p.player) == (1, 747, "turnover", "5461520"))
                action, owner, points, lineup, _ = observations[index]
                observations[index] = action, owner, points, lineup, True
            mapped = {}
            missing = set()
            discrepancies = []
            ncaa_url, game_minutes = (ncaa_minutes or {}).get(event, (None, {}))
            box_by_id = {club: {row[51]: row for row in rows.values()} for club, rows in boxes.items()}
            for (club, player), (name, jersey, _, source_minutes, source_points) in players.items():
                pk = key(name)
                target = ncaa_players.get((team_ids[club], pk))
                box = boxes[club].get(pk)
                if event == "401918987" and pk == "sebaswilliamsadams":
                    box = box_by_id[club].get(134430)
                    target = ncaa_players.get((team_ids[club], key("Sebastian Williams-Adams")))
                variant = ESPN_NAME_VARIANTS.get((club, name))
                if variant:
                    target, source_jersey, box_id = variant
                    box = box_by_id[club].get(box_id)
                    roster_row = roster_by_id.get((team_ids[club], target))
                    official_jersey = ESPN_ROSTER_JERSEYS.get(target, source_jersey)
                    if jersey != source_jersey or not roster_row or roster_row["jersey"] != official_jersey:
                        raise ValueError(f"Roster jersey mismatch for {name}")
                if target is None:
                    target = roster_jersey_player(jersey_ids, team_ids[club], jersey, name)
                if target is None:
                    missing.add((club, name, jersey))
                else:
                    mapped[club, player] = target
                actual = minutes[club, player]
                if box is not None and int(box[33]) != source_points:
                    known_difference = {"5105401": 3, "4894452": -3}.get(player)
                    if event != "401921747" or source_points - int(box[33]) != known_difference:
                        raise ValueError(f"Independent player scoring mismatch for {name}")
                if use_torvik and box is None and (source_points or (actual > 1 and target)):
                    raise ValueError(f"Missing player box for {name}")
                diff = None if box is None else round(actual - float(box[8]), 3)
                corroborating_minutes = game_minutes.get((team_ids[club], target, source_points))
                discrepancies.append((round(actual - source_minutes, 3), diff, name, actual,
                                      corroborating_minutes))
            report["reconstructed"] = len(observations)
            report["technical_points"] = sum(tech.values())
            report["non_five_player_seconds"] = bad_seconds
            report["actor_conflicts"] = sum(actor_errors.values())
            report["source_max_minute_error"] = max(abs(row[0]) for row in discrepancies)
            report["torvik_max_minute_error"] = max((abs(row[1]) for row in discrepancies if row[1] is not None), default=None)
            school_url, school_minutes = VERIFIED_REPLAY_MINUTES.get(event, (None, {}))
            timing_disputes = []
            ncaa_confirmed = False
            for source_diff, box_diff, name, actual, ncaa_time in discrepancies:
                if abs(source_diff if box_diff is None else box_diff) <= 1:
                    continue
                if name in school_minutes and abs(actual - school_minutes[name]) <= 0.5:
                    continue
                # The independent box must agree on the player, points and
                # replayed time. Keep the check on supplied Torvik boxes.
                if box_diff is None and ncaa_time is not None and abs(actual - ncaa_time) <= 0.5:
                    ncaa_confirmed = True
                    continue
                timing_disputes.append((source_diff, box_diff, name))
            if not timing_disputes:
                if school_url:
                    report["source_review"] = "Official school box corroborates disputed replay minutes"
                    report["review_source_url"] = school_url
                if ncaa_confirmed:
                    report["ncaa_minutes_source"] = ncaa_url
            report["minute_review"] = "; ".join(
                f"{name} ({source_diff:+.2f} source, {box_diff if box_diff is not None else 'unavailable'} Torvik)"
                for source_diff, box_diff, name in timing_disputes
            )
            counts = Counter(owner for _, owner, _, _, _ in observations)
            report["home_possessions"] = counts[teams["home"]]
            report["away_possessions"] = counts[teams["away"]]
            report["excluded_lineup_possessions"] = sum(disputed for _, _, _, _, disputed in observations)
            report["issue"] = ""
            if missing:
                report["issue"] = "Identity review: " + ", ".join(
                    f"{name} #{jersey} (ESPN team {team})" for team, name, jersey in sorted(missing)
                )
                continue
            if len(set((team_ids[club], pid) for (club, _), pid in mapped.items())) != len(mapped):
                raise ValueError("two source athletes map to one NCAA player")
            if (bad_seconds and (event, bad_seconds) != ("401924951", 11)) or report["minute_review"]:
                report["issue"] = "Lineup timing requires review"
                continue
            kept = Counter(owner for _, owner, _, _, disputed in observations if not disputed)
            if min(kept[team] for team in final) < 40 or abs(kept[teams["home"]] - kept[teams["away"]]) > 3:
                report["issue"] = "Unbalanced possession counts"
                continue

            game_rows = []
            for action, owner, points, lineup, disputed in observations:
                if disputed:
                    continue
                by_side = dict(lineup)
                ids = [mapped[club, player] for club in (owner, next(t for t in final if t != owner))
                       for player in by_side[club]]
                if len(set(ids)) != 10:
                    raise ValueError("non-unique NCAA lineup")
                game_rows.append((event, match["date"], team_ids[teams["home"]], team_ids[teams["away"]],
                                  final[teams["home"]], final[teams["away"]],
                                  action.period, action.clock, team_ids[owner], points, *ids))
            report["used"] = len(game_rows)
            output.extend(game_rows)
        except (KeyError, StopIteration, ValueError) as error:
            report["issue"] = str(error)

    columns = ("event", "date", "home_team_id", "away_team_id", "home_score", "away_score",
               "period", "clock", "offense", "points",
               *(f"{side}_{i}" for side in ("offense", "defense") for i in range(1, 6)))
    with gzip.open(destination, "wt", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(output)
    with report_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for row in reports for k in row)))
        writer.writeheader()
        writer.writerows(reports)
    print(f"Backfill contains {len(output)} possessions from {sum(bool(r.get('used')) for r in reports)} "
          f"of {len(matches)} games")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Local NCAA and Torvik provider files")
    parser.add_argument("--sources", type=Path, required=True, help="ESPN summaries and memphis.html")
    parser.add_argument("--matches", type=Path, default=ROOT / "espn_lineup_coverage_through_2026-11-10.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "rapm_backfill_2027.csv.gz")
    parser.add_argument("--report", type=Path, default=ROOT / "rapm_backfill_audit_through_2026-11-10.csv")
    args = parser.parse_args()
    reconstruct(args.inputs, args.sources, args.matches, args.output, args.report)
