# BPM and RAPM coverage review through November 22, 2026

The [November 23 publication run](https://github.com/Siddhu-S/ncaa-player-dashboard/actions/runs/45143773570)
reported 939 of 945 BPM games and 922 of 945 RAPM games through November 22.
The lists overlapped in four games, for 25 distinct matchups and 29 missing
metric inputs. Replaying that cohort through the daily job's collector with
input rules version 11 recovered all six BPM inputs and 20 RAPM inputs.
Three RAPM inputs still require review.

## Investigation and daily workflow changes

1. Matched the ESPN event IDs to the daily schedule and retried the original
   gaps with the same season cutoff. Compared the available SportsDataverse,
   ESPN, and school records. Checked team and player identity, final scores,
   and individual scoring before using the box and play-by-play.
2. Fixed three roster mismatches. South Alabama lists Keyne Costa under the
   same jersey as Keyne Costa Sanseverino in the NCAA roster. The official
   Cornell and TCU rosters confirmed Trey Williams and Robbie Avila. The job
   can carry their stable ESPN IDs pending a matching NCAA ID. [South Alabama roster](https://usajaguars.com/sports/mbball/roster)
   · [Cornell roster](https://cornellbigred.com/sports/mens-basketball/roster/2026-27)
   · [TCU roster](https://gofrogs.com/sports/mens-basketball/roster/robbie-avila/19316)
3. Recovered the positive playing time for UC Irvine's Dom Bentho and
   Grambling's Colin Byrd. Both official boxes rounded their minutes to zero.
   The new fallback measures complete, ordered substitution pairs for a bench
   player with recorded stats and requires a positive total under 60 seconds.
   The entire box must still pass validation.
4. Western Carolina's named player points add to 61 and the final is 63.
   At 7:00 of the second half the ESPN and school running scores both move
   from 74–41 to 74–43 on a team-only basket. The school event tags the home
   team, so the parser uses the unambiguous scoring change and checks it
   against ESPN. The bucket counts for Western Carolina's team margin and
   lineup scoring. Individual player points remain as credited.
5. Replayed starters and substitutions against both box scores. The daily
   job can now use matching school and ESPN replay times for a disputed
   player when their identity and scoring also agree and the times are
   within five seconds. Arkansas's final PDF confirms Ilia Frolov's 3:29.
   East Carolina's box and 1:50 substitution confirm Dreylan
   Creasy-Quirindoongo's 1:50 appearance. It ignores same-clock duplicate substitutions already
   reflected in the lineup and adjacent, zero-time in/out records for
   unlisted players with no other actions. The two source-confirmed historical
   corrections for Utah Tech, UC Santa Barbara, and Dayton are keyed to the
   exact source timestamp, player, and substitution direction. Dayton's
   official box includes Jaron McKie's 1:14, and Haynes's turnover between
   the two recorded swaps confirms the earlier entry.
6. Rechecked possession control. The replay skips a rebound without a
   recorded attempt to anchor a possession. A turnover recorded against
   the same player or team on a technical-foul stoppage does not close a
   possession when the charged team takes the next live shot. The normal
   lineup, scoring, minutes, and possession balance checks still apply.
7. Replayed the 25 games through the collector and checked the accepted rows
   by metric. Reconstructed RAPM can omit individual possessions whose
   lineups or control remain uncertain. Accepted fallbacks can be rechecked
   against SportsDataverse as fresh inputs arrive. Open games remain in the
   daily retry and review queue.

## Recovered inputs

Dates are in November 2026. RAPM counts are accepted possession rows and BPM
counts are accepted box input rows.

| Date | Matchup | BPM rows | RAPM rows | Evidence or change |
| --- | --- | ---: | ---: | --- |
| 11-19 | Texas A&M at TCU | 19 | 159 | Verified Robbie Avila’s identity against TCU’s roster. [Source](https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/401919050) |
| 11-19 | Notre Dame at Auburn | — | 133 | Removed possessions attributed to unanchored rebounds. [Source](https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/401920519) |
| 11-19 | Southeast Missouri State at St. Thomas | — | 133 | Rebuilt the lineup from the official school play-by-play. [Source](https://tommiesports.com/sports/mbball/stats/2026-27/semo/boxscore/10289) |
| 11-19 | Utah Tech at Washington | — | 131 | Removed Washington game book’s zero-time Dallin Grant pair and later duplicate swap. [Source](https://gohuskies.com/sports/mens-basketball/stats/2026-27/utah-tech/boxscore/25436) |
| 11-19 | UC Santa Barbara at Pepperdine | — | 133 | Restored Viljami Vartiainen’s check-in at 7:35 of the second half from the UCSB game book. [Source](https://ucsbgauchos.com/sports/mens-basketball/stats/2026-27/pepperdine/boxscore/12896) |
| 11-19 | Southern Miss at Grambling | 21 | — | Recovered Colin Byrd’s short stint from school substitutions. [Source](https://gsutigers.com/sports/mens-basketball/stats/2026-27/university-of-southern-mississippi/boxscore/7308) |
| 11-20 | Campbell at Kansas State | — | 138 | Rebuilt the lineup from the official school play-by-play. [Source](https://www.kstatesports.com:443/sports/mens-basketball/stats/2026-27/campbell/boxscore/19323) |
| 11-20 | South Dakota State at Idaho State | — | 132 | Rebuilt the lineup from the official school play-by-play. [Source](https://isubengals.com/sports/mens-basketball/stats/2026-27/south-dakota-state-university/boxscore/7976) |
| 11-20 | North Texas at App State | — | 137 | Rebuilt the lineup from the official school play-by-play. [Source](https://appstatesports.com/sports/mens-basketball/stats/2026-27/north-texas/boxscore/10147) |
| 11-20 | Western Kentucky at Richmond | — | 138 | Rebuilt the lineup from the official school play-by-play. [Source](https://richmondspiders.com/sports/mens-basketball/stats/2026-27/western-kentucky/boxscore/12126) |
| 11-20 | South Alabama at SMU | 25 | 128 | Verified Keyne Costa’s name and jersey. [Source](https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/401929459) |
| 11-20 | Mississippi Valley State at Arkansas | — | 135 | Confirmed Ilia Frolov’s 3:29 in the official final stats. [Game book](https://arkansasrazorbacks.com/wp-content/uploads/2026/11/Gm-6-Stats-vs-Miss-Valley-State.pdf) |
| 11-21 | Western Carolina at Wake Forest | 27 | 142 | Matched the team-only basket at 7:00 of the second half to both running scores. [Source](https://catamountsports.com/sports/mens-basketball/stats/2026-27/wake-forest/boxscore/16117) |
| 11-21 | Army at Binghamton | — | 125 | Rebuilt the lineup from the official school play-by-play. [Source](https://binghamtonbearcats.com/sports/mens-basketball/stats/2026-27/army/boxscore/15838) |
| 11-21 | Radford at Chattanooga | — | 125 | Rebuilt the lineup from the official school play-by-play. [Source](https://gomocs.com/sports/mens-basketball/stats/2026-27/radford/boxscore/18557) |
| 11-21 | Queens University at LSU | — | 140 | Recognized bookkeeping turnovers attached to technical fouls. [Source](https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/401921811) |
| 11-21 | Cornell at Morgan State | 27 | 146 | Verified Trey Williams’s identity against Cornell’s roster. [Source](https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/401924722) |
| 11-21 | Southern Illinois at East Carolina | — | 145 | Confirmed the 1:50 appearance in East Carolina’s school box and replay. [Game book](https://ecupirates.com/sports/mens-basketball/stats/2026/southern-ill-/boxscore/14335) |
| 11-22 | Towson at American University | — | 138 | Rebuilt the lineup from the official school play-by-play. [Source](https://aueagles.com/sports/mens-basketball/stats/2026-27/towson/boxscore/6069) |
| 11-22 | Loyola Marymount at UC Irvine | 23 | — | Recovered Dom Bentho’s short stint from school substitutions. [Source](https://ucirvinesports.com/sports/mens-basketball/stats/2026-27/lmu/boxscore/14445) |
| 11-22 | Delaware at UCLA | — | 133 | Removed a possession attributed to an unanchored rebound. [Source](https://www.espn.com/mens-college-basketball/playbyplay/_/gameId/401920905) |
| 11-22 | Dayton at Northwestern | — | 119 | Kept the 6:35 Randall–Haynes swap and dropped its 5:27 repeat in the school replay. The official box includes McKie’s 1:14. [Final official report](https://daytonflyers.com/documents/2026/11/22/2924476.PDF) |

## RAPM still open

| Date | Matchup | Why the replay stayed open |
| --- | --- | --- |
| 11-14 | Northwestern State at Little Rock | The ESPN replay still has inconsistent substitutes at the start of the second half and at later stoppages after same-clock duplicate rows are removed. The school page did not provide a supported box with verified venue for the alternate replay. [School game book](https://nsudemons.com/sports/mens-basketball/stats/2026-27/little-rock/boxscore/14760) |
| 11-19 | UTEP at New Mexico State | The same Moeller–Criss change is logged at 14:43 and 8:39 of the second half. Criss commits a foul and Moeller shoots free throws between the two timestamps. The final game book has the same conflicting sequence. Removing the earlier swap alone produces minutes several minutes away from both players’ official totals. [School game book](https://utepminers.com/sports/mens-basketball/stats/2026-27/nm-state/boxscore/12558) |
| 11-22 | Southern Miss at Valparaiso | The Gaines-Wyatt–Cerese exchange appears twice. The official report credits Cerese with a basket and Gaines-Wyatt with a shot between those entries. Removing the earlier swap leaves conflicting minutes in the school replay and conflicting individual scoring in ESPN. [School game book](https://valpoathletics.com/sports/mens-basketball/stats/2026-27/southern-miss/boxscore/13223) |
