# NCAA Archetype Dashboard Quick Guide

## Data Snapshot

| Division | Players | Teams |
| --- | ---: | ---: |
| D-I | 2,971 | 365 |
| D-II | 2,627 | 289 |
| D-III | 3,550 | 397 |


## Usage
This model helps identify players who fit target playstyles, then move the most interesting names into deeper scouting.

## What The Map Shows

The map is a PCA view of player style. It compresses passing/creation, shooting quality, shooting volume, ball security, defensive rebounding, and size into two visual axes.

- **PC1:** size vs. creator guard traits; players farther right tend to be bigger and less guard-creation driven
- **PC2:** shooting and spacing strength; players higher on the map tend to have stronger shooting profiles
- **Dot color:** the player's archetype they are most like

Note: Players under **6'7"** can still show an F/C Stretch score in their profile, but they cannot be colored or classified as F/C Stretch on the map.

Color key: blue = PG / Combo Guard, green = 2-4 Interchangeable Wing, orange = 5 / Stretch 4 / Big Wing.

## Archetypes

- PG / Combo Guard

- 2-4 Interchangeable Wing

- 5 / Stretch 4 / Big Wing


## Player Profiles

Click a dot to open a profile. The profile shows:

- Bio: division, team, class, height, games, minutes
- Season statline: points, rebounds, assists, steals, blocks, FG%, 3P%
- League-average bars: player value vs. division average
- Archetype scores: fit scores for PG, Wing, and F/C Stretch
- Passed qualification parameters: which filter checkpoints the player passed
- Similar players: closest style matches using the selected similarity metric

## Filter Controls

**Most Similar Archetype** only shows the player's highest eligible archetype score. 

**Filter Mode** applies stricter qualification logic:

- **None:** show the full player pool
- **General Player:** baseline pass-shoot-dribble screen
- **PG / Combo Guard:** General Player baseline plus guard requirements
- **2-4 Interchangeable Wing:** General Player baseline plus wing requirements
- **5 / Stretch 4 / Big Wing:** General Player baseline plus F/C size and spacing requirements

**Minimum archetype score** filters by score. When a role-specific Filter Mode is active, it uses that role's qualified-pool score.

Other filters let you narrow by conference, team, position, class, minutes, production, shooting, rebounding, assists, defensive stats, and height.

## General Player

Standard requirements:

- AST% percentile 70+
- eFG% 50%+
- 3P% 30%+
- AST/TO percentile 50+
- DREB requirement: guards 10%+, non-guards 15%+ when true DREB% is available

Exception:

- AST% percentile 85+
- DREB percentile 85+
- AST/TO percentile 50+

### PG / Combo Guard

- AST% percentile 70+
- AST/TO percentile 70+
- Standard shooting path: 3P% 33+ and 3P rate 30+
- Exception path: 2P% percentile 70+

Emphasis: AST%, AST/TO, 3P profile, 2P%.

### 2-4 Interchangeable Wing


- DREB percentile 70+
- 3P% 33+
- 3P rate 30+
- AST/TO percentile 50+

Emphasis: DREB, eFG, 3P profile, AST/TO.

### 5 / Stretch 4 / Big Wing

- Must be 6'7"+
- DREB percentile 70+
- 3P% 30+
- 3P rate 25+
- AST/TO percentile 50+

Emphasis: DREB first, then 3P profile, eFG, and AST/TO.

## Example: Tom Beattie, UC San Diego

Tom Beattie is a good example of how a player moves through the qualification system.

Baseline profile:

- Position: **G**
- Height: **6'4"**
- eFG%: **53.6%**
- 3P%: **32.3%**
- 3P rate: **45.2%**
- DRB%: **14.6**
- AST/TO: **2.32**

### Step 1: General Player

Tom passes the General Player baseline through the standard path:

- AST% percentile: **84.4** → passes 70+
- eFG%: **53.6%** → passes 50%+
- 3P%: **32.3%** → passes 30%+
- AST/TO percentile: **91.9** → passes 50+
- Guard DREB requirement: **14.6** → passes 10+

### Step 2: PG / Combo Guard

Tom qualifies as PG / Combo Guard through the 2P efficiency exception path:

- General Player baseline → passes
- AST% percentile: **84.4** → passes 70+
- AST/TO percentile: **91.9** → passes 70+
- 3P rate: **45.2%** → passes 30+
- 2P% percentile: **78.2** → passes the 70+ exception

He does not pass the standard PG shooting path because his 3P% is **32.3%**, just below the **33%** threshold.

### Why He Does Not Pass 2-4 Wing

Tom passes some wing checkpoints, but misses two required ones:

- General Player baseline → passes
- DREB percentile: **67.2** → below 70
- 3P%: **32.3%** → below 33%
- 3P rate: **45.2%** → passes 30+
- AST/TO percentile: **91.9** → passes 50+

### Why He Does Not Pass 5 / Stretch 4 / Big Wing

Tom has stretch-style markers, but he misses the hard size and position rules:

- General Player baseline → passes
- Position: **G** → not F/C
- Height: **6'4"** → below 6'7"
- DREB percentile: **67.2** → below 70
- 3P%: **32.3%** → passes 30+
- 3P rate: **45.2%** → passes 25+
- AST/TO percentile: **91.9** → passes 50+

So Tom can still show a useful 5 / Stretch 4 / Big Wing score in his profile, but he is classified as **PG / Combo Guard**, not F/C Stretch.

## Recommended Process

1. **Choose an archetype from Filter Mode.** Start with General Player for a broad screen, or choose PG / Combo Guard, 2-4 Interchangeable Wing, or 5 / Stretch 4 / Big Wing for a stricter qualified pool.
2. **Add specific filters.** Narrow the pool by year/class, team, conference, position, height, minutes, production, shooting, rebounding, assists, or defensive stats.
3. **Watchlist and compare.** Open player profiles, save interesting names to the watchlist, and use the radar chart or similar-player list to compare candidates before film review.

## Watchlist Usage

Use the watchlist to hold names for later comparison.

- Star a player from their profile.
- Open the Watchlist tab to see saved players.
- Use the radar chart to compare selected players by division percentile.
- Keep the watchlist as a working shortlist
