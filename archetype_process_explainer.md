# Archetype Beta Process Explainer

This document explains the archetype dashboard, what the PCA map is showing, and how the results should be used. The model is not final. It is a screening tool that can be tuned as the player pools are reviewed.

## Purpose

The goal is to turn preferred player profiles into a searchable, visual tool.

The current focus is on four filter profiles:

1. **General Player**
2. **PG / Combo Guard**
3. **2-4 Interchangeable Wing**
4. **5 / Stretch 4 / Big Wing**

Each player is evaluated against passing and creation, shooting quality, shooting volume, ball security, defensive rebounding, and size.

This is a **beta** model. It is designed to help find players for review, not to make final decisions by itself.

## Data Used

The website uses three datasets:

- **Division I:** 2,971 players
- **Division II:** 2,627 players
- **Division III:** 3,550 players

All qualification flags, archetype scores, and archetype PCA coordinates are created in memory when the app runs. The source CSV files are not modified by the dashboard.

## Map And Axes

The archetype map uses PCA to summarize several basketball traits into a two-dimensional view.

The map uses:

- **X-axis:** PC1, size vs. creator guard traits
- **Y-axis:** PC2, shooting and spacing strength
- **Color:** the player's highest displayed archetype score

Higher PC1 generally points toward more size and frontcourt profile. Lower PC1 generally points toward more guard creation, assist profile, and ball handling profile.

Higher PC2 generally points toward stronger shooting, spacing, and overall scoring efficiency. Lower PC2 generally points toward players whose value comes less from shooting profile and more from creation, rebounding, or other traits.

PCA axes are not fixed basketball truths. They are based on patterns in the current dataset, so they can shift if the data or feature set changes.

## How The System Works

The dashboard has two layers:

1. **Qualification filters**
2. **Weighted archetype scores**

The default dashboard keeps every player visible. A player is not removed unless a **Filter Mode** is selected.

The **Most Similar Archetype** filter only narrows by the player's highest displayed archetype score. It does not decide whether a player is qualified.

The **Filter Mode** dropdown applies the qualification rules. Role-specific filters are intentionally stricter than the General Player filter: a player must first qualify as a General Player before qualifying under PG, Wing, or F/C Stretch mode.

## General Player

The General Player profile is the baseline pass-shoot-dribble screen.

Standard path:

- AST% percentile **70+**
- eFG% **50%+**
- 3P% **30%+**
- AST/TO percentile **50+**
- DREB requirement: guards **10%+**, non-guards **15%+** when true DREB% is available

Exception path:

- AST% percentile **85+**
- DREB percentile **85+**
- AST/TO percentile **50+**

The exception path allows unusual creation-plus-rebounding players to stay in the pool even if the shooting profile is not standard.

## PG / Combo Guard

This profile is looking for high-level guard creation.

Standard path:

- General Player qualification
- AST% percentile **70+**
- AST/TO percentile **70+**
- 3P% **33%+**
- 3P rate **30%+**

Exception path:

- General Player qualification
- AST% percentile **70+**
- AST/TO percentile **70+**
- 2P% percentile **70+**

Score weights:

- AST% percentile: **30%**
- AST/TO percentile: **30%**
- 3P profile: **20%**
- 2P% percentile: **20%**

## 2-4 Interchangeable Wing

This profile is looking for perimeter and forward players who can rebound, space, and keep the ball moving.

Standard path:

- General Player qualification
- DREB percentile **70+**
- 3P% **33%+**
- 3P rate **30%+**
- AST/TO percentile **50+**

Score weights:

- DREB percentile: **30%**
- eFG percentile: **25%**
- 3P profile: **25%**
- AST/TO percentile: **20%**

## 5 / Stretch 4 / Big Wing

This profile is looking for F/C classified players with size, defensive rebounding, and enough shooting value to stretch the floor.

Standard path:

- General Player qualification
- Position classification **F/C**
- Height **6'7"+**
- DREB percentile **70+**
- 3P% **30%+**
- 3P rate **25%+**
- AST/TO percentile **50+**

Score weights:

- DREB percentile: **40%**
- 3P profile: **20%**
- eFG percentile: **20%**
- AST/TO percentile: **20%**

Height and F/C classification are mandatory qualification rules for this profile, but they are not part of the weighted score.

Primary map classification also uses the height gate. A player can still show a strong F/C Stretch score in the profile, but if he is under **6'7"**, he will not be colored or labeled as F/C Stretch on the map.

Example:

- **Tom Beattie, UC San Diego**
- Height: **6'4"**
- 3P%: **32.3%**
- 3P rate: **45.2%**
- DRB%: **14.6**
- AST/TO: **2.32**

Tom's stat line shows why he can appear in the stretch-style conversation: credible 3P percentage, real 3P volume, useful defensive rebounding context, and strong AST/TO. Those markers can push his stretch score upward.

His qualification flow is:

- **General Player:** qualifies through the standard path. He clears AST% percentile, eFG%, 3P%, AST/TO percentile, and the guard DREB requirement.
- **PG / Combo Guard:** qualifies through the 2P efficiency exception path. He clears the General Player baseline, AST% percentile, AST/TO percentile, 3P rate, and 2P% percentile exception.
- **2-4 Interchangeable Wing:** does not qualify because his DREB percentile and 3P% are below the wing thresholds.
- **5 / Stretch 4 / Big Wing:** does not qualify because he is classified as a guard, is 6'4", and is below the DREB percentile threshold.

Tom can still display a 5 / Stretch 4 / Big Wing score, but he is classified as **PG / Combo Guard**, not F/C Stretch.

## Percentiles And Raw Thresholds

Percentiles are used for context-dependent stats:

- AST%
- AST/TO
- DREB%
- 2P%

Raw basketball thresholds are used for benchmarks that are easier to interpret directly:

- eFG%
- 3P%
- 3P rate
- height

Raw shooting thresholds are used because basketball decisions often start from benchmarks like **30%** or **33%** from three. Percentiles are used where role, division, and stat environment matter more.

## Scores

Each player receives three displayed archetype scores:

- `PG / Combo Guard`
- `2-4 Interchangeable Wing`
- `5 / Stretch 4 / Big Wing`

The highest displayed score becomes the player's default archetype color on the map.

When a role-specific Filter Mode is selected, the minimum archetype score slider uses the qualified-pool score for that role. That means a PG filter ranks a player against qualified PG / combo guard candidates, not every player in the dataset.

Archetype scores are fit-within-role scores. They should not be treated as perfectly comparable across all archetypes.

The current weights are basketball-informed starting points. They should be adjusted as the player pools are reviewed and compared against film.

## Recommended Workflow

1. **Choose an archetype from Filter Mode.** Start with General Player for a broad screen, or choose PG / Combo Guard, 2-4 Interchangeable Wing, or 5 / Stretch 4 / Big Wing for a stricter qualified pool.
2. **Add specific filters.** Narrow the pool by year/class, team, conference, position, height, minutes, production, shooting, rebounding, assists, or defensive stats.
3. **Watchlist and compare.** Open player profiles, save interesting names to the watchlist, and use the radar chart or similar-player list to compare candidates before film review.

## Caveats

D-II and D-III do not have every exact rate stat available, so some values use proxies.

This matters most for:

- assist creation
- defensive rebounding

The dashboard should help identify names worth watching, discussing, and validating. It should not replace film, role context, health, eligibility, or direct evaluation.
