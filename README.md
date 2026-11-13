# ncaa-player-dashboard
Player Dashboard for NCAA D1, D2 &amp; D3 Players

### Daily men's RAPM

The Daily RAPM workflow runs at 7:30 a.m. Pacific from the default branch.
It refits through the previous Pacific calendar day with validated NCAA
possession lineups and the bundled historical corrections. When a box-score
matchup is missing from those inputs, the runner looks for the ESPN game by
team, date, and final score, then replays and validates its starters and
substitutions. Unresolved matchups are listed in the snapshot.

Download the `rapm-daily` artifact from a successful Actions run for the
rating snapshot and the possessions included in the fit. Each run produces
its own artifact. For a manual check, dispatch the workflow or run:

```sh
python -m pip install -r ratings/requirements.txt
python ratings/daily_rapm.py --output rapm-ratings.json --possessions-output rapm-possessions.json
```
