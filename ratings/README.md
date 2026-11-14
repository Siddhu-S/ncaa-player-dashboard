# Daily BPM and RAPM

The data job runs at 7:30 a.m. Pacific on the default branch. A pull request
runs a calculation without a database connection. Production runs require
the Actions secret `RATINGS_DATABASE_URL`, a TLS-enabled Supabase session
pooler URI for an approved `ratings_writer` login. Provision its privileges
with [ratings_writer_grants.sql](sql/ratings_writer_grants.sql) after the
[schema migration](sql/ratings_schema.sql) is applied. The grants script
assumes the login already exists.

The first run discovers completed Division I games from the NCAA schedule
and ESPN scoreboards. Subsequent runs check yesterday, any missed calendar
days since the last published run, and dates with due retries. The job
validates SportsDataverse and ESPN box scores separately, compares the
fields BPM uses, and logs disagreements for review. RAPM can fall back
to replaying ESPN starters and substitutions when the NCAA lineup feed
has no valid game. Historical verified corrections ship with the code.

`ratings.game_sources` identifies the evidence attached to each game.
`ratings.game_metric_status` tracks each metric, its next retry, and its last
accepted input revision. A failed retry preserves that revision.
`ratings.open_review_queue` lists open issues with evidence, source URLs,
and affected player or possession details when available. A reviewer can
record a resolution and timestamp in `ratings.review_issues`. To reprocess a
reviewed game, set its metric's `next_retry_at` to `now()`. The pipeline can
close an issue automatically after a successful recheck.

The model reads accepted revisions from Supabase and publishes the snapshot
and player rows together in one transaction. The snapshot includes the
input revisions, coverage counts, model version, and each metric's cutoff.

```sql
select game_date, game_key, metric, evidence
from ratings.open_review_queue
order by game_date desc, game_key, metric;

select through_date, coverage
from ratings.snapshots
order by through_date desc
limit 1;
```

## Dashboard CSV

After publishing the Supabase snapshot, the daily workflow exports
`player-ratings.csv` to the `ratings-data` branch. The GitHub Pages Ratings
tab fetches the CSV when opened, so daily updates do not require a site rebuild.
The CSV contains the player ratings, their sample sizes, and snapshot coverage.
Missing ratings are blank. The CSV is accessible from the public GitHub repository.
