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
fields BPM uses, and logs disagreements for review. ESPN roster aliases
must match uniquely by team, jersey, surname, and first initial when a
full-name match fails. Previously verified name variants also require the
audited ESPN and NCAA jerseys. When reported minutes conflict or a box
fails validation, the job can use playing time reconstructed from ESPN
substitutions after checking every period, lineup, and player point total.
If both boxes provide counting stats, those stats must agree for the
replay to resolve the game. When an audited school box provides a
third value for every disputed field, the job uses the value supported
by two sources, validates the resulting box, and records the votes.
Unresolved votes stay in review. RAPM can fall back
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
