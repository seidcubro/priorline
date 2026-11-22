# Database

`init.sql` creates the base tables on a fresh database. Everything after that is
a migration.

| folder | what is in it |
|---|---|
| `migrations/` | every schema change since, applied in order by `deploy/apply_migrations.sh` |
| `views/` | the materialized views the feature builder reads, and their refresh |
| `backfills/` | the two repairs the scheduled refresh runs every time: fill the team on new feature rows, and prune props for games already played |

If a script becomes permanently required, it belongs in a migration or in the
application, not here.
