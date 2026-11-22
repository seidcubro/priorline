#!/bin/sh
# Run one lab script on the server against the production database.
#   sh lab/run.sh research_count_rank.py
# The training image holds the platform's own modules in /app. The lab
# folder is mounted beside it, so a lab script can import both.
set -e
cd /opt/priorline
set -a; . /etc/priorline/env; set +a
docker compose -f deploy/docker-compose.prod.yml run --rm \
  -v /opt/priorline/lab/training:/lab -e PYTHONPATH=/app \
  training python "/lab/$1"
