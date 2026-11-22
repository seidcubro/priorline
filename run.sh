#!/bin/sh
# Run one lab script on the server against the production database.
#   sh lab/run.sh research_count_rank.py
# The training image holds the platform's own modules in /app. The lab
# folder is mounted beside it, so a lab script can import both.
#
# It runs in the background and writes to a log, because a run of twenty
# minutes should not die with the terminal that started it.
set -e
cd /opt/priorline
set -a; . /etc/priorline/env; set +a
name=$(basename "$1" .py)
log="/var/log/priorline/lab-$name.log"
docker compose -f deploy/docker-compose.prod.yml build -q training
nohup docker compose -f deploy/docker-compose.prod.yml run --rm -T \
  -v /opt/priorline/lab/training:/lab -e PYTHONPATH=/app \
  ${MARKETS:+-e MARKETS=$MARKETS} ${SEASONS:+-e SEASONS=$SEASONS} \
  training python -u "/lab/$1" > "$log" 2>&1 &
echo "started. Watch it with:  tail -f $log"
