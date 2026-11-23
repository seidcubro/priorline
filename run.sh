#!/bin/sh
# Run one lab script on the server and watch it to the end.
#
#   sh lab/run.sh research_who_sorts.py
#   MARKETS=recs,rec_yds sh lab/run.sh research_reproject_blanks.py
#
# The output prints here as it comes and the command returns when the script
# is finished, with a line that says so. Nothing to tail and nothing to guess.
#
# The script itself runs detached, so a dropped connection does not kill it.
# If that happens, the whole output is in the log named on the first line, and
# its last line is === FINISHED === once it is done.
set -e
cd /opt/priorline
set -a; . /etc/priorline/env; set +a
name=$(basename "$1" .py)
log="/var/log/priorline/lab-$name.log"
export LAB_FILE="$1" LAB_LOG="$log"
docker compose -f deploy/docker-compose.prod.yml build -q training
: > "$log"
echo "running $1 (log: $log)"
nohup sh -c 'docker compose -f deploy/docker-compose.prod.yml run --rm -T \
  -v /opt/priorline/lab/training:/lab -e PYTHONPATH=/app \
  ${MARKETS:+-e MARKETS=$MARKETS} ${SEASONS:+-e SEASONS=$SEASONS} \
  training python -u "/lab/$LAB_FILE" >> "$LAB_LOG" 2>&1
  echo "=== FINISHED ===" >> "$LAB_LOG"' > /dev/null 2>&1 &
pid=$!
# Follows the log and stops by itself when the run does.
tail --pid="$pid" -n +1 -f "$log" | grep -v -e '^ Container ' -e 'dropping [0-9]* ' -e 'RuntimeWarning' -e 'return np.where' || true
echo "=== DONE: $1 has finished. Everything above is the whole output. ==="
