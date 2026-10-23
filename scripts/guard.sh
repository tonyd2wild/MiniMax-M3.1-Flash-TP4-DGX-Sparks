#!/bin/bash
# guard.sh [MIN_MB]  (runs ON a Spark, nohup setsid): docker kill m31-tp4 when MemAvailable < MIN_MB (default 3072).
# Logs every 10 s plus the running minimum to /var/tmp/m31flash/logs/guard_mem.log, kills to guard.log.
# Exits when /var/tmp/m31flash/guard.stop is newer than this guard's start marker (touch it to stop; nothing is deleted).
MIN=${1:-3072}; L=/var/tmp/m31flash/logs; mkdir -p $L
exec 9>/var/tmp/m31flash/guard.lock; flock -n 9 || exit 0   # one guard per node
touch $L/guard.started; low=999999; n=0
echo "$(date '+%F %T') guard start min=${MIN}MB pid=$$" >> $L/guard.log
while ! [ /var/tmp/m31flash/guard.stop -nt $L/guard.started ]; do
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  up=$(docker ps -q -f name=^m31-tp4$)
  if [ -n "$up" ]; then
    [ "$a" -lt "$low" ] && low=$a
    if [ "$a" -lt "$MIN" ]; then
      docker kill m31-tp4 >/dev/null 2>&1
      echo "$(date '+%F %T') KILLED m31-tp4 avail_mb=$a" >> $L/guard.log
    fi
  fi
  n=$((n+1)); [ $((n % 10)) = 0 ] && echo "$(date +%T) avail_mb=$a low_mb=$low up=${up:+1}" >> $L/guard_mem.log
  sleep 1
done
echo "$(date '+%F %T') guard exit low_mb=$low" >> $L/guard.log
