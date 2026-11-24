#!/bin/bash
# Samples memory every N seconds into a CSV while a training process lives,
# so a silent death (no traceback -- run4 on 2026-11-06, run5's first launch on
# 2026-11-23) can be checked against what memory was doing just before it.
#
# Read-only: it reads /proc and /sys and never signals, renices or changes
# oom_score_adj of anything (unlike oom_guard.sh).
#
# Columns:
#   time            UTC
#   mem_avail_mb    MemAvailable (/proc/meminfo) -- the number that hits 0 at OOM
#   cgroup_mb       this session cgroup's memory.current
#   train_rss_mb    the training python process
#   envserver_rss_mb / envserver_procs   every EnvServer.ts process (training
#                   and eval workers, tsx wrappers included)
#   tmpfs_mb        used on /tmp (tmpfs here, so it is RAM too)
#   oom_kill        the container cgroup's cumulative oom_kill counter -- a
#                   rise between two rows is an OOM kill in that window
#
# usage: ./mem_logger.sh <train_pid> <out.csv> [interval_seconds]
# Exits by itself one sample after <train_pid> is gone.

set -u
PID="$1"
OUT="$2"
INTERVAL="${3:-30}"

# The root file only: memory.events counts are hierarchical, so adding the
# child cgroups' would count each kill twice.
oom_kills() {
  awk '/^oom_kill / {print $2}' /sys/fs/cgroup/memory.events 2>/dev/null || echo NA
}

[ -s "$OUT" ] || echo "time,mem_avail_mb,cgroup_mb,train_rss_mb,envserver_rss_mb,envserver_procs,tmpfs_mb,oom_kill,train_alive" > "$OUT"
while true; do
  alive=0; [ -d "/proc/$PID" ] && alive=1
  avail=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
  cg=$(( $(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0) / 1048576 ))
  train=$(awk '/^VmRSS:/ {print int($2/1024)}' "/proc/$PID/status" 2>/dev/null || echo 0)
  read -r env_rss env_n < <(ps -eo rss=,args= | awk '/EnvServer\.ts/ && !/awk/ {s += $1; n++} END {print int(s/1024), n + 0}')
  tmpfs=$(df -m --output=used /tmp | tail -1 | tr -d ' ')
  echo "$(date -u +%FT%TZ),$avail,$cg,${train:-0},$env_rss,$env_n,$tmpfs,$(oom_kills),$alive" >> "$OUT"
  [ "$alive" = 1 ] || exit 0
  sleep "$INTERVAL"
done
