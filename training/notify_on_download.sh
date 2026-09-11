#!/bin/bash
# Pushes an ntfy alert when a long-running background download finishes.
#
# Written for `kaggle kernels output`, which pulls the whole /kaggle/working
# tree (~750MB+, dominated by the cloned node_modules) and routinely runs
# past any reasonable foreground timeout.
#
# Takes a PID, not a pgrep pattern. The first version of this script matched
# on `pgrep -f "kernels output"` and would have hung forever: the watcher's
# OWN command line contains that pattern, so it always saw at least one
# match and never fired. A PID is unambiguous.
#
# Unlike notify_on_finish.sh this checks curl's exit code and retries. A
# silent DNS failure already cost one missed alert on this box (Tailscale
# MagicDNS dropping its upstream resolvers), and a notification that can
# vanish that way is worse than useless -- you trust it and it lies.
#
# Usage: ./notify_on_download.sh <pid> <watch_dir> <topic>

PID="${1:?usage: notify_on_download.sh <pid> <watch_dir> <topic>}"
WATCH_DIR="${2:?}"
TOPIC="${3:?}"

while kill -0 "$PID" 2>/dev/null; do
  sleep 20
done

SIZE="$(du -sh "$WATCH_DIR" 2>/dev/null | cut -f1)"
REPLAYS="$(find "$WATCH_DIR" -name '*.json' -path '*replays*' 2>/dev/null | wc -l)"
HAS_OUT="no"
[ -d "$WATCH_DIR/out" ] && HAS_OUT="yes"

MSG="Kaggle download finished: ${SIZE} at ${WATCH_DIR}. replays=${REPLAYS}, out/=${HAS_OUT}"

for attempt in 1 2 3; do
  if curl -s -m 20 -d "$MSG" "https://ntfy.sh/$TOPIC" >/dev/null 2>&1; then
    echo "$(date -u +%H:%M:%S) notified: $MSG"
    exit 0
  fi
  echo "$(date -u +%H:%M:%S) ntfy attempt $attempt failed (DNS/network?), retrying..."
  sleep 15
done

echo "$(date -u +%H:%M:%S) ntfy FAILED after 3 attempts. Message was: $MSG"
exit 1
