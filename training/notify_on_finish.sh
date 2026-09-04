#!/bin/bash
# Watches a training run's PID and posts to ntfy.sh when it exits, whether
# that's a clean finish or a crash. Fully detached (launched with nohup +
# disown) so it survives the Claude Code session / VS Code closing --
# depends only on this machine staying on and networked.
#
# Usage: ./notify_on_finish.sh <pid> <log_file> <topic>

PID="$1"
LOG="$2"
TOPIC="$3"

while kill -0 "$PID" 2>/dev/null; do
  sleep 20
done

LAST_LINES=$(tail -5 "$LOG")
if echo "$LAST_LINES" | grep -qiE "error|traceback"; then
  STATUS="stopped -- looks like a crash"
else
  # Training runs indefinitely (--updates 0) -- there is no "finished"
  # state to detect anymore, so any exit that isn't an obvious crash is
  # reported as unexpected (a clean stop should come from an intentional
  # kill, which this watcher isn't told about and can't distinguish here).
  STATUS="process exited unexpectedly (check log for details)"
fi

curl -s -m 15 -d "OpenFront RL training run: $STATUS. PID $PID." "https://ntfy.sh/$TOPIC" >/dev/null
