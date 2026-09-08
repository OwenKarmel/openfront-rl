#!/bin/bash
# Keeps the long-running training process from being chosen by the kernel's
# global OOM killer.
#
# Why it works this way round: protecting a process directly means a NEGATIVE
# oom_score_adj, which needs CAP_SYS_RESOURCE (root). This box has neither
# root nor passwordless sudo. Raising oom_score_adj on your OWN processes is
# unprivileged, so instead of protecting training we make everything else a
# more attractive victim. Net effect is the same ordering.
#
# The asymmetry that justifies it: VS Code / language servers / the dev game
# server all restart automatically and lose nothing. A killed training run
# loses every update since the last checkpoint and needs a manual restart.
#
# Motivating incident (2026-09-08 02:02 UTC): training was OOM-killed at
# update 437 with 4.5GB still free at the previous sample -- 19 vscode-server
# processes totalling 2.3GB spawned within the same ~60s window (triggered by
# opening files in the IDE) and the kernel picked the single largest RSS on
# the box, which was training. No leak was involved; worker memory was flat
# at -3.3 MB/hour.
#
# Runs as a loop because VS Code spawns new processes constantly -- a one-shot
# pass would not cover extension hosts started after it ran.
#
# Usage: ./oom_guard.sh <training_pid> [interval_seconds]

TRAIN_PID="$1"
INTERVAL="${2:-30}"

if [ -z "$TRAIN_PID" ]; then
  echo "usage: $0 <training_pid> [interval_seconds]" >&2
  exit 1
fi

# oom_score_adj is in [-1000, 1000]; the kernel adds (adj/1000)*total_ram to a
# process's badness score, so 800 on a 15GB box is worth ~12GB of phantom RSS
# -- more than enough to outrank training's ~2-3GB no matter how it grows.
ADJ_EXPENDABLE=800   # extension hosts, language servers: restart silently
ADJ_SERVER=600       # dev game server, vite, ReplayServer: restart on demand
ADJ_IDE_CORE=400     # vscode core server: killing it disconnects the IDE
                     # (recoverable, but noticeable) so it is ranked below the
                     # things that recover invisibly

set_adj() { # pattern, value
  for pid in $(pgrep -f "$1" 2>/dev/null); do
    [ "$pid" = "$TRAIN_PID" ] && continue
    echo "$2" > "/proc/$pid/oom_score_adj" 2>/dev/null
  done
}

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  set_adj "vscode-server/extensions" "$ADJ_EXPENDABLE"
  set_adj "language-features|languageserver|tsserver|jsonServerMain|htmlServerMain|markdown-language" "$ADJ_EXPENDABLE"
  set_adj "claude-code.*resources/native" "$ADJ_EXPENDABLE"
  set_adj "src/server/Server.ts|node_modules/.bin/vite" "$ADJ_SERVER"
  set_adj "ReplayServer.ts" "$ADJ_SERVER"
  set_adj "vscode-server/cli/servers" "$ADJ_IDE_CORE"

  # Training itself stays at the default 0 -- the best we can do unprivileged,
  # and now the lowest score among the heavyweight processes on the box.
  echo 0 > "/proc/$TRAIN_PID/oom_score_adj" 2>/dev/null

  sleep "$INTERVAL"
done
