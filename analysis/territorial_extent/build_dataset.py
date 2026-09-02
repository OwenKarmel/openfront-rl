"""Merges FindMaxTerritory.ts's per-replay results with an approximate
training-update number for each eval, then writes the combined dataset as
a CSV for plot_territorial_extent.py to read.

Update-number recovery: replay filenames only encode a wall-clock timestamp
and a gameID hash (see EnvServer.ts's flushRecord()) -- the update number
that produced a given eval episode isn't recoverable from the filename
alone. train_stdout.log's lines ARE strictly chronological, though, and
every "eval vs <difficulty>: {...}" line is immediately preceded (possibly
several lines earlier, across intervening "update N ..." lines) by the
"update N ..." line for whatever update triggered that eval -- so walking
the log in order and remembering the last "update N" seen recovers the
update number for each eval in order. Pairing that ordered list with the
chronologically-sorted replay files (by filename, which sort correctly
since the timestamp prefix is zero-padded ISO-like) by POSITION is the best
available correlation -- not exact (counts were off by one in practice,
likely a training restart's checkpoint/log boundary), but close enough to
see the real trend, and documented here rather than silently assumed.
"""
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STDOUT_LOG = REPO_ROOT / "training" / "checkpoints" / "train_stdout.log"
REPLAYS_DIR = REPO_ROOT / "training" / "replays"
RESULTS_LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/territory_results.log")
OUT_CSV = Path(__file__).resolve().parent / "territorial_extent.csv"

UPDATE_RE = re.compile(r"^update\s+(\d+)\s")
EVAL_RE = re.compile(r"^\s*eval vs \w+: (\{.*\})\s*$")


def extract_eval_sequence() -> list[dict]:
    """Returns, in chronological order, one dict per eval line encountered:
    {"update": int, "winner": ..., "ticks": ..., "self_tiles": ..., "opp_tiles": ...}."""
    evals = []
    current_update = None
    with open(STDOUT_LOG) as f:
        for line in f:
            m = UPDATE_RE.match(line)
            if m:
                current_update = int(m.group(1))
                continue
            m = EVAL_RE.match(line)
            if m:
                # dict literal uses Python repr (single quotes, None) -- eval() is safe here,
                # this is our own trusted log, not external input.
                d = eval(m.group(1), {"__builtins__": {}}, {})
                evals.append({"update": current_update, **d})
    return evals


def main() -> None:
    eval_seq = extract_eval_sequence()
    files = sorted(p.name for p in REPLAYS_DIR.glob("*.json"))

    results_by_file = {}
    with open(RESULTS_LOG) as f:
        for line in f:
            parts = line.strip().split(" ")
            if parts[0] != "RESULT":
                continue
            _, peak, final, ticks, status, fname = parts
            results_by_file[fname] = {
                "peak_tiles": int(peak),
                "final_tiles": int(final),
                "ticks": int(ticks),
                "status": status,
            }

    print(f"eval log entries: {len(eval_seq)}, replay files: {len(files)}, results: {len(results_by_file)}")
    n = min(len(eval_seq), len(files))
    if len(eval_seq) != len(files):
        print(
            f"NOTE: counts differ (eval log entries={len(eval_seq)} vs files={len(files)}) -- "
            f"pairing the first {n} by chronological order; the tail is dropped rather than misaligned."
        )

    rows = []
    for i in range(n):
        fname = files[i]
        r = results_by_file.get(fname)
        if r is None:
            continue
        rows.append(
            {
                "eval_index": i + 1,
                "approx_update": eval_seq[i]["update"],
                "file": fname,
                "peak_tiles": r["peak_tiles"],
                "final_tiles": r["final_tiles"],
                "ticks": r["ticks"],
                "status": r["status"],
                "logged_self_tiles": eval_seq[i].get("self_tiles"),
                "logged_opp_tiles": eval_seq[i].get("opp_tiles"),
                "logged_winner": eval_seq[i].get("winner"),
            }
        )

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
