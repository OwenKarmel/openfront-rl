"""Extracts one row per eval episode from the live run's stdout log and
writes territory_moving_average.csv next to this file.

Each `eval vs <difficulty>: {...}` line is one greedy/argmax evaluation
episode, in chronological order, so the row index IS the evaluation-run
number. `peak_territory_frac` is AGENT's share of the map's LAND tiles at
its high-water mark during that episode (see territory_fraction() in
train.py -- read off tile_grid's own class proportions, so it stays valid on
the resized map_pool canvas).

Reads nohup_launch.log rather than train_stdout.log: this run is launched by
training/run_bbf_gamma098.sh under nohup, which redirects train.py's stdout
there. The earlier runs' train_stdout.log does not exist for this run.

Keeps the gamma0995 version's approach of reading the update number from the
most recent `update N` line preceding each eval, rather than computing it as
(index+1)*eval_every. This run has had no crash/resume so far (the `segment`
column is all zeros), but the robust form costs nothing and stays correct if
one happens later.
"""

import ast
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = Path("/home/developer/openfront-rl/training/checkpoints/nohup_launch.log")
OUT = HERE / "territory_moving_average.csv"

GAMMA = 0.98
MAX_EPISODE_STEPS = 20000


def main() -> None:
    update_re = re.compile(r"^update\s+(\d+)\s")
    eval_re = re.compile(r"eval vs (\w+): (\{.*\})")
    resume_re = re.compile(r"resumed from .* at update (\d+)")

    rows = []
    current_update = 0
    resumes = []
    for line in LOG.read_text().splitlines():
        mu = update_re.match(line)
        if mu:
            current_update = int(mu.group(1))
            continue
        mr = resume_re.search(line)
        if mr:
            resumes.append((len(rows), int(mr.group(1))))
            continue
        me = eval_re.search(line)
        if not me:
            continue
        try:
            d = ast.literal_eval(me.group(2))
        except (ValueError, SyntaxError):
            continue
        rows.append((current_update, me.group(1), d))

    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "eval_index", "update", "segment", "gamma", "max_episode_steps", "difficulty",
            "peak_territory_frac", "self_tiles", "opp_tiles", "ticks", "winner",
        ])
        resume_at = {idx for idx, _ in resumes}
        segment = 0
        for i, (update, difficulty, d) in enumerate(rows):
            if i in resume_at:
                segment += 1
            w.writerow([
                i, update, segment, GAMMA, MAX_EPISODE_STEPS, difficulty,
                d["peak_territory_frac"], d["self_tiles"], d["opp_tiles"],
                d["ticks"], d["winner"] if d["winner"] is not None else "",
            ])

    print(f"wrote {len(rows)} eval rows -> {OUT}")
    for eval_idx, update in resumes:
        print(f"  note: run resumed at update {update} (after eval #{eval_idx - 1})")


if __name__ == "__main__":
    main()
