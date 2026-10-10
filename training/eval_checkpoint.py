"""Evaluate a checkpoint (or the scripted baseline) on fixed seeds, per difficulty.

Runs train.py's own run_eval_suite() -- same seeds, same sampled policy, same
construction as the in-training eval suites -- against each requested
difficulty, and writes one row per game. Seeds are the stage's eval seeds
(eval-suite-<difficulty>-<maps>-k), so results line up game for game with a
run's eval_episodes.csv at the same stage.

    python eval_checkpoint.py --checkpoint checkpoints/latest.pt --out results.csv
    python eval_checkpoint.py --scripted --out scripted.csv

--scripted plays a 10-line troop-threshold policy instead of a network: noop
until troops >= 45% of maxTroops, then expand > attack_opponent > boat_attack
(random legal cell) by legality. Not learned -- a reference bar on the same
action space. Set the OPENFRONT_* env-bridge flags the run trained with.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from envs.openfront_env import ACTIONS, DIFFICULTIES, MAP_POOLS, REWARD_MODES, requested_env_flags
from models.network import ActorCritic
from train import run_eval_suite, summarize_suite

NOOP, EXPAND, ATTACK, BOAT = (ACTIONS.index(a) for a in ("noop", "expand", "attack_opponent", "boat_attack"))
TROOPS_RATIO_IDX = 2  # SELF_FEATURES = (tiles, troops, troops_ratio, gold)


class ScriptedPolicy:
    """Duck-types ActorCritic.act() for run_eval_suite()."""

    def __init__(self, trigger: float = 0.45):
        self.trigger = trigger

    def act(self, tile_grid, self_features, opponent_features, opponent_mask,
            tile_target_mask, attack_target_mask, action_mask):
        b = action_mask.shape[0]
        out = torch.zeros(b, 3, dtype=torch.long)
        for i in range(b):
            legal = action_mask[i]
            if self_features[i, TROOPS_RATIO_IDX] < self.trigger:
                a = NOOP
            else:
                a = next((x for x in (EXPAND, ATTACK, BOAT) if legal[x]), NOOP)
            out[i, 0] = a
            if a == BOAT:
                cells = tile_target_mask[i].reshape(-1).nonzero().flatten()
                out[i, 1] = cells[torch.randint(len(cells), (1,))].item()
        return out, None, None


def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint")
    src.add_argument("--scripted", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--difficulties", default=",".join(DIFFICULTIES))
    p.add_argument("--maps", choices=sorted(MAP_POOLS), default="small")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--replays", type=int, default=0, help="dump GameRecords for seeds < this")
    p.add_argument("--replay-dir", default=str(Path(__file__).parent / "replays" / "eval_checkpoint"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    device = torch.device(a.device)
    if a.scripted:
        policy, label = ScriptedPolicy(), "scripted-threshold-0.45"
    else:
        ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
        head = "big" if any(k.startswith("tile_head_big.") for k in ckpt["model"]) else "small"
        policy = ActorCritic(tile_head=head).to(device)
        policy.load_state_dict(ckpt["model"])
        policy.eval()
        label = f"{a.checkpoint}@{ckpt['update']}"
    print(f"{label}  env flags: {requested_env_flags()}")

    args = argparse.Namespace(
        map=None, ticks_per_step=10, max_episode_steps=a.max_steps, reward_mode=REWARD_MODES[0],
        potential_beta=0.0, eval_episodes=a.episodes, eval_envs=a.envs, eval_max_steps=a.max_steps,
        eval_replays=a.replays, eval_dir=str(Path(a.replay_dir).resolve()),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["policy", "difficulty", "maps", "seed", "map", "winner", "ticks",
                                           "truncated", "peak_territory_frac", "final_territory_frac"])
        w.writeheader()
        for difficulty in a.difficulties.split(","):
            t0 = time.time()
            episodes = run_eval_suite(args, policy, device, difficulty, a.maps)
            for e in episodes:
                w.writerow({"policy": label, "difficulty": difficulty, "maps": a.maps, **e})
            fh.flush()
            s = summarize_suite(episodes)
            print(f"{difficulty:10s} won {s['wins']}/{s['episodes']} = {s['win_rate']:.3f} "
                  f"[{s['ci_low']:.3f}-{s['ci_high']:.3f}] mean_peak={s['mean_peak_territory']:.3f} "
                  f"truncated={s['truncated']} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
