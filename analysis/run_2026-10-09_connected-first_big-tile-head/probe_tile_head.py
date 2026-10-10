"""How random is a checkpoint's boat_attack targeting? Probe on real games.

Plays the checkpoint's sampled policy for --steps decision steps on each map
(fixed seed, medium opponent) and, at every step where boat_attack is legal
with more than one legal cell, measures the boat-target head's entropy as a
fraction of its maximum, log(#legal cells): 1.0 is uniformly random, lower
is more targeted. Also counts how many sampled boat_attacks visibly launched
(troops dropped >10% that step) -- a pick the engine cannot resolve is a
silent no-op.

Reference numbers (final checkpoints, 500 steps, seed 1):
  big head, update 1435:     caucasus 0.92  tourney4 0.94  fourislands 0.96;
                             tourney4 boats 112 sampled, 68 launched
  default head, update 1650: caucasus 0.99  tourney4 0.99  fourislands 0.98;
                             tourney4 boats 92 sampled, 3 launched
Sampling is unseeded, so expect a few hundredths of drift run to run.

usage (from the repo root, with the env-bridge set up):
  OPENFRONT_EXPAND_MASK=1 OPENFRONT_BOAT_MASK=1 OPENFRONT_NEUTRAL_BOATS=1 \\
    python analysis/run_2026-10-09_connected-first_big-tile-head/probe_tile_head.py <checkpoint.pt>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

TRAINING = Path(__file__).resolve().parents[2] / "training"
sys.path.insert(0, str(TRAINING))

from envs.openfront_env import ACTIONS, OpenFrontEnv  # noqa: E402
from models.network import ActorCritic  # noqa: E402
from train import net_inputs, obs_to_batch  # noqa: E402

BOAT = ACTIONS.index("boat_attack")
TROOPS = 1  # SELF_FEATURES index


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--maps", default="caucasus,tourney4,fourislands")
    p.add_argument("--steps", type=int, default=500)
    a = p.parse_args()

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    head = "big" if any(k.startswith("tile_head_big.") for k in ckpt["model"]) else "small"
    net = ActorCritic(tile_head=head)
    net.load_state_dict(ckpt["model"])
    net.eval()
    cpu = torch.device("cpu")

    for m in a.maps.split(","):
        env = OpenFrontEnv(map_name=m, seed="probe", difficulty="medium", max_steps=a.steps)
        obs, info = env.reset(seed=1)
        fracs, boats, launched = [], 0, 0
        for step in range(a.steps):
            inputs = net_inputs(obs_to_batch([obs]), info["action_mask"][None], cpu)
            with torch.no_grad():
                _, tile_logits, _, _ = net(*inputs)
                action, _, _ = net.act(*inputs)
            n = int(obs["boat_target_mask"].sum())
            if info["action_mask"][BOAT] and n > 1:
                pr = torch.softmax(tile_logits[0], -1)
                fracs.append(float(-(pr * torch.log(pr.clamp_min(1e-12))).sum()) / np.log(n))
            before = obs["self_features"][TROOPS]
            obs, _, term, trunc, info = env.step(action[0].numpy())
            if int(action[0, 0]) == BOAT:
                boats += 1
                launched += bool(obs["self_features"][TROOPS] < 0.9 * before)
            if term or trunc:
                break
        env.close()
        print(f"{head} head, update {ckpt['update']}, {m:12s} entropy/max={np.mean(fracs):.2f} "
              f"over {len(fracs)} steps; boat_attack sampled {boats}/{step + 1} steps, {launched} launched")


if __name__ == "__main__":
    main()
