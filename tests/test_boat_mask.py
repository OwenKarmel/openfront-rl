"""Does requiring AGENT to border shore move probability off boat_attack
when it is landlocked?

Sibling of test_expand_mask.py, same construction. Reproduces eval
cc83d2cc (Irish Sea, update 1319) -- an episode where the agent was
landlocked for 100% of its 102,011 ticks, i.e. never owned a single shore
tile, so no boat could ever have launched.

At every step it runs one network forward per mask variant on the SAME
observation, isolating the mask rather than conflating it with a different
trajectory:

  new = the mask the env reports with OPENFRONT_BOAT_MASK=1
  old = the same mask with boat_attack forced legal (the existing
        approximate mask: any opponent-owned tile anywhere => legal)

Caveat, same as the expand test: the policy is the gamma=0.98 run's
update-3000 snapshot, not the update-1319 weights -- checkpoints overwrite.
The scenario is reproduced
exactly; the weights are newer.
"""

import argparse
import os
import sys
from pathlib import Path

# Must be set BEFORE the env subprocess is spawned -- Popen inherits environ.
os.environ["OPENFRONT_BOAT_MASK"] = "1"

import numpy as np
import torch

TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

from models.network import ActorCritic
from train import ACTIONS, make_env, net_inputs, obs_to_batch, territory_fraction

# The update-3000 policy snapshot of the gamma=0.98 run, kept beside that
# run's archive. A snapshot rather than a live latest.pt on purpose: the test
# compares masked vs unmasked probabilities for ONE fixed policy, so pointing
# at whatever run is live would silently change what is being compared. This
# path previously pointed into a /tmp session scratchpad -- tmpfs, so it would
# have vanished on the next reboot.
CKPT = (Path(__file__).resolve().parent.parent / "training" / "checkpoints" / "archive"
        / "run_2026-09-11_bbf-adamw_gamma098_rollout50" / "policy_snapshot_update3000.pt")
UPDATE = 1319          # eval cc83d2cc: landlocked for the whole episode
MAX_STEPS = 1200
BOAT = ACTIONS.index("boat_attack")


def main() -> None:
    args = argparse.Namespace(map=None, ticks_per_step=10, max_episode_steps=20000)
    device = torch.device("cpu")
    net = ActorCritic().to(device)
    if not CKPT.exists():
        sys.exit(f"missing policy snapshot: {CKPT}\n"
                 "It lives beside the gamma=0.98 run's archive, which is gitignored "
                 "like every checkpoint -- this test needs that archive present.")
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"loaded policy snapshot (saved at update {ckpt['update']})")

    env = make_env(args, "easy", seed_prefix=f"eval-{UPDATE}")
    obs, info = env.reset(seed=UPDATE)

    masked_steps = 0
    chosen = np.zeros(len(ACTIONS), dtype=int)
    chosen_old = np.zeros(len(ACTIONS), dtype=int)
    p_new_sum = np.zeros(len(ACTIONS))
    p_old_sum = np.zeros(len(ACTIONS))
    peak = territory_fraction(obs)

    for step in range(MAX_STEPS):
        mask_new = info["action_mask"].copy()
        mask_old = mask_new.copy()
        mask_old[BOAT] = True  # existing approximate mask would allow it

        batch = obs_to_batch([obs])
        with torch.no_grad():
            out_new = net(*net_inputs(batch, mask_new[None], device))
            tl_old = net(*net_inputs(batch, mask_old[None], device))[0]
        p_new = torch.softmax(out_new[0], dim=-1)[0].numpy()
        p_old = torch.softmax(tl_old, dim=-1)[0].numpy()

        if not mask_new[BOAT]:      # landlocked: boat masked out by the new rule
            masked_steps += 1
            p_new_sum += p_new
            p_old_sum += p_old
            chosen[int(p_new.argmax())] += 1
            chosen_old[int(p_old.argmax())] += 1

        action_type = int(p_new.argmax())
        tile_idx = int(torch.argmax(out_new[1]).item())
        player_idx = int(torch.argmax(out_new[2]).item())
        obs, _, term, trunc, info = env.step([action_type, tile_idx, player_idx])
        peak = max(peak, territory_fraction(obs))
        if term or trunc:
            print(f"episode ended at step {step}")
            break

    env.close()

    print(f"\nsteps run: {step + 1}   peak territory: {peak:.4f}")
    print(f"steps where boat_attack was MASKED (agent landlocked): {masked_steps}")
    if masked_steps == 0:
        print("agent was never landlocked here -- inconclusive")
        return

    print("\nOn those masked steps, mean action-type probability:")
    print(f"  {'action':<18}{'OLD (boat legal)':>20}{'NEW (boat masked)':>22}")
    for i, a in enumerate(ACTIONS):
        print(f"  {a:<18}{p_old_sum[i]/masked_steps:>20.4f}{p_new_sum[i]/masked_steps:>22.4f}")

    print("\nGreedy pick on those steps:")
    for i, a in enumerate(ACTIONS):
        if chosen_old[i] or chosen[i]:
            print(f"  {a:<18}old={chosen_old[i]:>5}   new={chosen[i]:>5}")

    b_old = p_old_sum[BOAT] / masked_steps
    print(f"\nboat_attack probability: {b_old:.4f} -> 0.0000 "
          f"({b_old * 100:.1f}pp of mass redistributed to launchable actions)")


if __name__ == "__main__":
    main()
