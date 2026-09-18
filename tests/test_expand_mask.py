"""Does masking `expand` when nothing is conquerable move probability onto
boat_attack?

Reproduces eval 0ab47d4b (Irish Sea, update 2879 -- the episode where the
agent conquered its spawn island and then issued 9,159 expand intents over
91,591 ticks while never once sampling boat_attack), this time with the
canExpand() mask live in EnvServer.ts.

At every step it runs ONE network forward per mask variant on the SAME
observation, so the comparison isolates the mask rather than conflating it
with a different trajectory:

  new = the mask the modified env actually reports
  old = the same mask with `expand` forced legal (previous behaviour)

Caveat: the policy is the gamma=0.98 run's update-3000 snapshot, not the
update-2879 weights that actually produced 0ab47d4b -- checkpoints overwrite,
so those are gone. The scenario (map, seed, opponent) is reproduced exactly;
the policy is ~120 updates newer.
"""

import argparse
import os
import sys

# Must be set BEFORE the env subprocess is spawned -- Popen inherits environ.
os.environ["OPENFRONT_EXPAND_MASK"] = "1"
from pathlib import Path

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
UPDATE = 2879          # the eval that produced replay 0ab47d4b
MAX_STEPS = 1200       # enough to conquer the island and sit in the frozen state
EXPAND = ACTIONS.index("expand")
BOAT = ACTIONS.index("boat_attack")


def main() -> None:
    # map=None so make_env uses map_pool=SMALL_MAPS, exactly as the live
    # eval does -- the pool draw is seeded, so seed 2879 reselects Irish Sea.
    # Forcing map_name="irishsea" with map_pool=None builds the game down a
    # different path and tripped a pre-existing engine bug (NationNukeBehavior
    # line 44 reads this.game in an instance-member initializer, before the
    # constructor parameter property is assigned).
    args = argparse.Namespace(
        map=None, ticks_per_step=10, max_episode_steps=20000,
    )
    device = torch.device("cpu")
    net = ActorCritic().to(device)
    if not CKPT.exists():
        sys.exit(f"missing policy snapshot: {CKPT}\n"
                 "It lives beside the gamma=0.98 run's archive, which is gitignored "
                 "like every checkpoint -- this test needs that archive present.")
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"loaded {CKPT} (saved at update {ckpt['update']})")

    env = make_env(args, "easy", seed_prefix=f"eval-{UPDATE}")
    obs, info = env.reset(seed=UPDATE)

    masked_steps = 0
    chosen = np.zeros(len(ACTIONS), dtype=int)
    chosen_old = np.zeros(len(ACTIONS), dtype=int)
    p_new_sum = np.zeros(len(ACTIONS))
    p_old_sum = np.zeros(len(ACTIONS))
    peak = territory_fraction(obs)
    first_masked_step = None

    for step in range(MAX_STEPS):
        mask_new = info["action_mask"].copy()
        mask_old = mask_new.copy()
        mask_old[EXPAND] = True  # previous behaviour: always legal

        batch = obs_to_batch([obs])
        with torch.no_grad():
            tl_new = net(*net_inputs(batch, mask_new[None], device))[0]
            tl_old = net(*net_inputs(batch, mask_old[None], device))[0]
        p_new = torch.softmax(tl_new, dim=-1)[0].numpy()
        p_old = torch.softmax(tl_old, dim=-1)[0].numpy()

        expand_masked = not mask_new[EXPAND]
        if expand_masked:
            masked_steps += 1
            p_new_sum += p_new
            p_old_sum += p_old
            chosen[int(p_new.argmax())] += 1
            chosen_old[int(p_old.argmax())] += 1
            if first_masked_step is None:
                first_masked_step = step

        action_type = int(p_new.argmax())  # greedy, as eval does
        tile_idx = int(torch.argmax(net(*net_inputs(batch, mask_new[None], device))[1]).item())
        player_idx = int(torch.argmax(net(*net_inputs(batch, mask_new[None], device))[2]).item())
        obs, _, term, trunc, info = env.step([action_type, tile_idx, player_idx])
        peak = max(peak, territory_fraction(obs))
        if term or trunc:
            print(f"episode ended at step {step}")
            break

    env.close()

    print(f"\nsteps run: {step + 1}   peak territory: {peak:.4f}")
    print(f"steps where expand was MASKED (nothing conquerable): {masked_steps}"
          f"  (first at step {first_masked_step})")
    if masked_steps == 0:
        print("no frozen states reached -- inconclusive, try more steps")
        return

    print("\nOn those masked steps, mean action-type probability:")
    print(f"  {'action':<18}{'OLD (expand legal)':>20}{'NEW (expand masked)':>22}")
    for i, a in enumerate(ACTIONS):
        print(f"  {a:<18}{p_old_sum[i]/masked_steps:>20.4f}{p_new_sum[i]/masked_steps:>22.4f}")

    print("\nGreedy pick on those steps:")
    for i, a in enumerate(ACTIONS):
        if chosen_old[i] or chosen[i]:
            print(f"  {a:<18}old={chosen_old[i]:>5}   new={chosen[i]:>5}")

    b_old, b_new = p_old_sum[BOAT] / masked_steps, p_new_sum[BOAT] / masked_steps
    print(f"\nboat_attack probability: {b_old:.4f} -> {b_new:.4f} "
          f"({b_new / b_old:.2f}x)" if b_old > 0 else "\nboat_attack was 0 under both")


if __name__ == "__main__":
    main()
