"""--param-entropy-norm: parameter-head entropy divided by log(#legal choices).

Same weights, same inputs, flag off vs on, through the real
ActorCritic.evaluate_actions(). Checks that only the entropy changes, and
only the parameter heads' terms of it:

  - log_prob and value are identical (the policy itself is untouched)
  - a noop/expand row's entropy is identical (no parameter head involved)
  - a boat_attack row's entropy is type entropy + raw tile entropy / log(n),
    n = that row's number of legal macro-cells
  - a single legal cell (zero entropy) gives 0, not NaN

usage: python tests/test_param_entropy_norm.py   (exit 1 on any failure)
"""

import math
import sys
from pathlib import Path

import torch

TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

from envs.openfront_env import ACTIONS, MACRO_GRID
from models.network import ActorCritic

BOAT, EXPAND = ACTIONS.index("boat_attack"), ACTIONS.index("expand")


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    return ok


def main() -> None:
    torch.manual_seed(0)
    raw, norm = ActorCritic(), ActorCritic(param_entropy_norm=True)
    norm.load_state_dict(raw.state_dict())

    B = 3
    tile_grid = torch.randint(0, 4, (B, 128, 128), dtype=torch.int8)
    self_features = torch.rand(B, 4) * 1000
    opponent_features = torch.rand(B, 1, 6) * 1000
    opponent_mask = torch.ones(B, 1, dtype=torch.bool)
    tile_mask = torch.zeros(B, MACRO_GRID, MACRO_GRID, dtype=torch.bool)
    tile_mask[0, :10, :10] = True   # row 0: 100 legal cells, boat
    tile_mask[1, 3, 3] = True       # row 1: exactly one legal cell, boat
    tile_mask[2, :5, :5] = True     # row 2: expand, tile head irrelevant
    attack_mask = torch.ones(B, 1, dtype=torch.bool)
    action_mask = torch.ones(B, len(ACTIONS), dtype=torch.bool)
    actions = torch.tensor([[BOAT, 5, 0], [BOAT, 3 * MACRO_GRID + 3, 0], [EXPAND, 0, 0]])
    inputs = (tile_grid, self_features, opponent_features, opponent_mask, tile_mask, attack_mask, action_mask)

    with torch.no_grad():
        lp_r, ent_r, v_r = raw.evaluate_actions(*inputs, actions)
        lp_n, ent_n, v_n = norm.evaluate_actions(*inputs, actions)
        type_logits, tile_logits, _, _ = raw(*inputs)
    type_ent = torch.distributions.Categorical(logits=type_logits).entropy()
    tile_ent = torch.distributions.Categorical(logits=tile_logits).entropy()

    fails = 0
    fails += not check("log_prob identical", torch.equal(lp_r, lp_n))
    fails += not check("value identical", torch.equal(v_r, v_n))
    fails += not check("off: boat row = type + raw tile entropy",
                       torch.isclose(ent_r[0], type_ent[0] + tile_ent[0]).item())
    want = type_ent[0] + tile_ent[0] / math.log(100)
    fails += not check("on: boat row = type + tile / log(100 legal cells)", torch.isclose(ent_n[0], want).item(),
                       f"{ent_r[0]:.3f} -> {ent_n[0]:.3f} (tile term {tile_ent[0]:.3f} -> {tile_ent[0] / math.log(100):.3f})")
    fails += not check("on: one legal cell -> finite, type entropy only",
                       torch.isfinite(ent_n[1]).item() and torch.isclose(ent_n[1], type_ent[1], atol=1e-5).item())
    fails += not check("non-parameter action row unchanged", torch.equal(ent_r[2], ent_n[2]))

    print(f"\n{'ALL PASSED' if fails == 0 else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
