"""--attack-size-head, network side (no env needed; tests/test_attack_size.py
covers the env-bridge).

  - off: no new parameters, so existing checkpoints load strictly, and act()
    fills slot 3 with the default index without sampling it
  - on: starts at P(20%) = 0.9 and 0.025 for each other fraction, in every
    state (zero weights); a fresh head over a head-off checkpoint is missing
    exactly attack_frac_head.{weight,bias}
  - masked-sum composition: the attack-size term is in log_prob and entropy
    for expand / attack_opponent / boat_attack and nowhere else, so for the
    other types the head-on and head-off networks agree exactly
  - gradient reaches the head only through attack-type rows
  - --param-entropy-norm divides its entropy by log(5)

usage: python tests/test_attack_size_head.py   (exit 1 on any failure)
"""

import math
import sys
from pathlib import Path

import torch

TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

from envs.openfront_env import ACTIONS, ATTACK_FRACTIONS, DEFAULT_ATTACK_FRACTION_IDX  # noqa: E402
from models.network import (  # noqa: E402
    DEFAULT_ATTACK_FRACTION_IDX as NET_DEFAULT_IDX,
    NUM_ATTACK_FRACTIONS,
    ActorCritic,
)

fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")


def inputs(b, size=64):
    return (torch.randint(0, 4, (b, size, size), dtype=torch.int8), torch.rand(b, 4) * 1000,
            torch.rand(b, 1, 6) * 1000, torch.ones(b, 1, dtype=torch.bool), torch.rand(b, 32, 32) < 0.3,
            torch.ones(b, 1, dtype=torch.bool), torch.ones(b, len(ACTIONS), dtype=torch.bool))


check("constants agree with openfront_env.py",
      NUM_ATTACK_FRACTIONS == len(ATTACK_FRACTIONS) and NET_DEFAULT_IDX == DEFAULT_ATTACK_FRACTION_IDX
      and ATTACK_FRACTIONS[NET_DEFAULT_IDX] == 0.20)

torch.manual_seed(0)
off = ActorCritic(tile_head="big")
on = ActorCritic(tile_head="big", attack_size_head=True)
missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
check("head-off checkpoint into head-on net: missing exactly the new head",
      sorted(missing) == ["attack_frac_head.bias", "attack_frac_head.weight"] and not unexpected, f"{missing} {unexpected}")
check("head-off net has no attack_frac_head parameters",
      not any(k.startswith("attack_frac_head") for k in off.state_dict()))

x = inputs(32)
with torch.no_grad():
    frac_logits = on._heads(*x)[4]
probs = torch.softmax(frac_logits, -1)
target = torch.tensor([0.025, 0.9, 0.025, 0.025, 0.025])
check(f"initial probabilities {probs[0].tolist()} in every state",
      torch.allclose(probs, target.expand_as(probs), atol=1e-6))
check("attack_frac_diagnostics() matches, None with the head off",
      torch.allclose(on.attack_frac_diagnostics(*x), target, atol=1e-6) and off.attack_frac_diagnostics(*x) is None)

torch.manual_seed(1)
a_off, _, _ = off.act(*x)
a_on, _, _ = on.act(*x)
check("act() returns 4 slots; head off fills slot 3 with the default index",
      a_off.shape == (32, 4) and bool((a_off[:, 3] == NET_DEFAULT_IDX).all()) and a_on.shape == (32, 4))

# Composition: same shared weights in both nets; one row per action type,
# every fraction index.
b = len(ACTIONS) * NUM_ATTACK_FRACTIONS
xb = inputs(b)
acts = torch.tensor([[t, 0, 0, f] for t in range(len(ACTIONS)) for f in range(NUM_ATTACK_FRACTIONS)])
torch.manual_seed(2)
with torch.no_grad():
    on.attack_frac_head.weight.normal_(0, 0.05)  # state-dependent, so the test is not trivial
    lp_on, ent_on, v_on = on.evaluate_actions(*xb, acts)
    lp_off, ent_off, v_off = off.evaluate_actions(*xb, acts[:, :3])
    frac_lp = torch.log_softmax(on._heads(*xb)[4], -1).gather(1, acts[:, 3:]).squeeze(1)
    frac_ent = torch.distributions.Categorical(logits=on._heads(*xb)[4]).entropy()
sized = torch.tensor([ACTIONS[t] in ("expand", "attack_opponent", "boat_attack") for t in acts[:, 0].tolist()])
check("log_prob = head-off log_prob + attack-size term, for attack types only",
      torch.allclose(lp_on, lp_off + sized.float() * frac_lp, atol=1e-5))
check("entropy likewise", torch.allclose(ent_on, ent_off + sized.float() * frac_ent, atol=1e-5))
check("noop rows identical to the head-off net",
      torch.equal(lp_on[~sized], lp_off[~sized]) and torch.equal(ent_on[~sized], ent_off[~sized]))
check("value unchanged by the head", torch.equal(v_on, v_off))

on.zero_grad()
noop_rows = acts[:, 0] == ACTIONS.index("noop")
on.evaluate_actions(*[t[noop_rows] for t in xb], acts[noop_rows])[0].sum().backward()
g_noop = on.attack_frac_head.weight.grad.abs().sum().item()
on.zero_grad()
on.evaluate_actions(*[t[~noop_rows] for t in xb], acts[~noop_rows])[0].sum().backward()
g_att = on.attack_frac_head.weight.grad.abs().sum().item()
check(f"gradient reaches the head from attack rows only (noop {g_noop:.2e}, attacks {g_att:.2e})",
      g_noop == 0.0 and g_att > 0)

norm = ActorCritic(tile_head="big", attack_size_head=True, param_entropy_norm=True)
norm.load_state_dict(on.state_dict())
with torch.no_grad():
    _, ent_norm, _ = norm.evaluate_actions(*xb, acts)
    off_norm = ActorCritic(tile_head="big", param_entropy_norm=True)
    off_norm.load_state_dict(off.state_dict())
    _, ent_off_norm, _ = off_norm.evaluate_actions(*xb, acts[:, :3])
check("--param-entropy-norm divides the attack-size entropy by log(5)",
      torch.allclose(ent_norm, ent_off_norm + sized.float() * frac_ent / math.log(NUM_ATTACK_FRACTIONS), atol=1e-5))

print("ALL PASSED" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
