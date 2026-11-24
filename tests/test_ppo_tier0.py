"""PPO Tier-0 options: --kl-stop, --adv-norm, --value-norm (no env needed).

  1. flag-off bit-identity: with every new option at its default, the
     network (attack-size head off) and ppo_update() produce bit-identical
     weights, actions, log-probs and stats to the pre-change code, vendored
     verbatim in tests/reference/ -- with and without the KL early stop firing
  2. --kl-stop epoch: on a batch whose KL stays small, every epoch runs
     (grad_steps = epochs x full minibatches, the partial one folded in); on
     a batch whose KL is large, it stops BETWEEN epochs, after the first
  3. minibatch_bounds(): "minibatch" keeps the trailing partial minibatch,
     "epoch" folds it into the last one; both cover every sample once
  4. --value-norm: ValueNormalizer's statistics update leaves the head's
     denormalized output unchanged (PopArt), identity at start, the std floor,
     state round-trip -- and end to end, a ppo_update() with lr=0 leaves every
     denormalized prediction where it was
  5. --adv-norm minibatch runs and stays finite

usage: python tests/test_ppo_tier0.py   (exit 1 on any failure; seconds)
"""

import copy
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent / "training"
sys.path.insert(0, str(TRAINING))

from models.network import ActorCritic  # noqa: E402
from ppo import RolloutBuffer, ValueNormalizer, minibatch_bounds, ppo_update  # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")


def load_reference(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / "reference" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


old_net_mod = load_reference("network_v0")
old_ppo_mod = load_reference("ppo_v0")

T, N, H = 9, 3, 64  # 27 samples: minibatch 8 leaves a partial minibatch of 3
MB = 8


def make_obs(rng, b=N):
    tile = rng.integers(0, 4, (b, H, H)).astype(np.int8)
    self_f = (rng.random((b, 4)) * 1000).astype(np.float32)
    opp_f = (rng.random((b, 1, 6)) * 1000).astype(np.float32)
    opp_m = np.ones((b, 1), dtype=bool)
    tile_m = rng.random((b, 32, 32)) < 0.3
    att_m = np.ones((b, 1), dtype=bool)
    act_m = np.ones((b, 4), dtype=bool)
    return tile, self_f, opp_f, opp_m, tile_m, att_m, act_m


def tens(obs):
    tile, self_f, opp_f, opp_m, tile_m, att_m, act_m = obs
    return (torch.as_tensor(tile), torch.as_tensor(self_f), torch.as_tensor(opp_f), torch.as_tensor(opp_m),
            torch.as_tensor(tile_m), torch.as_tensor(att_m), torch.as_tensor(act_m))


def collect(net, seed, act_seed):
    """Rollout of T steps x N envs on random observations, actions from net.act()."""
    rng = np.random.default_rng(seed)
    steps = []
    torch.manual_seed(act_seed)
    for _ in range(T):
        obs = make_obs(rng)
        a, lp, v = net.act(*tens(obs))
        steps.append((obs, a.numpy(), lp.numpy(), v.numpy(),
                      rng.normal(0, 0.05, N).astype(np.float32), (rng.random(N) < 0.1)))
    last = rng.normal(0, 0.1, N).astype(np.float32)
    return steps, last


def to_buffer(steps, action_cols=None, lp_shift=0.0):
    buf = RolloutBuffer()
    for obs, a, lp, v, r, d in steps:
        tile, self_f, opp_f, opp_m, tile_m, att_m, act_m = obs
        acts = a if action_cols is None else a[:, :action_cols]
        buf.add(tile.copy(), self_f, opp_f, opp_m, act_m, tile_m, att_m, acts.copy(), lp + lp_shift, v, r, d)
    return buf


def optimizer_for(net, lr=3e-4):
    decayed = [p for p in net.parameters() if p.dim() >= 2]
    rest = [p for p in net.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decayed, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}], lr=lr)


print("1. flag-off bit-identity vs the pre-change network.py / ppo.py")
torch.manual_seed(0)
old_net = old_net_mod.ActorCritic(tile_head="big")
torch.manual_seed(0)
new_net = ActorCritic(tile_head="big")
sd_old, sd_new = old_net.state_dict(), new_net.state_dict()
check("same parameter names and init values",
      list(sd_old) == list(sd_new) and all(torch.equal(sd_old[k], sd_new[k]) for k in sd_old))

old_steps, old_last = collect(old_net, seed=11, act_seed=5)
new_steps, new_last = collect(new_net, seed=11, act_seed=5)
same_acts = all(np.array_equal(o[1], n[1][:, :3]) and np.array_equal(o[2], n[2]) and np.array_equal(o[3], n[3])
                for o, n in zip(old_steps, new_steps))
check("act(): same sampled actions, log-probs and values (RNG stream unchanged)", same_acts)
check("act(): attack_frac slot is the default index 1 with the head off",
      all((n[1][:, 3] == 1).all() for n in new_steps))

for label, target_kl in (("no early stop", 10.0), ("early stop fires", 1e-4)):
    nets = [copy.deepcopy(old_net), copy.deepcopy(new_net)]
    opts = [optimizer_for(nets[0]), optimizer_for(nets[1])]
    torch.manual_seed(3)
    s_old = old_ppo_mod.ppo_update(nets[0], opts[0], to_buffer(old_steps), old_last, torch.device("cpu"),
                                   gamma=0.999, epochs=4, minibatch_size=MB, target_kl=target_kl)
    torch.manual_seed(3)
    s_new = ppo_update(nets[1], opts[1], to_buffer(new_steps), new_last, torch.device("cpu"),
                       gamma=0.999, epochs=4, minibatch_size=MB, target_kl=target_kl)
    p_old, p_new = dict(nets[0].named_parameters()), dict(nets[1].named_parameters())
    check(f"[{label}] parameters bit-identical after ppo_update",
          all(torch.equal(p_old[k], p_new[k]) for k in p_old))
    check(f"[{label}] logged stats identical", all(s_old[k] == s_new[k] for k in s_old),
          f"old={s_old} new={ {k: s_new[k] for k in s_old} }")
    # Early stop: the first minibatch's k1 is exactly 0 (no step taken yet),
    # so the earliest it can fire is after the second.
    expect_steps = 2 if target_kl < 1 else 4 * 4
    check(f"[{label}] grad_steps={s_new['grad_steps']} (expected {expect_steps}), epochs_completed="
          f"{s_new['epochs_completed']}", s_new["grad_steps"] == expect_steps
          and s_new["epochs_completed"] == (0 if target_kl < 1 else 4))

print("2. --kl-stop epoch")
net = copy.deepcopy(new_net)
s = ppo_update(net, optimizer_for(net, lr=1e-6), to_buffer(new_steps), new_last, torch.device("cpu"),
               gamma=0.999, epochs=4, minibatch_size=MB, target_kl=0.03, kl_stop="epoch")
check("small KL: all 4 epochs run, 3 grad steps each (27 // 8, remainder folded)",
      s["epochs_completed"] == 4 and s["grad_steps"] == 12 and s["stopped_early"] == 0.0
      and s["minibatches_per_epoch"] == 3, str(s))
net = copy.deepcopy(new_net)
s = ppo_update(net, optimizer_for(net, lr=1e-6), to_buffer(new_steps, lp_shift=-3.0), new_last, torch.device("cpu"),
               gamma=0.999, epochs=4, minibatch_size=MB, target_kl=0.03, kl_stop="epoch")
check("large KL: stops between epochs, after exactly one full epoch",
      s["epochs_completed"] == 1 and s["grad_steps"] == 3 and s["stopped_early"] == 1.0
      and s["last_epoch_kl"] > 0.03, str(s))
net = copy.deepcopy(new_net)
s = ppo_update(net, optimizer_for(net, lr=1e-6), to_buffer(new_steps, lp_shift=-3.0), new_last, torch.device("cpu"),
               gamma=0.999, epochs=4, minibatch_size=MB, target_kl=0.03, kl_stop="minibatch")
# The same shift is invisible to the original check: its k1 estimate
# mean(old_logp - logp) is -3 here, and only a large POSITIVE k1 stops.
check(f"the original k1 check misses this KL entirely (k1={s['approx_kl']:+.2f}, k3={s['approx_kl_k3']:.1f}): "
      f"all 16 minibatches run", s["grad_steps"] == 16 and s["stopped_early"] == 0.0, str(s))
check("k3 estimator is >= 0 and finite", s["approx_kl_k3"] >= 0 and math.isfinite(s["approx_kl_k3"]))
net = copy.deepcopy(new_net)
s = ppo_update(net, optimizer_for(net, lr=1e-6), to_buffer(new_steps, lp_shift=3.0), new_last, torch.device("cpu"),
               gamma=0.999, epochs=4, minibatch_size=MB, target_kl=0.03, kl_stop="minibatch")
check("large positive k1 under 'minibatch': stops after the first minibatch, mid-epoch",
      s["grad_steps"] == 1 and s["epochs_completed"] == 0 and s["stopped_early"] == 1.0, str(s))

print("3. minibatch_bounds")
for n_samples, mb in ((27, 8), (24, 8), (5, 8), (10000, 128)):
    for mode in ("minibatch", "epoch"):
        b = minibatch_bounds(n_samples, mb, mode)
        covered = [i for lo, hi in b for i in range(lo, hi)]
        sizes = [hi - lo for lo, hi in b]
        ok = covered == list(range(n_samples))
        if mode == "epoch":
            ok &= all(sz >= min(mb, n_samples) for sz in sizes) and len(b) == max(n_samples // mb, 1)
        else:
            ok &= len(b) == math.ceil(n_samples / mb)
        check(f"{mode} n={n_samples} mb={mb}: sizes {sizes[:3]}..{sizes[-1:]} ({len(b)} minibatches)", ok)

print("4. --value-norm (PopArt)")
torch.manual_seed(7)
head = nn.Linear(16, 1)
vn = ValueNormalizer(head)
x = torch.randn(200, 16)
with torch.no_grad():
    raw0 = head(x).squeeze(-1)
check("identity at start: denormalize(head(x)) == head(x) exactly",
      torch.equal(vn.denormalize(raw0), raw0) and vn.mean == 0.0 and vn.std == 1.0)
rng = np.random.default_rng(0)
worst = 0.0
for mu, sigma in ((5.0, 3.0), (-2.0, 0.1), (100.0, 50.0), (0.3, 0.6), (0.0, 1e-6)):
    with torch.no_grad():
        before = vn.denormalize(head(x).squeeze(-1))
    vn.update(rng.normal(mu, sigma, 5000))
    with torch.no_grad():
        after = vn.denormalize(head(x).squeeze(-1))
    worst = max(worst, float(((after - before).abs() / before.abs().clamp_min(1.0)).max()))
check(f"statistics updates preserve the denormalized output (worst rel. error {worst:.1e})", worst < 1e-5)
vn2 = ValueNormalizer(nn.Linear(16, 1))
for _ in range(2000):
    vn2.update(rng.normal(4.0, 2.0, 1000))
check(f"tracks the return distribution: mean {vn2.mean:.3f} std {vn2.std:.3f} (true 4, 2)",
      abs(vn2.mean - 4.0) < 0.05 and abs(vn2.std - 2.0) < 0.05)
vn3 = ValueNormalizer(nn.Linear(16, 1))
vn3.update(np.full(100, 0.5))
check("std floor on constant returns", vn3.std == 0.01 and abs(vn3.mean - 0.5) < 1e-12)
vn4 = ValueNormalizer(nn.Linear(16, 1))
vn4.load_state_dict(vn2.state_dict())
check("state_dict round-trip", vn4.mean == vn2.mean and vn4.std == vn2.std)

net = copy.deepcopy(new_net)
vn = ValueNormalizer(net.value_head)
vn.update(rng.normal(0.4, 0.3, 1000))  # non-identity before the update under test
obs_eval = tens(make_obs(np.random.default_rng(99), b=16))
with torch.no_grad():
    before = vn.denormalize(net(*obs_eval)[3])
steps_vn, last_vn = collect(net, seed=12, act_seed=6)
s = ppo_update(net, optimizer_for(net, lr=0.0), to_buffer(steps_vn), last_vn, torch.device("cpu"),
               gamma=0.999, epochs=2, minibatch_size=MB, kl_stop="epoch", adv_norm="minibatch",
               value_normalizer=vn)
with torch.no_grad():
    after = vn.denormalize(net(*obs_eval)[3])
check(f"ppo_update with lr=0: statistics moved (mean {vn.mean:+.3f}, std {vn.std:.3f}) but every "
      f"denormalized prediction is unchanged (max diff {float((after - before).abs().max()):.1e})",
      float((after - before).abs().max()) < 1e-5 and vn.w > 0.01)

print("5. --adv-norm minibatch")
net = copy.deepcopy(new_net)
s = ppo_update(net, optimizer_for(net), to_buffer(new_steps), new_last, torch.device("cpu"),
               gamma=0.999, epochs=2, minibatch_size=MB, adv_norm="minibatch")
check("runs, finite losses", all(math.isfinite(s[k]) for k in ("policy_loss", "value_loss", "entropy")), str(s))

print("ALL PASSED" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
