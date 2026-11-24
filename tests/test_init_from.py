"""--init-from, checkpoint value-norm handling, and the eval-report helpers
(no env needed).

--init-from (train.py's init_from_checkpoint):
  - loads every shared weight, carries the update count and curriculum
    position, and adds a fresh attack-size head (P(20%) = 0.9)
  - moves the optimizer state of every matched parameter by NAME (the param
    groups shift when a head is added), leaves the new head's empty
  - refuses: a missing parameter other than the new head's, an unexpected
    one (e.g. a big-tile-head checkpoint into a small-head run), a different
    curriculum ladder
load_checkpoint / check_value_norm:
  - a --value-norm checkpoint refuses to resume without --value-norm; one
    without statistics loads as identity; attack-size-head mismatch refuses
Eval reporting:
  - eval_seed_maps() reproduces the env's own seed -> map draw
  - canary_seeds(): a fixed, map-stratified subset of the eval seeds
  - summarize_suite(exclude_maps), per_map_line(), open_csv() with an old header

usage: python tests/test_init_from.py   (exit 1 on any failure; seconds)
"""

import argparse
import csv
import io
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import torch

TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))

import train  # noqa: E402
from curriculum import CurriculumScheduler  # noqa: E402
from envs.openfront_env import OpenFrontEnv, SMALL_MAPS  # noqa: E402
from models.network import ActorCritic  # noqa: E402
from ppo import ValueNormalizer  # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")


def make_optimizer(model):
    decayed = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decayed, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}], lr=3e-4)


def exits(fn) -> tuple[bool, str]:
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            fn()
    except SystemExit as e:
        return True, str(e)
    return False, ""


tmp = Path(tempfile.mkdtemp(prefix="test_init_from_", dir=Path(__file__).resolve().parent))
try:
    print("--init-from")
    torch.manual_seed(0)
    src = ActorCritic(tile_head="big")
    src_opt = make_optimizer(src)
    # A few real steps so every parameter has optimizer state worth moving.
    x = (torch.randint(0, 4, (8, 64, 64), dtype=torch.int8), torch.rand(8, 4) * 1000, torch.rand(8, 1, 6) * 1000,
         torch.ones(8, 1, dtype=torch.bool), torch.rand(8, 32, 32) < 0.3, torch.ones(8, 1, dtype=torch.bool),
         torch.ones(8, 4, dtype=torch.bool))
    for _ in range(3):
        a, _, _ = src.act(*x)
        lp, ent, v = src.evaluate_actions(*x, a)
        src_opt.zero_grad()
        (lp.mean() + v.pow(2).mean() - ent.mean()).backward()
        src_opt.step()
    sched = CurriculumScheduler(ladder="connected-first")
    sched._index = 3
    ckpt_path = tmp / "src.pt"
    train.save_checkpoint(ckpt_path, src, src_opt, sched, 5310)

    dst = ActorCritic(tile_head="big", attack_size_head=True)
    dst_opt = make_optimizer(dst)
    dst_sched = CurriculumScheduler(ladder="connected-first")
    with redirect_stdout(io.StringIO()):
        upd = train.init_from_checkpoint(ckpt_path, dst, dst_opt, dst_sched, ValueNormalizer(dst.value_head),
                                         make_optimizer)
    sd_s, sd_d = src.state_dict(), dst.state_dict()
    check("update count and curriculum stage carried", upd == 5310 and dst_sched._index == 3)
    check("every shared weight loaded", all(torch.equal(sd_s[k], sd_d[k]) for k in sd_s))
    probs = torch.softmax(dst.attack_frac_head.bias, -1)
    check("new head initialized (zero weights, P(20%) = 0.9)",
          torch.count_nonzero(dst.attack_frac_head.weight) == 0 and abs(probs[1].item() - 0.9) < 1e-6)
    sp, dp = dict(src.named_parameters()), dict(dst.named_parameters())
    moved = all(torch.equal(src_opt.state[sp[k]]["exp_avg"], dst_opt.state[dp[k]]["exp_avg"])
                and torch.equal(src_opt.state[sp[k]]["exp_avg_sq"], dst_opt.state[dp[k]]["exp_avg_sq"])
                and float(dst_opt.state[dp[k]]["step"]) == 3 for k in sp)
    check("optimizer state moved by name for every matched parameter", moved)
    check("new head has no optimizer state (AdamW starts it fresh)",
          dst.attack_frac_head.weight not in dst_opt.state and dst.attack_frac_head.bias not in dst_opt.state)
    # The moved state must be usable: one optimizer step must run.
    a, _, _ = dst.act(*x)
    lp, ent, v = dst.evaluate_actions(*x, a)
    dst_opt.zero_grad()
    (lp.mean() + v.pow(2).mean() - ent.mean()).backward()
    dst_opt.step()
    check("optimizer steps after the move (new head now has state)", dst.attack_frac_head.weight in dst_opt.state)

    small = ActorCritic(tile_head="small", attack_size_head=True)
    hit, msg = exits(lambda: train.init_from_checkpoint(ckpt_path, small, make_optimizer(small),
                                                         CurriculumScheduler(ladder="connected-first"), None, make_optimizer))
    check("refuses a different tile head (unexpected + missing keys)", hit and "architecture mismatch" in msg, msg[:90])

    broken = torch.load(ckpt_path, weights_only=False)
    del broken["model"]["trunk.0.weight"]
    torch.save(broken, tmp / "broken.pt")
    d2 = ActorCritic(tile_head="big", attack_size_head=True)
    hit, msg = exits(lambda: train.init_from_checkpoint(tmp / "broken.pt", d2, make_optimizer(d2),
                                                         CurriculumScheduler(ladder="connected-first"), None, make_optimizer))
    check("refuses a missing parameter that is not the new head's", hit and "trunk.0.weight" in msg, msg[:90])

    d3 = ActorCritic(tile_head="big", attack_size_head=True)
    hit, msg = exits(lambda: train.init_from_checkpoint(ckpt_path, d3, make_optimizer(d3),
                                                         CurriculumScheduler(ladder="difficulty"), None, make_optimizer))
    check("refuses a different curriculum ladder", hit and "ladder" in msg, msg[:90])

    print("checkpoint value-norm / head checks")
    vn_net = ActorCritic(tile_head="big")
    vn = ValueNormalizer(vn_net.value_head)
    vn.update(torch.randn(500).numpy() * 0.3 + 0.2)
    train.save_checkpoint(tmp / "vn.pt", vn_net, make_optimizer(vn_net), sched, 10, vn, {"train": 5.0, "eval": 1.0})
    r = ActorCritic(tile_head="big")
    hit, msg = exits(lambda: train.load_checkpoint(tmp / "vn.pt", r, make_optimizer(r),
                                                   CurriculumScheduler(ladder="connected-first"), None))
    check("a --value-norm checkpoint refuses to resume without --value-norm", hit and "--value-norm" in msg, msg[:80])
    r = ActorCritic(tile_head="big")
    vn_r = ValueNormalizer(r.value_head)
    wall = {"train": 0.0, "eval": 0.0}
    with redirect_stdout(io.StringIO()):
        train.load_checkpoint(tmp / "vn.pt", r, make_optimizer(r), CurriculumScheduler(ladder="connected-first"), vn_r, wall)
    check("statistics and cumulative wall time restored on resume",
          vn_r.mean == vn.mean and vn_r.std == vn.std and wall == {"train": 5.0, "eval": 1.0})
    r = ActorCritic(tile_head="big")
    vn_id = ValueNormalizer(r.value_head)
    with redirect_stdout(io.StringIO()):
        train.load_checkpoint(ckpt_path, r, make_optimizer(r), CurriculumScheduler(ladder="connected-first"), vn_id)
    check("a checkpoint without statistics loads as identity", vn_id.mean == 0.0 and vn_id.std == 1.0)
    r = ActorCritic(tile_head="big", attack_size_head=True)
    hit, msg = exits(lambda: train.load_checkpoint(ckpt_path, r, make_optimizer(r),
                                                   CurriculumScheduler(ladder="connected-first"), None))
    check("resume refuses an --attack-size-head mismatch", hit and "--attack-size-head" in msg, msg[:80])
    keys = set(torch.load(ckpt_path, weights_only=False))
    check("saved without value-norm or wall time, a checkpoint has exactly the original keys",
          keys == {"model", "optimizer", "update", "curriculum_index", "curriculum_ladder", "curriculum_mode"}, str(keys))

    print("eval reporting")
    args = argparse.Namespace(map=None, eval_episodes=64, canary_episodes=16)
    seed_maps = train.eval_seed_maps(args, "small", 64)
    env_maps = []
    for k in range(64):
        e = OpenFrontEnv.__new__(OpenFrontEnv)  # the env's own draw, without a Node process
        e._np_random = None
        import gymnasium
        gymnasium.Env.reset(e, seed=k)
        env_maps.append(str(e.np_random.choice(SMALL_MAPS)))
    check("eval_seed_maps() matches OpenFrontEnv.reset()'s map draw", seed_maps == env_maps)
    canary = train.canary_seeds(args, "small")
    cmaps = [seed_maps[k] for k in canary]
    check(f"canary: 16 distinct eval seeds over 16 distinct maps ({canary})",
          len(set(canary)) == 16 and len(set(cmaps)) == 16 and all(0 <= k < 64 for k in canary))
    check("canary seeds are fixed", train.canary_seeds(args, "small") == canary)
    args.canary_episodes = 64
    check("canary of the full size is the whole suite", train.canary_seeds(args, "small") == list(range(64)))

    eps = [{"seed": i, "map": m, "winner": w, "ticks": 10, "truncated": False, "peak_territory_frac": 0.1,
            "final_territory_frac": 0.1}
           for i, (m, w) in enumerate([("labyrinth", "AGENT"), ("labyrinth", "AGENT"), ("onion", "OPPONENT"),
                                       ("onion", "AGENT"), ("baltics", "OPPONENT")])]
    s = train.summarize_suite(eps, ("labyrinth",))
    check("summarize_suite excludes maps from wins_excl/episodes_excl only",
          s["wins"] == 3 and s["episodes"] == 5 and s["wins_excl"] == 1 and s["episodes_excl"] == 3, str(s))
    check("no exclusion -> no excl columns", "wins_excl" not in train.summarize_suite(eps))
    check("per_map_line", train.per_map_line(eps) == "labyrinth 2/2 onion 1/2 baltics 0/1", train.per_map_line(eps))

    old = tmp / "old_eval_log.csv"
    old.write_text("a,b\n1,2\n")
    with redirect_stdout(io.StringIO()):
        fh, w, dropped = train.open_csv(old, ["a", "b", "c"], new_columns=("c",))
        w.writerow({"a": 3, "b": 4, "c": 5})
        fh.close()
    rows = list(csv.reader(open(old)))
    check("open_csv keeps an older file's header and drops the new column", dropped == ["c"] and rows[-1] == ["3", "4"])
    hit, _ = exits(lambda: train.open_csv(old, ["a", "x"]))
    check("open_csv still refuses an unrelated header", hit)
finally:
    shutil.rmtree(tmp)

print("ALL PASSED" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
