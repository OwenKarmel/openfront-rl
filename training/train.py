"""Phase 3: the actual PPO training loop -- self-play/curriculum training
of an agent against OpenFrontIO's real built-in Nation-AI, working toward
the v1 milestone (beat Impossible 1v1).

Usage:
    python train.py [--num-envs 4] [--rollout-length 64] [--updates 0]
                     [--map onion] [--checkpoint-dir checkpoints]
                     [--checkpoint-every 20] [--eval-every 20]

`--updates` defaults to 0, meaning run indefinitely (no fixed stopping
point) -- pass a positive value to cap it instead. Resumes automatically
from <checkpoint-dir>/latest.pt if present (model, optimizer, curriculum
scheduler state, update count) -- safe to Ctrl-C and rerun, which matters
given this trains across Kaggle session boundaries (12h/session cap) as
well as the local GPU.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from curriculum import LADDERS, CurriculumScheduler
from gymnasium.utils import seeding

from envs.openfront_env import (
    ACTIONS,
    ATTACK_FRACTIONS,
    DIFFICULTIES,
    MAP_POOLS,
    REWARD_MODES,
    RESIZE_DIM,
    SMALL_MAPS,
    OpenFrontEnv,
    requested_env_flags,
)
from envs.vector_env import VecEnv
from models.network import ActorCritic
from ppo import ADV_NORM_MODES, KL_STOP_MODES, RolloutBuffer, ValueNormalizer, ppo_update
from selfplay import POOL_FILENAME, OpponentChooser, OpponentPool, opponent_actions, parse_expert_frac

# Every observation field the network consumes, stacked across envs. Kept as
# one dict rather than a tuple of parallel locals: the network takes seven
# inputs now, and threading seven same-looking arrays through the loop
# positionally is how they end up silently swapped.
BATCH_KEYS = (
    "tile_grid",
    "self_features",
    "opponent_features",
    "opponent_mask",
    "boat_target_mask",
    "attack_target_mask",
)


def obs_to_batch(obs_list: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.stack([o[k] for o in obs_list]) for k in BATCH_KEYS}


def net_inputs(batch: dict[str, np.ndarray], action_mask: np.ndarray, device) -> tuple:
    """The tensors every ActorCritic entry point takes, in its argument
    order -- forward(), act(), and type_diagnostics() all share this
    signature (evaluate_actions() appends the taken actions)."""
    return (
        torch.as_tensor(batch["tile_grid"], device=device),
        torch.as_tensor(batch["self_features"], dtype=torch.float32, device=device),
        torch.as_tensor(batch["opponent_features"], dtype=torch.float32, device=device),
        torch.as_tensor(batch["opponent_mask"], dtype=torch.bool, device=device),
        torch.as_tensor(batch["boat_target_mask"], dtype=torch.bool, device=device),
        torch.as_tensor(batch["attack_target_mask"], dtype=torch.bool, device=device),
        torch.as_tensor(action_mask, dtype=torch.bool, device=device),
    )


def mask_batch(infos: list[dict]) -> np.ndarray:
    return np.stack([info["action_mask"] for info in infos])


# Schedules. Every function below is a pure function of the update counter
# (plus args), never of mutable loop state. That is what makes them resume-
# safe: checkpoints store the update number but not hyperparameters, so a
# resume reconstructs the schedule only by recomputing it. Anything held in a
# running variable would silently restart the anneal on every resume.


def gamma_at(update: int, args: argparse.Namespace) -> float:
    """Discount for this update.

    Constant --gamma unless --gamma-start is set, in which case gamma anneals
    from --gamma-start to --gamma over --gamma-anneal-updates updates,
    beginning at --gamma-anneal-start.

    The anneal is geometric in (1-gamma), not linear in gamma, so the
    effective horizon 1/(1-gamma) grows by a constant FRACTION per update.
    Linear-in-gamma is badly non-uniform in the horizon: 0.98 -> 0.999 moves
    it only 50 -> 174 steps over the first 75% of the anneal, then 174 ->
    1000 in the last 25%, dumping nearly all of the value target's
    non-stationarity into the end.
    """
    if args.gamma_start is None:
        return args.gamma
    progress = (update - args.gamma_anneal_start) / args.gamma_anneal_updates
    if progress <= 0.0:
        return args.gamma_start
    if progress >= 1.0:
        return args.gamma  # exact, rather than 1 - (1-gamma) with float drift
    ratio = (1.0 - args.gamma) / (1.0 - args.gamma_start)
    return 1.0 - (1.0 - args.gamma_start) * ratio**progress


def horizon_steps(gamma: float) -> int:
    """gamma's effective horizon, 1/(1-gamma), in decision steps. One decision
    step is 1.0s of game time (ticks-per-step 10 x msPerTick 100)."""
    return round(1.0 / (1.0 - gamma))


def rollout_at(update: int, args: argparse.Namespace) -> int:
    """Rollout length (decision steps per env) for this update.

    --rollout-length exactly, unless --rollout-match-horizon is set: then the
    rollout is sized to exactly one discount horizon of this update's gamma,
    so it grows with the gamma anneal. Collecting past the point the return
    stops weighting buys nothing for credit assignment, and stopping short of
    it truncates the return at the bootstrap more often than it has to.
    """
    if not args.rollout_match_horizon:
        return args.rollout_length
    return horizon_steps(gamma_at(update, args))


def steps_before(update: int, args: argparse.Namespace) -> int:
    """Decision steps per env collected by updates [0, update). Recomputed from
    the schedule on resume rather than read from a counter, for the reason in
    the section comment above."""
    return sum(rollout_at(u, args) for u in range(update))


def eval_due(update: int, steps_prev: int, steps_now: int, args: argparse.Namespace) -> bool:
    """Whether to run an eval after this update.

    With --eval-every-steps, evals fire each time cumulative per-env decision
    steps cross a multiple of it. That keeps "eval n" meaning the same amount
    of gameplay when the rollout length varies -- which analysis/README.md
    relies on as the one deliberately-comparable cross-run axis. Keyed on
    update count instead, 60 updates is 3000 steps at T=50 but 60000 at
    T=1000. For a constant rollout the two are identical.
    """
    if args.eval_every_steps > 0:
        return steps_prev // args.eval_every_steps != steps_now // args.eval_every_steps
    return (update + 1) % args.eval_every == 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--map", default=None,
        help="fixed map name for every episode; if omitted (default), each "
             "episode independently picks a random map from SMALL_MAPS "
             "(every <=1500x1500 map, see openfront_env.py)",
    )
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--rollout-length", type=int, default=64)
    p.add_argument(
        "--rollout-match-horizon", action="store_true",
        help="size every update's rollout to one discount horizon of that update's "
             "gamma, round(1/(1-gamma)), so it grows as gamma anneals. --rollout-length "
             "is ignored when set",
    )
    p.add_argument("--ticks-per-step", type=int, default=10)
    # Safety cap only -- see OpenFrontEnv's max_steps docstring. Episodes
    # (training and eval alike) should end via a real win/loss
    # (Episode.isDone()), not by running out of decision steps.
    p.add_argument("--max-episode-steps", type=int, default=20000)
    p.add_argument(
        "--reward-mode", choices=REWARD_MODES, default=REWARD_MODES[0],
        help="which reward terms the env pays: shaping_and_terminal (default: tile-share "
             "delta every step plus +/-1 when the game ends), shaping_only (tile-share "
             "delta only), or terminal_only (+/-1 at the end only). Applies to training "
             "and eval alike. A --max-episode-steps truncation never gets the terminal "
             "term in any mode",
    )

    p.add_argument(
        "--potential-beta", type=float, default=0.0,
        help="convexity of the shaping potential: Phi(p) = (exp(beta*p)-1)/beta over "
             "AGENT's tile share p, so one tile is worth exp(beta*p) times what it is "
             "worth at spawn -- gains AND losses at high territory count for more than "
             "at low territory. 0 (default) is the linear tile-share delta, "
             "bit-identical to before this option existed. Only affects the shaping "
             "term, so it does nothing under --reward-mode terminal_only",
    )
    p.add_argument(
        "--updates", type=int, default=0,
        help="stop after this many updates; 0 (default) or negative runs indefinitely",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--weight-decay", type=float, default=0.0,
        help="AdamW decoupled weight decay, applied to >=2D params only (conv "
             "kernels, linear weight matrices), never to biases. 0 (default) "
             "reproduces plain Adam. BBF uses 0.1 to offset the overfitting "
             "that reusing each transition many times invites",
    )
    p.add_argument(
        "--gamma", type=float, default=0.99,
        help="discount; with --gamma-start set, the FINAL value the anneal reaches",
    )
    p.add_argument(
        "--gamma-start", type=float, default=None,
        help="anneal gamma from this value to --gamma, geometrically in (1-gamma) "
             "(see gamma_at). Unset (default) keeps gamma constant at --gamma",
    )
    p.add_argument(
        "--gamma-anneal-updates", type=int, default=0,
        help="length of the gamma anneal, in updates; required with --gamma-start",
    )
    p.add_argument(
        "--gamma-anneal-start", type=int, default=0,
        help="update at which the anneal begins. 0 for a cold start; set it to the "
             "checkpoint's update count to warm-start a schedule from an existing run, "
             "which would otherwise resume already past the end of the anneal",
    )
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument(
        "--tile-head", choices=["small", "big"], default="small",
        help="boat_attack's target head: 'small' (default) is the original 1x1 conv; 'big' is "
             "network.py's BigTileHead (dilated residual convs over the 32x32 target grid, "
             "conditioned on the trunk's game state). Changes the weights, so a checkpoint "
             "only resumes under the head it was trained with",
    )
    p.add_argument(
        "--attack-size-head", action="store_true",
        help="add a categorical head choosing how many troops each expand/attack_opponent/"
             "boat_attack sends, as a fraction of current troops (openfront_env.py's "
             "ATTACK_FRACTIONS). Off (default), every attack sends troops/5 as before. Adds "
             "weights, so a checkpoint only resumes under the setting it was trained with "
             "(--init-from may add the head to a checkpoint without one)",
    )
    p.add_argument(
        "--init-from", default=None,
        help="warm start a NEW run directory from another run's checkpoint: weights, optimizer "
             "state (matched parameters), update count, curriculum position and ladder. Only "
             "the attack-size head may be missing from it. Ignored when --checkpoint-dir already "
             "holds latest.pt, which is resumed instead, so relaunching a launcher is safe",
    )
    p.add_argument(
        "--param-entropy-norm", action="store_true",
        help="divide the tile/player heads' entropy by log(#legal choices) in the entropy "
             "bonus, so each contributes a fraction of its own maximum instead of up to "
             "ln(1024) nats against the type head's ln(4) (see network.py). The logged "
             "`entropy` is then that normalized bonus entropy; type_entropy is unaffected",
    )
    p.add_argument(
        "--target-kl",
        type=float,
        default=0.03,
        help="KL early stop threshold; see --kl-stop for how it is applied",
    )
    p.add_argument(
        "--kl-stop", choices=KL_STOP_MODES, default=KL_STOP_MODES[0],
        help="'minibatch' (default, the original): end the update after any minibatch whose k1 "
             "approx_kl exceeds 1.5x --target-kl, trailing partial minibatch included. 'epoch': "
             "average the k3 estimator over each epoch and stop BETWEEN epochs when that mean "
             "exceeds --target-kl; the partial minibatch is folded into the last full one",
    )
    p.add_argument(
        "--adv-norm", choices=ADV_NORM_MODES, default=ADV_NORM_MODES[0],
        help="normalize advantages over the whole rollout ('batch', default) or per minibatch",
    )
    p.add_argument(
        "--value-norm", action="store_true",
        help="PopArt value normalization (ppo.py's ValueNormalizer): the value head predicts "
             "returns normalized by a running mean/std, rescaled on every statistics change so "
             "its denormalized output is preserved. The statistics are stored in checkpoints; a "
             "checkpoint without them starts from identity (mean 0, std 1)",
    )
    p.add_argument("--epochs", type=int, default=4)
    # Kept well below the full-rollout batch size (num_envs * rollout_length,
    # 1200 at the current defaults) -- a single full-batch gradient step per
    # epoch converges (and can collapse policy entropy) much faster than
    # several smaller, noisier minibatch steps do; this was a real
    # contributor to an observed near-total entropy collapse within ~10
    # updates in an earlier run (see archive/run_2026-09-02_easy-plateau...).
    # nvtop showed GPU utilization pinned at ~4% during training -- the
    # bottleneck is the CPU-bound Node env-bridge simulation, not GPU
    # compute -- so there's some headroom to size minibatches up from the
    # original 64. Tried 256 first: it pushed this 6GB card's VRAM to
    # ~94.5% (5809/6144 MiB, nvidia-smi) with only ~330MB of headroom left
    # (risky against a transient spike, e.g. an eval episode's forward pass
    # running alongside a training update) and, contrary to the goal, made
    # each update *slower* (51.4s vs ~35s at minibatch=64) rather than
    # better using idle GPU time -- the CNN's conv layers are apparently
    # memory-bandwidth-bound enough at that size to lose more to memory
    # pressure than they gain from fewer/bigger steps. 128 is the settled
    # middle ground.
    p.add_argument("--minibatch-size", type=int, default=128)
    p.add_argument(
        "--curriculum", choices=sorted(LADDERS), default="difficulty",
        help="curriculum ladder (see curriculum.py's LADDERS): 'difficulty' (default) walks "
             "easy->impossible on the launch map pool; 'connected-first' starts on "
             "CONNECTED_MAPS and adds the island maps before raising the difficulty. "
             "Stored in checkpoints; a resume under a different ladder refuses to start",
    )
    p.add_argument(
        "--promote-threshold", type=float, default=0.75,
        help="eval-suite win rate at which the curriculum promotes to its next stage",
    )
    p.add_argument(
        "--expert-frac", default="1",
        help="probability that a training episode's opponent is the EXPERT -- the Nation AI at "
             "the curriculum stage's difficulty -- rather than a self-play copy of the agent "
             "(see selfplay.py). One number for every stage, or a comma-separated list with one "
             "entry per stage of --curriculum's ladder, so the mix can follow the curriculum. "
             "1 (default) is all-expert: no self-play, identical to before this option existed. "
             "Eval suites and promotion are always against the Nation AI",
    )
    p.add_argument(
        "--selfplay-pool-size", type=int, default=20,
        help="frozen snapshots of the agent kept as self-play opponents (oldest dropped first)",
    )
    p.add_argument(
        "--selfplay-snapshot-every", type=int, default=20,
        help="updates between snapshots added to the self-play pool",
    )
    p.add_argument(
        "--selfplay-latest-prob", type=float, default=0.5,
        help="probability a self-play episode's opponent is the live network rather than a "
             "uniformly drawn snapshot from the pool (always the live one while the pool is empty)",
    )
    p.add_argument("--checkpoint-dir", default=str(Path(__file__).parent / "checkpoints"))
    p.add_argument("--checkpoint-every", type=int, default=20, help="updates between checkpoints")
    p.add_argument("--eval-every", type=int, default=20, help="updates between eval suites")
    p.add_argument(
        "--eval-every-steps", type=int, default=0,
        help="decision steps per env between eval suites; overrides --eval-every when "
             "positive. Use it whenever the rollout length varies (see eval_due)",
    )
    p.add_argument(
        "--canary-every-steps", type=int, default=0,
        help="decision steps per env between CANARY evals: --canary-episodes games on a fixed, "
             "map-stratified subset of the stage's eval seeds, logged to --eval-log-csv as "
             "kind=canary with gates_curriculum=False -- never promotes. 0 (default) = off. "
             "Skipped on an update that runs a full suite",
    )
    p.add_argument(
        "--canary-episodes", type=int, default=0,
        help="games per canary eval (see --canary-every-steps); at most --eval-episodes",
    )
    p.add_argument(
        "--eval-report-exclude-maps", default="",
        help="comma-separated maps left out of eval_log.csv's wins_excl/episodes_excl columns "
             "(e.g. labyrinth, where nearly all of run4's hard wins came from). Reporting only: "
             "promotion still uses every game",
    )
    p.add_argument(
        "--eval-episodes", type=int, default=64,
        help="episodes per eval suite, per difficulty, played by the SAMPLED policy on the "
             "fixed seeds 0..N-1 (same maps, spawns and opponents every eval). The "
             "curriculum promotes on this suite's win rate",
    )
    p.add_argument(
        "--eval-envs", type=int, default=16,
        help="env workers an eval suite runs in parallel (one Node process each). A "
             "competent policy's games against easy average ~80k ticks (~8,000 steps), "
             "so suite wall time is roughly episodes/eval-envs x that",
    )
    p.add_argument(
        "--eval-difficulties", default="current",
        help="comma-separated difficulties to run a suite against at every eval, or "
             "'current' (default) for the curriculum's own. The curriculum's difficulty "
             "is always evaluated, since promotion needs it",
    )
    p.add_argument(
        "--eval-replays", type=int, default=1,
        help="dump a watchable GameRecord for the suite episodes with seed < this "
             "(per difficulty) into --eval-dir. Same seed means same gameID, so "
             "ReplayServer serves the newest file for it",
    )
    p.add_argument(
        "--eval-max-steps", type=int, default=0,
        help="step cap for eval episodes; 0 (default) uses --max-episode-steps. A "
             "capped episode counts as not won",
    )
    p.add_argument("--eval-dir", default=str(Path(__file__).parent / "replays"))
    p.add_argument("--log-csv", default=str(Path(__file__).parent / "checkpoints" / "train_log.csv"))
    p.add_argument(
        "--eval-log-csv", default=str(Path(__file__).parent / "checkpoints" / "eval_log.csv"),
        help="one row per (eval, difficulty) suite. Per-episode rows go to "
             "eval_episodes.csv in the same directory",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def map_pool_for(args: argparse.Namespace, maps: str | None) -> list[str] | None:
    """The map pool a curriculum stage plays on: the named MAP_POOLS entry, or
    for maps=None whatever the run was launched with -- --map if given (no
    pool), else SMALL_MAPS."""
    if maps is not None:
        return MAP_POOLS[maps]
    return None if args.map else SMALL_MAPS


def maps_label(args: argparse.Namespace, maps: str | None) -> str:
    """What map_pool_for() resolves to, for logs: a MAP_POOLS key, or
    map:<name> for a fixed --map."""
    if maps is not None:
        return maps
    return f"map:{args.map}" if args.map else "small"


def make_env(
    args: argparse.Namespace,
    difficulty: str,
    seed_prefix: str,
    maps: str | None = None,
    max_steps: int | None = None,
) -> OpenFrontEnv:
    return OpenFrontEnv(
        map_name=args.map or "onion",
        map_pool=map_pool_for(args, maps),
        seed=seed_prefix,
        difficulty=difficulty,
        ticks_per_step=args.ticks_per_step,
        max_steps=max_steps or args.max_episode_steps,
        # getattr: several probe/bench scripts build a bare Namespace for
        # make_env; they predate the option and mean the default reward.
        reward_mode=getattr(args, "reward_mode", REWARD_MODES[0]),
        potential_beta=getattr(args, "potential_beta", 0.0),
        attack_size_head=getattr(args, "attack_size_head", False),
    )


def save_checkpoint(
    path: Path, net, optimizer, scheduler: CurriculumScheduler, update: int,
    value_normalizer: ValueNormalizer | None = None, wall: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if value_normalizer is not None:
        # Only with --value-norm, so a default run's checkpoint has exactly
        # the keys it always had. Its presence is what load_checkpoint()
        # checks to refuse a resume that forgot --value-norm.
        extra["value_norm"] = value_normalizer.state_dict()
    if wall is not None:
        extra["wall_seconds"] = dict(wall)
    torch.save(
        {
            **extra,
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "curriculum_index": scheduler._index,
            # The index only means something within its ladder -- see
            # load_checkpoint().
            "curriculum_ladder": scheduler.ladder,
            # Promotion is gated on sampled eval suites (curriculum.py's
            # docstring). Rename this if what record_eval() is fed changes.
            "curriculum_mode": "eval_suite",
        },
        path,
    )


MILESTONE_DIFFICULTIES = ("medium", "hard", "impossible")
# Peak territorial-extent fractions (see territory_fraction()) worth a
# one-time notification, per difficulty -- ascending, so a single eval that
# jumps straight past more than one (e.g. 0% -> 45% in one eval) notifies
# every newly-crossed threshold, not just the highest.
TERRITORY_MILESTONES = (0.2, 0.4, 0.6)


def send_ntfy(message: str) -> None:
    """Fire-and-forget push notification via ntfy.sh, same topic/mechanism
    notify_on_finish.sh already uses for run-went-down alerts -- topic read
    from training/.ntfy_topic (sibling of this file), one unified
    notification stream rather than a second topic to track. Never raises:
    a notification failure (offline, ntfy.sh down) must not take training
    down with it."""
    topic_path = Path(__file__).parent / ".ntfy_topic"
    try:
        topic = topic_path.read_text().strip()
        if not topic:
            return
        subprocess.run(
            ["curl", "-s", "-m", "15", "-d", message, f"https://ntfy.sh/{topic}"],
            check=False, capture_output=True, timeout=20,
        )
    except Exception as e:  # noqa: BLE001 -- deliberately broad, see docstring
        print(f"  (ntfy notification failed, continuing: {e})")


def load_notified_state(checkpoint_dir: Path) -> dict:
    """{"wins": {difficulty, ...}, "territory": {difficulty: {frac, ...}}}
    -- tracks which one-time notifications have already fired, persisted
    alongside the checkpoint so a restart doesn't re-fire one. Backward
    compatible with the pre-territory-milestone format (a flat list of
    win-difficulties)."""
    path = checkpoint_dir / "notified_milestones.json"
    if not path.exists():
        return {"wins": set(), "territory": {}}
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return {"wins": set(raw), "territory": {}}
    return {
        "wins": set(raw.get("wins", [])),
        "territory": {k: set(v) for k, v in raw.get("territory", {}).items()},
    }


def save_notified_state(checkpoint_dir: Path, state: dict) -> None:
    path = checkpoint_dir / "notified_milestones.json"
    path.write_text(json.dumps({
        "wins": sorted(state["wins"]),
        "territory": {k: sorted(v) for k, v in state["territory"].items()},
    }))


def notify_milestones(notified: dict, checkpoint_dir: Path, difficulty: str, summary: dict, update: int) -> None:
    """One-time ntfy alerts for an eval suite: the first suite with any win
    against a milestone difficulty, and the first to reach each
    TERRITORY_MILESTONES peak share in any of its episodes. Per difficulty --
    20% against impossible is a much bigger deal than 20% against easy."""
    if summary["wins"] > 0 and difficulty in MILESTONE_DIFFICULTIES and difficulty not in notified["wins"]:
        notified["wins"].add(difficulty)
        save_notified_state(checkpoint_dir, notified)
        send_ntfy(
            f"OpenFront RL milestone: agent won {summary['wins']}/{summary['episodes']} sampled eval "
            f"games vs {difficulty} -- its first eval win there (update {update})."
        )
        print(f"  *** milestone: first eval win vs {difficulty} -- ntfy sent ***")
    # Ascending, so a suite that jumps past several thresholds at once
    # notifies each of them, not just the highest.
    territory_hit = notified["territory"].setdefault(difficulty, set())
    for threshold in TERRITORY_MILESTONES:
        if summary["max_peak_territory"] >= threshold and threshold not in territory_hit:
            territory_hit.add(threshold)
            save_notified_state(checkpoint_dir, notified)
            send_ntfy(
                f"OpenFront RL milestone: agent reached {threshold:.0%} peak territorial extent vs "
                f"{difficulty} for the first time (update {update}, best eval game "
                f"{summary['max_peak_territory']:.1%})."
            )
            print(f"  *** milestone: first {threshold:.0%} peak territory vs {difficulty} -- ntfy sent ***")


def check_value_norm(path: Path, ckpt: dict, value_normalizer: ValueNormalizer | None) -> None:
    """Restore --value-norm statistics, or refuse a mismatch. A checkpoint
    trained with them holds a value head whose outputs are NORMALIZED; read
    as raw values, every GAE target would be silently wrong."""
    if value_normalizer is None:
        if "value_norm" in ckpt:
            sys.exit(f"{path} was trained with --value-norm (its value head predicts normalized "
                     f"returns); resume it with --value-norm, via its launcher")
        return
    if "value_norm" in ckpt:
        value_normalizer.load_state_dict(ckpt["value_norm"])
        print(f"  value-norm statistics restored: mean={value_normalizer.mean:+.4f} std={value_normalizer.std:.4f}")
    else:
        print("  value-norm: checkpoint has no statistics -- starting from identity (mean 0, std 1), "
              "so the loaded value head is read exactly as before")


def load_checkpoint(
    path: Path, net, optimizer, scheduler: CurriculumScheduler,
    value_normalizer: ValueNormalizer | None = None, wall: dict | None = None,
) -> int:
    ckpt = torch.load(path, map_location=next(net.parameters()).device, weights_only=False)
    # A stage index from one ladder means a different (difficulty, maps) on
    # another, and hyperparameters are not stored in checkpoints -- so a
    # resume that forgets --curriculum (or a new launcher pointed at a live
    # run's checkpoints/) would silently continue at the wrong stage. Refuse.
    # Checkpoints predating ladders were all trained on "difficulty".
    ckpt_ladder = ckpt.get("curriculum_ladder", "difficulty")
    if ckpt_ladder != scheduler.ladder:
        sys.exit(
            f"{path} was trained under curriculum ladder {ckpt_ladder!r}, not {scheduler.ladder!r}. "
            f"Pass --curriculum {ckpt_ladder} to resume it, or archive it and start a new run."
        )
    has_head = any(k.startswith("attack_frac_head.") for k in ckpt["model"])
    if has_head != net.use_attack_size_head:
        sys.exit(f"{path} was trained {'with' if has_head else 'without'} --attack-size-head; resume it "
                 f"{'with' if has_head else 'without'} that flag, via its launcher")
    net.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    check_value_norm(path, ckpt, value_normalizer)
    if wall is not None:
        wall.update(ckpt.get("wall_seconds", {}))
    scheduler._index = ckpt["curriculum_index"]
    # Older checkpoints carry curriculum_results: single-episode outcomes
    # (greedy evals, or before that training episodes) from a window that no
    # longer exists. The stage itself is still valid and kept; the next eval
    # suite decides promotion from scratch.
    if ckpt.get("curriculum_mode") != "eval_suite" and ckpt.get("curriculum_results"):
        print("  (discarding the checkpoint's pre-eval-suite curriculum_results; stage kept)")
    return ckpt["update"]


# Parameters --init-from may find missing from the source checkpoint: those
# of modules a flag adds on top of the network it was trained with. Anything
# else missing (or unexpected) means a different architecture -- an error.
INIT_FROM_NEW_PREFIXES = ("attack_frac_head.",)


def init_from_checkpoint(
    path: Path, net: ActorCritic, optimizer, scheduler: CurriculumScheduler,
    value_normalizer: ValueNormalizer | None, make_optimizer,
) -> int:
    """--init-from: warm start this run from another run's checkpoint.

    Loads the weights (only INIT_FROM_NEW_PREFIXES may be absent from it), the
    curriculum position under the same ladder check as a resume, the update
    count -- so every update-keyed schedule (gamma, rollout length, eval
    cadence) continues where the source run was -- and the optimizer state of
    every matched parameter. Parameters new in this run get no optimizer
    state: AdamW starts their moments at zero and their step count at 0, as
    for a fresh parameter. Value-norm statistics come along if the source has
    them, else identity. Returns the update to start at."""
    ckpt = torch.load(path, map_location=next(net.parameters()).device, weights_only=False)
    ckpt_ladder = ckpt.get("curriculum_ladder", "difficulty")
    if ckpt_ladder != scheduler.ladder:
        sys.exit(f"--init-from {path}: trained under curriculum ladder {ckpt_ladder!r}, not "
                 f"{scheduler.ladder!r}. Pass --curriculum {ckpt_ladder}.")
    src_keys = set(ckpt["model"])
    own_keys = set(net.state_dict())
    missing = sorted(own_keys - src_keys)
    unexpected = sorted(src_keys - own_keys)
    bad_missing = [k for k in missing if not k.startswith(INIT_FROM_NEW_PREFIXES)]
    if bad_missing or unexpected:
        sys.exit(f"--init-from {path}: architecture mismatch beyond the new heads -- missing "
                 f"{bad_missing}, unexpected {unexpected}. Check --tile-head and the other "
                 f"architecture flags against the run it came from.")
    net.load_state_dict(ckpt["model"], strict=False)

    # The source optimizer's state is indexed by position within ITS param
    # groups, which shift once a new head is added. Rebuild the source
    # network's grouping (a network without the new modules), load the state
    # there -- which also validates it -- and move each entry across by
    # parameter name.
    src_net = ActorCritic(
        param_entropy_norm=net.param_entropy_norm, tile_head=net.tile_head,
        attack_size_head=any(k.startswith("attack_frac_head.") for k in src_keys),
    ).to(next(net.parameters()).device)
    src_net.load_state_dict(ckpt["model"])
    src_opt = make_optimizer(src_net)
    src_opt.load_state_dict(ckpt["optimizer"])
    src_params = dict(src_net.named_parameters())
    moved, fresh = 0, []
    for name, param in net.named_parameters():
        src = src_params.get(name)
        if src is not None and src in src_opt.state:
            optimizer.state[param] = {k: (v.clone() if torch.is_tensor(v) else v)
                                      for k, v in src_opt.state[src].items()}
            moved += 1
        else:
            fresh.append(name)
    del src_net, src_opt
    check_value_norm(path, ckpt, value_normalizer)
    scheduler._index = ckpt["curriculum_index"]
    print(f"  init-from: {len(src_keys)} tensors loaded, optimizer state moved for {moved} parameters; "
          f"new (no weights or optimizer state in the source): {fresh or 'none'}")
    return ckpt["update"]


def territory_fraction(obs: dict) -> float:
    """AGENT's share of the map's land, straight from tile_grid's own class
    proportions (0=neutral,1=self,2=opponent,3=water) rather than the
    self_tiles/opp_tiles raw counts -- those are real per-tile counts, but
    tile_grid may be the resized (map_pool) canvas (see openfront_env.py's
    RESIZE_DIM), a different unit system. Class *proportions* survive a
    nearest-neighbor resize essentially unchanged, so this ratio is valid
    whether or not the observation was resized, without needing to know
    which case applies."""
    grid = obs["tile_grid"]
    land = int(np.count_nonzero(grid != 3))
    if land == 0:
        return 0.0
    return int(np.count_nonzero(grid == 1)) / land


# Episode seeds of every eval suite are
# f"{EVAL_SEED_PREFIX}-{difficulty}-{maps}-{k}" for k in 0..--eval-episodes-1,
# at every eval: the seed fixes the map (drawn from the stage's pool by the
# env's seeded RNG), the spawns and the opponent, so two suites at the same
# stage differ only in the policy playing them. Difficulty and pool are in the
# prefix so a gameID (derived from the seed) names exactly one scenario --
# otherwise seed k's replays from different stages would share a gameID.
EVAL_SEED_PREFIX = "eval-suite"


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win rate. Unlike the normal
    approximation it stays inside [0, 1] and is still informative at 0/n and
    n/n, which is where most suites sit."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def eval_seed_maps(args: argparse.Namespace, maps: str | None, n: int) -> list[str]:
    """The map each eval seed 0..n-1 plays, without running anything: the
    env draws it from the pool with gymnasium's np_random seeded by the seed
    (OpenFrontEnv.reset). A fixed --map has one map for every seed."""
    pool = map_pool_for(args, maps)
    if not pool:
        return [args.map] * n
    return [str(seeding.np_random(k)[0].choice(pool)) for k in range(n)]


def canary_seeds(args: argparse.Namespace, maps: str | None) -> list[int]:
    """--canary-episodes seeds out of the stage's --eval-episodes eval seeds,
    stratified by map: round-robin over the maps in order of their first eval
    seed, taking each map's lowest unused seed. Fixed for a stage, so canary
    rows are comparable with each other (and each canary game is also a
    full-suite game, seed for seed)."""
    by_map: dict[str, list[int]] = {}
    for k, m in enumerate(eval_seed_maps(args, maps, args.eval_episodes)):
        by_map.setdefault(m, []).append(k)
    queues = list(by_map.values())
    picked: list[int] = []
    while len(picked) < args.canary_episodes:
        for q in queues:
            if q and len(picked) < args.canary_episodes:
                picked.append(q.pop(0))
    return sorted(picked)


def run_eval_suite(
    args: argparse.Namespace, net: ActorCritic, device: torch.device, difficulty: str, maps: str | None,
    seeds: list[int] | None = None, dump_replays: bool = True,
) -> list[dict]:
    """Plays --eval-episodes episodes on the fixed eval seeds with the SAMPLED
    policy -- the one training optimizes and the one training episodes are
    played with. The greedy (argmax) policy this replaced is a different
    policy: it turned the near-50/50 noop/expand split the agent holds at
    spawn into "never act", winning 3/198 evals while the sampled policy won
    ~24% of training episodes (see curriculum.py's docstring).

    Episodes run --eval-envs at a time, one batched forward per step across
    every live env, each worker taking the next pending seed as its episode
    ends. Construction is the same as training's (see GameSetup.ts), so the
    dumped GameRecords (--eval-replays) are faithful, replayable records.

    seeds: which of the suite's seeds to play (default all of 0..N-1) --
    the canary passes a subset. dump_replays=False skips --eval-replays.

    Returns one dict per episode, in seed order."""
    seeds = list(range(args.eval_episodes)) if seeds is None else list(seeds)
    n = len(seeds)
    max_steps = args.eval_max_steps or args.max_episode_steps
    prefix = f"{EVAL_SEED_PREFIX}-{difficulty}-{maps_label(args, maps)}"
    envs = [
        make_env(args, difficulty, prefix, maps=maps, max_steps=max_steps)
        for _ in range(min(args.eval_envs, n))
    ]
    pending = list(seeds)
    live: dict[int, dict] = {}  # worker index -> its episode in progress
    episodes: list[dict] = []

    def start(w: int) -> None:
        seed = pending.pop(0)
        env = envs[w]
        env.dump_game_record_dir = args.eval_dir if dump_replays and seed < args.eval_replays else None
        obs, info = env.reset(seed=seed)
        live[w] = {"seed": seed, "map": env.map_name, "obs": obs, "info": info,
                   "peak": territory_fraction(obs)}

    try:
        for w in range(len(envs)):
            start(w)
        while live:
            workers = sorted(live)
            batch = obs_to_batch([live[w]["obs"] for w in workers])
            action_mask = mask_batch([live[w]["info"] for w in workers])
            actions, _, _ = net.act(*net_inputs(batch, action_mask, device))
            actions = actions.cpu().numpy()
            for w, action in zip(workers, actions):
                envs[w].step_send(action)
            for w in workers:
                obs, _, term, trunc, info = envs[w].step_recv()
                ep = live[w]
                ep["obs"], ep["info"] = obs, info
                # Peak, not final -- an episode can reach a high territorial
                # share and then lose ground before ending.
                ep["peak"] = max(ep["peak"], territory_fraction(obs))
                if term or trunc:
                    episodes.append({
                        "seed": ep["seed"], "map": ep["map"], "winner": info["winner"],
                        "ticks": info["ticks"], "truncated": bool(trunc and not term),
                        "peak_territory_frac": ep["peak"],
                        "final_territory_frac": territory_fraction(obs),
                    })
                    del live[w]
                    if pending:
                        start(w)
    finally:
        for env in envs:
            env.close()
    return sorted(episodes, key=lambda e: e["seed"])


def summarize_suite(episodes: list[dict], exclude_maps: tuple[str, ...] = ()) -> dict:
    wins = sum(e["winner"] == "AGENT" for e in episodes)
    n = len(episodes)
    low, high = wilson_interval(wins, n)
    kept = [e for e in episodes if e["map"] not in exclude_maps]
    excl = {}
    if exclude_maps:
        excl = {"wins_excl": sum(e["winner"] == "AGENT" for e in kept), "episodes_excl": len(kept),
                "excluded_maps": "+".join(exclude_maps)}
    return {
        **excl,
        "episodes": n, "wins": wins, "win_rate": wins / n, "ci_low": low, "ci_high": high,
        "truncated": sum(e["truncated"] for e in episodes),
        "mean_peak_territory": float(np.mean([e["peak_territory_frac"] for e in episodes])),
        "max_peak_territory": float(max(e["peak_territory_frac"] for e in episodes)),
        "mean_ticks": float(np.mean([e["ticks"] for e in episodes])),
    }


def per_map_line(episodes: list[dict]) -> str:
    """'map wins/games' for every map in a suite, most games first."""
    counts: dict[str, list[int]] = {}
    for e in episodes:
        c = counts.setdefault(e["map"], [0, 0])
        c[0] += e["winner"] == "AGENT"
        c[1] += 1
    order = sorted(counts, key=lambda m: (-counts[m][1], m))
    return " ".join(f"{m} {counts[m][0]}/{counts[m][1]}" for m in order)


def open_csv(path: Path, columns: list[str], new_columns: tuple[str, ...] = ()) -> tuple:
    """Append-mode DictWriter, writing the header only for a new/empty file.
    Refuses an existing file with a different header rather than append rows
    that would be misaligned under it -- except that a file whose header is
    `columns` minus the trailing `new_columns` (written before they existed)
    keeps its own header, with a note. Returns (file, writer, dropped)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    dropped: list[str] = []
    if not is_new:
        with open(path, newline="") as fh:
            existing = next(csv.reader(fh), None)
        old = [c for c in columns if c not in new_columns]
        if existing == old and new_columns:
            dropped = list(new_columns)
            print(f"  NOTE: {path} predates columns {dropped}; not logging them to it.")
            columns = old
        elif existing != columns:
            sys.exit(f"{path} has columns {existing}, expected {columns} -- point the run at a new file")
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
    if is_new:
        writer.writeheader()
    return fh, writer, dropped


def main() -> None:
    args = parse_args()
    if args.gamma_start is not None:
        if args.gamma_anneal_updates <= 0:
            sys.exit("--gamma-start needs a positive --gamma-anneal-updates")
        if not (0.0 < args.gamma_start < 1.0 and 0.0 < args.gamma < 1.0):
            sys.exit("--gamma-start and --gamma must both lie strictly inside (0, 1)")
    if args.rollout_match_horizon and not 0.0 < args.gamma < 1.0:
        sys.exit("--rollout-match-horizon needs gamma strictly inside (0, 1)")
    if not math.isfinite(args.potential_beta) or args.potential_beta < 0:
        sys.exit("--potential-beta must be a finite number >= 0")
    if args.map and any(stage.maps is not None for stage in LADDERS[args.curriculum]):
        sys.exit(f"--curriculum {args.curriculum} chooses its own map pools; drop --map")
    if args.eval_episodes < 1 or args.eval_envs < 1:
        sys.exit("--eval-episodes and --eval-envs must be at least 1")
    if not 0.0 < args.promote_threshold <= 1.0:
        sys.exit("--promote-threshold must lie in (0, 1]")
    try:
        args.expert_fracs = parse_expert_frac(args.expert_frac, len(LADDERS[args.curriculum]))
    except ValueError as e:
        sys.exit(str(e))
    selfplay = any(f < 1.0 for f in args.expert_fracs)
    if args.selfplay_pool_size < 1 or args.selfplay_snapshot_every < 1:
        sys.exit("--selfplay-pool-size and --selfplay-snapshot-every must be at least 1")
    if not 0.0 <= args.selfplay_latest_prob <= 1.0:
        sys.exit("--selfplay-latest-prob must lie in [0, 1]")
    if args.canary_every_steps < 0 or args.canary_episodes < 0:
        sys.exit("--canary-every-steps and --canary-episodes must be >= 0")
    if (args.canary_every_steps > 0) != (args.canary_episodes > 0):
        sys.exit("--canary-every-steps and --canary-episodes go together: set both, or neither")
    if args.canary_episodes > args.eval_episodes:
        sys.exit("--canary-episodes must be at most --eval-episodes (canary seeds are a subset of the suite's)")
    args.exclude_maps = tuple(m for m in args.eval_report_exclude_maps.split(",") if m)
    unknown_maps = set(args.exclude_maps) - set(SMALL_MAPS)
    if unknown_maps:
        sys.exit(f"--eval-report-exclude-maps: unknown maps {sorted(unknown_maps)}")
    if args.eval_difficulties != "current":
        unknown = set(args.eval_difficulties.split(",")) - set(DIFFICULTIES)
        if unknown:
            sys.exit(f"--eval-difficulties: unknown {sorted(unknown)}, expected 'current' or a subset of {DIFFICULTIES}")
    # The replay is written by the Node env subprocess, which runs with
    # cwd=env-bridge/ -- a relative --eval-dir would land there, not where
    # it was typed. Resolve against Python's cwd before handing it over.
    args.eval_dir = str(Path(args.eval_dir).resolve())
    device = torch.device(args.device)
    if any(stage.maps is not None for stage in LADDERS[args.curriculum]):
        map_desc = "set per curriculum stage"
    else:
        map_desc = args.map if args.map else f"random from SMALL_MAPS ({len(SMALL_MAPS)} maps)"
    print(f"device={device} num_envs={args.num_envs} map={map_desc} reward_mode={args.reward_mode} potential_beta={args.potential_beta} "
          f"param_entropy_norm={args.param_entropy_norm} tile_head={args.tile_head}")
    print(f"ppo: kl_stop={args.kl_stop} target_kl={args.target_kl} adv_norm={args.adv_norm} "
          f"value_norm={args.value_norm} attack_size_head={args.attack_size_head}"
          + (f" (fractions {list(ATTACK_FRACTIONS)})" if args.attack_size_head else ""))
    if args.canary_every_steps > 0:
        print(f"canary: {args.canary_episodes} games every {args.canary_every_steps} steps/env (logged only)")
    if args.exclude_maps:
        print(f"eval report: wins_excl/episodes_excl leave out {list(args.exclude_maps)}")
    # The env-bridge's env-var gated behaviours are set by the launcher's
    # environment, not by a flag here, so record them in the log explicitly.
    # Every env's reset checks the Node worker really runs with them.
    flags = " ".join(f"{var}={int(on)}" for var, on in requested_env_flags().items())
    print(f"env flags: {flags}")
    ladder = " -> ".join(f"{s.difficulty}/{maps_label(args, s.maps)}" for s in LADDERS[args.curriculum])
    if selfplay:
        mix = ", ".join(f"{s.difficulty}/{maps_label(args, s.maps)}={f:g}"
                        for s, f in zip(LADDERS[args.curriculum], args.expert_fracs))
        print(f"self-play: expert (Nation AI) fraction per stage [{mix}]; pool of "
              f"{args.selfplay_pool_size} snapshots every {args.selfplay_snapshot_every} updates, "
              f"live-network opponent prob {args.selfplay_latest_prob}")
    else:
        print("self-play: off (every training episode is against the Nation AI)")
    print(f"curriculum {args.curriculum}: {ladder}; promote at sampled eval win rate >= {args.promote_threshold} "
          f"over {args.eval_episodes} fixed-seed episodes")
    if args.gamma_start is not None or args.rollout_match_horizon:
        end = args.gamma_anneal_start + args.gamma_anneal_updates
        peak_rollout = max(rollout_at(args.gamma_anneal_start, args), rollout_at(end, args))
        # tile_grid dominates the rollout buffer, and ppo_update's flatten
        # np.stack()s a second full copy -- hence x2. A 1500x1500 version of
        # this arithmetic is what OOM-killed training before (openfront_env.py).
        grid_gb = 2 * peak_rollout * args.num_envs * RESIZE_DIM * RESIZE_DIM / 1e9
        print(
            f"schedule: gamma {gamma_at(args.gamma_anneal_start, args):.4f} -> {gamma_at(end, args):.4f} "
            f"over updates {args.gamma_anneal_start}-{end}; rollout "
            f"{rollout_at(args.gamma_anneal_start, args)} -> {rollout_at(end, args)} steps; "
            f"peak tile_grid memory ~{grid_gb:.1f}GB"
        )

    scheduler = CurriculumScheduler(ladder=args.curriculum, promote_threshold=args.promote_threshold)

    def make_net() -> ActorCritic:
        return ActorCritic(param_entropy_norm=args.param_entropy_norm, tile_head=args.tile_head,
                           attack_size_head=args.attack_size_head)

    def make_optimizer(model: ActorCritic) -> torch.optim.AdamW:
        # AdamW rather than Adam so --weight-decay is *decoupled* (applied as
        # p -= lr*wd*p, not folded into the gradient and thus into Adam's second
        # moment, where an adaptive optimizer largely cancels it out again).
        # Biases are excluded: they carry none of the overfitting risk weight
        # decay targets, and shrinking them just drags the logits toward zero.
        # The network has no normalization layers, so >=2D vs 1D fully separates
        # "weights" from "biases" here -- revisit this split if one is ever added.
        decayed = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        not_decayed = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        return torch.optim.AdamW(
            [
                {"params": decayed, "weight_decay": args.weight_decay},
                {"params": not_decayed, "weight_decay": 0.0},
            ],
            lr=args.lr,
        )

    net = make_net().to(device)
    optimizer = make_optimizer(net)
    value_normalizer = ValueNormalizer(net.value_head) if args.value_norm else None
    # Cumulative wall time, training (rollout + update) vs evaluation (suites
    # + canaries), carried in checkpoints so the ratio survives resumes.
    wall = {"train": 0.0, "eval": 0.0}

    checkpoint_dir = Path(args.checkpoint_dir)
    latest_path = checkpoint_dir / "latest.pt"
    start_update = 0
    if latest_path.exists():
        start_update = load_checkpoint(latest_path, net, optimizer, scheduler, value_normalizer, wall)
        print(f"resumed from {latest_path} at update {start_update}, difficulty={scheduler.difficulty} "
              f"maps={maps_label(args, scheduler.maps)}"
              + (f" (--init-from {args.init_from} ignored: this run already has a checkpoint)"
                 if args.init_from else ""))
    elif args.init_from:
        start_update = init_from_checkpoint(
            Path(args.init_from), net, optimizer, scheduler, value_normalizer, make_optimizer
        )
        print(f"initialized from {args.init_from} at update {start_update}, difficulty={scheduler.difficulty} "
              f"maps={maps_label(args, scheduler.maps)} -- a new run in {checkpoint_dir}")

    # The self-play pool lives beside latest.pt, not inside it, so tools that
    # only read the policy (eval_checkpoint.py, the tests) never load it.
    pool_path = checkpoint_dir / POOL_FILENAME
    if pool_path.exists() and not selfplay:
        # Hyperparameters are not in checkpoints (see README): a self-play run
        # resumed without its --expert-frac would silently go all-expert.
        sys.exit(f"{pool_path} exists, so this run trained with self-play, but --expert-frac is all 1 "
                 f"here. Resume it via its launcher, or archive it and start a new run.")
    pool = OpponentPool(
        make_net=make_net,
        device=device, max_size=args.selfplay_pool_size, latest_prob=args.selfplay_latest_prob,
    )
    if selfplay and pool_path.exists():
        pool.load(pool_path)
        print(f"  self-play pool: {len(pool)} snapshots (updates {pool.keys()}) from {pool_path}")
    chooser = OpponentChooser(args.expert_fracs, lambda: scheduler._index, pool, args.num_envs)

    # Built after the checkpoint so the envs start on the resumed stage.
    vec_env = VecEnv(
        [
            (lambda i=i: make_env(args, scheduler.difficulty, seed_prefix=f"train-{i}", maps=scheduler.maps))
            for i in range(args.num_envs)
        ],
        before_reset=chooser if selfplay else None,
    )

    # First-time-beats-{medium,hard,impossible} and first-reaches-{20,40,60}%
    # territorial-extent milestone notifications -- persisted alongside the
    # checkpoint so a restart doesn't re-fire one already hit in a prior run.
    notified = load_notified_state(checkpoint_dir)

    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_columns = [
        "update", "difficulty", "win_rate", "mean_reward", "policy_loss",
        "value_loss", "entropy", "approx_kl", "stopped_early", "episodes", "seconds",
        "type_entropy", *[f"type_prob_{a}" for a in ACTIONS], "gamma", "rollout_length",
        "reward_mode", "potential_beta", "maps", "param_entropy_norm", "tile_head",
        "expert_frac", "selfplay_episodes", "selfplay_win_rate", "selfplay_pool_size",
        "kl_stop", "adv_norm", "value_norm", "attack_size_head", "grad_steps", "epochs_completed",
        "approx_kl_k3", "last_epoch_kl", "value_norm_mean", "value_norm_std",
        *[f"attack_frac_prob_{f:.2f}" for f in ATTACK_FRACTIONS],
        "train_seconds_total", "eval_seconds_total",
    ]
    log_is_new = not log_path.exists() or log_path.stat().st_size == 0
    if not log_is_new:
        # Appending rows wider than an existing log's header would leave every
        # later row misaligned. Keep writing that log's own schema instead,
        # and say so -- a varying gamma that goes unlogged is exactly the
        # silent-hyperparameter problem the run_*.sh launchers exist to stop.
        with open(log_path, newline="") as fh:
            existing = next(csv.reader(fh), None) or log_columns
        dropped = [c for c in log_columns if c not in existing]
        if dropped:
            print(f"  NOTE: {log_path} predates columns {dropped}; not logging them to it. "
                  f"Point --log-csv at a new file to record them.")
            if "gamma" in dropped and (args.gamma_start is not None or args.rollout_match_horizon):
                sys.exit(f"refusing to run a gamma/rollout schedule into {log_path}, "
                         f"which cannot record it -- pass a new --log-csv")
            if "reward_mode" in dropped and args.reward_mode != REWARD_MODES[0]:
                sys.exit(f"refusing to run reward mode {args.reward_mode!r} into {log_path}, "
                         f"which cannot record it -- pass a new --log-csv")
            if "potential_beta" in dropped and args.potential_beta != 0.0:
                sys.exit(f"refusing to run potential beta {args.potential_beta!r} into {log_path}, "
                         f"which cannot record it -- pass a new --log-csv")
            if "tile_head" in dropped and args.tile_head != "small":
                sys.exit(f"refusing to run --tile-head {args.tile_head} into {log_path}, "
                         f"which cannot record it -- pass a new --log-csv")
            if "param_entropy_norm" in dropped and args.param_entropy_norm:
                sys.exit(f"refusing to run --param-entropy-norm into {log_path}, "
                         f"which cannot record it -- pass a new --log-csv")
            if "selfplay_episodes" in dropped and selfplay:
                sys.exit(f"refusing to run self-play (--expert-frac {args.expert_frac}) into {log_path}, "
                         f"which cannot record it -- pass a new --log-csv")
            if "kl_stop" in dropped and (args.kl_stop != KL_STOP_MODES[0] or args.adv_norm != ADV_NORM_MODES[0]
                                         or args.value_norm or args.attack_size_head):
                sys.exit(f"refusing to run --kl-stop/--adv-norm/--value-norm/--attack-size-head into "
                         f"{log_path}, which cannot record them -- pass a new --log-csv")
            if "maps" in dropped and args.curriculum != "difficulty":
                sys.exit(f"refusing to run curriculum {args.curriculum!r}, whose stages change the map "
                         f"pool, into {log_path}, which cannot record it -- pass a new --log-csv")
        log_columns = existing
    log_file = open(log_path, "a", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_columns, extrasaction="ignore")
    if log_is_new:
        log_writer.writeheader()

    eval_log_path = Path(args.eval_log_csv)
    # kind: "suite" (a full --eval-episodes suite) or "canary". The other new
    # columns are filled only with --eval-report-exclude-maps.
    eval_new = ("kind", "wins_excl", "episodes_excl", "excluded_maps")
    eval_file, eval_writer, eval_dropped = open_csv(eval_log_path, [
        "update", "steps_per_env", "difficulty", "maps", "gates_curriculum", "episodes", "wins",
        "win_rate", "ci_low", "ci_high", "truncated", "mean_peak_territory", "max_peak_territory",
        "mean_ticks", "seconds", *eval_new,
    ], new_columns=eval_new)
    eval_episodes_file, eval_episodes_writer, episodes_dropped = open_csv(eval_log_path.parent / "eval_episodes.csv", [
        "update", "difficulty", "maps", "seed", "map", "winner", "ticks", "truncated",
        "peak_territory_frac", "final_territory_frac", "kind",
    ], new_columns=("kind",))
    if (eval_dropped or episodes_dropped) and (args.canary_every_steps > 0 or args.exclude_maps):
        sys.exit(f"refusing to run canaries or --eval-report-exclude-maps into {eval_log_path} (or its "
                 f"eval_episodes.csv), which cannot record them -- pass a new --eval-log-csv")

    obs_list, infos = vec_env.reset()
    batch = obs_to_batch(obs_list)
    action_mask = mask_batch(infos)

    update_iter = range(start_update, args.updates) if args.updates > 0 else itertools.count(start_update)
    steps_done = steps_before(start_update, args)
    for update in update_iter:
        t0 = time.time()
        gamma = gamma_at(update, args)
        rollout_length = rollout_at(update, args)
        buffer = RolloutBuffer()
        # True = AGENT won. Split by opponent: `episode_outcomes` (and the
        # logged win_rate/episodes) are Nation-AI episodes only, so they mean
        # what they meant before self-play existed.
        episode_outcomes: list[bool] = []
        selfplay_outcomes: list[bool] = []
        reward_sum = 0.0

        for _ in range(rollout_length):
            actions_t, log_probs_t, values_t = net.act(*net_inputs(batch, action_mask, device))
            actions_np = actions_t.cpu().numpy()
            # From `infos` of the step that produced `batch`: each self-play
            # slot's own-seat observation of the same state AGENT just saw.
            opp_actions = (
                opponent_actions(infos, chooser, net, obs_to_batch, net_inputs, device) if selfplay else None
            )

            next_obs_list, rewards, terms, truncs, infos = vec_env.step(actions_np, opp_actions)
            dones = terms | truncs
            reward_sum += float(rewards.sum())

            buffer.add(
                batch["tile_grid"],
                batch["self_features"],
                batch["opponent_features"],
                batch["opponent_mask"],
                action_mask,
                batch["boat_target_mask"],
                batch["attack_target_mask"],
                actions_np,
                log_probs_t.cpu().numpy(),
                values_t.cpu().numpy(),
                rewards,
                dones,
            )

            for info in infos:
                if "episode_info" in info:
                    ep = info["episode_info"]
                    outcomes = selfplay_outcomes if ep["opponent_kind"] == "policy" else episode_outcomes
                    outcomes.append(ep["winner"] == "AGENT")

            batch = obs_to_batch(next_obs_list)
            action_mask = mask_batch(infos)

        with torch.no_grad():
            inputs = net_inputs(batch, action_mask, device)
            _, _, _, last_values_t = net(*inputs)
            last_values = last_values_t.cpu().numpy()

            # Type-head-only entropy/probabilities -- see network.py's
            # type_diagnostics() docstring. Reuses the same post-rollout
            # batch as the bootstrap value above, so this is a free extra
            # forward pass, not an extra environment interaction.
            type_probs_t, type_entropy_t = net.type_diagnostics(*inputs)
            type_probs = type_probs_t.cpu().numpy()
            type_entropy = float(type_entropy_t.item())
            frac_probs_t = net.attack_frac_diagnostics(*inputs)
            frac_probs = (frac_probs_t.cpu().numpy() if frac_probs_t is not None
                          else np.full(len(ATTACK_FRACTIONS), np.nan))

        stats = ppo_update(
            net,
            optimizer,
            buffer,
            last_values,
            device,
            gamma=gamma,
            lam=args.gae_lambda,
            clip_eps=args.clip_eps,
            entropy_coef=args.entropy_coef,
            epochs=args.epochs,
            minibatch_size=args.minibatch_size,
            target_kl=args.target_kl,
            kl_stop=args.kl_stop,
            adv_norm=args.adv_norm,
            value_normalizer=value_normalizer,
        )
        # Released now rather than when the next update rebinds it: an eval
        # suite below stands up --eval-envs more Node workers, and at T=1000
        # this buffer's tile_grids alone are ~2.6GB.
        del buffer

        if selfplay and (update + 1) % args.selfplay_snapshot_every == 0:
            pool.add(update + 1, net)
            pool.prune_cache(chooser.in_use())

        win_rate = (sum(episode_outcomes) / len(episode_outcomes)) if episode_outcomes else float("nan")
        selfplay_win_rate = (
            sum(selfplay_outcomes) / len(selfplay_outcomes) if selfplay_outcomes else float("nan")
        )
        mean_reward = reward_sum / (rollout_length * args.num_envs)
        steps_prev, steps_done = steps_done, steps_done + rollout_length
        elapsed = time.time() - t0
        wall["train"] += elapsed
        early_flag = " EARLY-STOP" if stats["stopped_early"] else ""
        type_probs_str = " ".join(f"{a}={p:.2f}" for a, p in zip(ACTIONS, type_probs))
        print(
            f"update {update:5d} difficulty={scheduler.difficulty:10s} "
            f"episodes={len(episode_outcomes):3d} win_rate={win_rate:.2f} "
            f"mean_reward={mean_reward:+.4f} policy_loss={stats['policy_loss']:+.4f} "
            f"value_loss={stats['value_loss']:.4f} entropy={stats['entropy']:.3f} "
            f"kl={stats['approx_kl']:.4f} gamma={gamma:.4f} T={rollout_length} ({elapsed:.1f}s){early_flag}\n"
            f"  type_probs[{type_probs_str}] type_entropy={type_entropy:.3f}\n"
            f"  ppo: grad_steps={stats['grad_steps']} epochs={stats['epochs_completed']}/{args.epochs} "
            f"({stats['minibatches_per_epoch']} minibatches/epoch) kl_k3={stats['approx_kl_k3']:.4f}"
            + (f" value_norm mean={value_normalizer.mean:+.4f} std={value_normalizer.std:.4f}"
               if value_normalizer is not None else "")
            + (" attack_frac[" + " ".join(f"{f:.2f}={q:.2f}" for f, q in zip(ATTACK_FRACTIONS, frac_probs)) + "]"
               if args.attack_size_head else "")
            + (f"\n  self-play: expert_frac={chooser.expert_frac:g} episodes={len(selfplay_outcomes)} "
               f"win_rate={selfplay_win_rate:.2f} pool={len(pool)}" if selfplay else "")
        )
        log_writer.writerow(
            {
                "update": update, "difficulty": scheduler.difficulty, "win_rate": win_rate,
                "mean_reward": mean_reward, "policy_loss": stats["policy_loss"],
                "value_loss": stats["value_loss"], "entropy": stats["entropy"],
                "approx_kl": stats["approx_kl"], "stopped_early": stats["stopped_early"],
                "episodes": len(episode_outcomes), "seconds": elapsed,
                "type_entropy": type_entropy,
                **{f"type_prob_{a}": p for a, p in zip(ACTIONS, type_probs)},
                "gamma": gamma, "rollout_length": rollout_length,
                "reward_mode": args.reward_mode,
                "potential_beta": args.potential_beta,
                "maps": maps_label(args, scheduler.maps),
                "param_entropy_norm": args.param_entropy_norm,
                "tile_head": args.tile_head,
                "expert_frac": chooser.expert_frac,
                "selfplay_episodes": len(selfplay_outcomes),
                "selfplay_win_rate": selfplay_win_rate,
                "selfplay_pool_size": len(pool),
                "kl_stop": args.kl_stop, "adv_norm": args.adv_norm, "value_norm": args.value_norm,
                "attack_size_head": args.attack_size_head,
                "grad_steps": stats["grad_steps"], "epochs_completed": stats["epochs_completed"],
                "approx_kl_k3": stats["approx_kl_k3"], "last_epoch_kl": stats["last_epoch_kl"],
                "value_norm_mean": value_normalizer.mean if value_normalizer is not None else float("nan"),
                "value_norm_std": value_normalizer.std if value_normalizer is not None else float("nan"),
                **{f"attack_frac_prob_{f:.2f}": q for f, q in zip(ATTACK_FRACTIONS, frac_probs)},
                "train_seconds_total": wall["train"], "eval_seconds_total": wall["eval"],
            }
        )
        log_file.flush()

        if (update + 1) % args.checkpoint_every == 0:
            save_checkpoint(latest_path, net, optimizer, scheduler, update + 1, value_normalizer, wall)
            if selfplay:
                pool.save(pool_path)
            print(f"  checkpoint saved: {latest_path}")

        suite_due = eval_due(update, steps_prev, steps_done, args)
        if not suite_due and args.canary_every_steps > 0 and (
            steps_prev // args.canary_every_steps != steps_done // args.canary_every_steps
        ):
            # Logged only: never fed to the curriculum. Same sampled policy,
            # stage and seed construction as a full suite, on a fixed subset.
            canary_maps = scheduler.maps
            label = maps_label(args, canary_maps)
            t_eval = time.time()
            episodes = run_eval_suite(args, net, device, scheduler.difficulty, canary_maps,
                                      seeds=canary_seeds(args, canary_maps), dump_replays=False)
            seconds = time.time() - t_eval
            wall["eval"] += seconds
            summary = summarize_suite(episodes, args.exclude_maps)
            excl = (f" excl {'+'.join(args.exclude_maps)}: {summary['wins_excl']}/{summary['episodes_excl']}"
                    if args.exclude_maps else "")
            print(f"  canary vs {scheduler.difficulty} on {label}: won {summary['wins']}/{summary['episodes']}"
                  f"{excl} mean_peak={summary['mean_peak_territory']:.3f} ({seconds:.0f}s; eval "
                  f"{wall['eval'] / max(wall['eval'] + wall['train'], 1e-9):.1%} of wall time)")
            eval_writer.writerow({
                "update": update + 1, "steps_per_env": steps_done, "difficulty": scheduler.difficulty,
                "maps": label, "gates_curriculum": False, "seconds": seconds, "kind": "canary", **summary,
            })
            for e in episodes:
                eval_episodes_writer.writerow({"update": update + 1, "difficulty": scheduler.difficulty,
                                               "maps": label, "kind": "canary", **e})
            eval_file.flush()
            eval_episodes_file.flush()

        if suite_due:
            # Captured before record_eval() below, which can promote and
            # change the stage as a side effect -- everything logged for this
            # eval must describe the stage it actually ran against.
            stage_difficulty, stage_maps = scheduler.difficulty, scheduler.maps
            difficulties = [stage_difficulty]
            if args.eval_difficulties != "current":
                difficulties += [d for d in args.eval_difficulties.split(",") if d != stage_difficulty]
            gating_summary = None
            for difficulty in difficulties:
                t_eval = time.time()
                episodes = run_eval_suite(args, net, device, difficulty, stage_maps)
                summary = summarize_suite(episodes, args.exclude_maps)
                seconds = time.time() - t_eval
                wall["eval"] += seconds
                gates = difficulty == stage_difficulty
                if gates:
                    gating_summary = summary
                label = maps_label(args, stage_maps)
                print(
                    f"  eval-suite vs {difficulty} on {label}: won {summary['wins']}/{summary['episodes']} "
                    f"= {summary['win_rate']:.3f} [95% CI {summary['ci_low']:.3f}-{summary['ci_high']:.3f}] "
                    f"truncated={summary['truncated']} mean_peak={summary['mean_peak_territory']:.3f} "
                    f"mean_ticks={summary['mean_ticks']:.0f} ({seconds:.0f}s)"
                    + (" -- gates curriculum" if gates else "")
                    + (f"\n    excl {'+'.join(args.exclude_maps)}: won {summary['wins_excl']}/{summary['episodes_excl']}"
                       if args.exclude_maps else "")
                    + f"\n    per map: {per_map_line(episodes)}"
                    + f"\n    eval is {wall['eval'] / max(wall['eval'] + wall['train'], 1e-9):.1%} of wall time so far"
                )
                eval_writer.writerow({
                    "update": update + 1, "steps_per_env": steps_done, "difficulty": difficulty,
                    "maps": label, "gates_curriculum": gates, "seconds": seconds, "kind": "suite", **summary,
                })
                for e in episodes:
                    eval_episodes_writer.writerow({"update": update + 1, "difficulty": difficulty, "maps": label,
                                                   "kind": "suite", **e})
                eval_file.flush()
                eval_episodes_file.flush()
                notify_milestones(notified, checkpoint_dir, difficulty, summary, update + 1)

            if scheduler.record_eval(gating_summary["wins"], gating_summary["episodes"]) == "promoted":
                vec_env.set_difficulty(scheduler.difficulty)
                vec_env.set_map_pool(map_pool_for(args, scheduler.maps))
                stage = f"{scheduler.difficulty}/{maps_label(args, scheduler.maps)}"
                print(f"  curriculum promoted -> {stage}")
                # Promotion only ever moves forward (curriculum.py), so this
                # needs no persisted "already notified" state.
                send_ntfy(f"OpenFront RL curriculum: promoted to {stage} at update {update + 1}.")
            state = scheduler.state()
            print(
                f"  curriculum state: stage {state['index'] + 1}/{len(scheduler.stages)} "
                f"difficulty={state['difficulty']} maps={maps_label(args, state['maps'])} "
                f"last_eval_win_rate={state['last_eval_win_rate']}"
            )

    save_checkpoint(latest_path, net, optimizer, scheduler, args.updates, value_normalizer, wall)
    if selfplay:
        pool.save(pool_path)
    vec_env.close()
    log_file.close()
    eval_file.close()
    eval_episodes_file.close()


if __name__ == "__main__":
    main()
