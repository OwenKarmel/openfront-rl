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
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from curriculum import CurriculumScheduler
from envs.openfront_env import (
    ACTIONS,
    OPPONENT_TILES_IDX,
    SELF_TILES_IDX,
    SMALL_MAPS,
    OpenFrontEnv,
)
from envs.vector_env import VecEnv
from models.network import ActorCritic
from ppo import RolloutBuffer, ppo_update

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
    p.add_argument("--ticks-per-step", type=int, default=10)
    # Safety cap only -- see OpenFrontEnv's max_steps docstring. Episodes
    # (training and eval alike) should end via a real win/loss
    # (Episode.isDone()), not by running out of decision steps.
    p.add_argument("--max-episode-steps", type=int, default=20000)
    p.add_argument(
        "--updates", type=int, default=0,
        help="stop after this many updates; 0 (default) or negative runs indefinitely",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument(
        "--target-kl",
        type=float,
        default=0.03,
        help="abort the rest of a PPO update's epochs/minibatches if approx_kl exceeds 1.5x this",
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
        "--curriculum-window", type=int, default=12,
        help="number of eval (greedy) episodes the curriculum's rolling win rate is computed over",
    )
    p.add_argument("--checkpoint-dir", default=str(Path(__file__).parent / "checkpoints"))
    p.add_argument("--checkpoint-every", type=int, default=20, help="updates between checkpoints")
    p.add_argument("--eval-every", type=int, default=20, help="updates between eval episodes")
    p.add_argument("--eval-dir", default=str(Path(__file__).parent / "replays"))
    p.add_argument("--log-csv", default=str(Path(__file__).parent / "checkpoints" / "train_log.csv"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_env(args: argparse.Namespace, difficulty: str, seed_prefix: str) -> OpenFrontEnv:
    return OpenFrontEnv(
        map_name=args.map or "onion",
        map_pool=None if args.map else SMALL_MAPS,
        seed=seed_prefix,
        difficulty=difficulty,
        ticks_per_step=args.ticks_per_step,
        max_steps=args.max_episode_steps,
    )


def save_checkpoint(path: Path, net, optimizer, scheduler: CurriculumScheduler, update: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "curriculum_index": scheduler._index,
            "curriculum_results": list(scheduler._results),
            # Marks curriculum_results as eval-episode (greedy) outcomes --
            # see load_checkpoint()'s comment. Bump/rename this if the
            # semantics of what record_episode() is fed ever change again.
            "curriculum_mode": "eval",
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


def load_checkpoint(path: Path, net, optimizer, scheduler: CurriculumScheduler) -> int:
    ckpt = torch.load(path, map_location=next(net.parameters()).device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler._index = ckpt["curriculum_index"]
    scheduler._results.clear()
    # curriculum_results predating "curriculum_mode": "eval" (see
    # save_checkpoint()) holds stochastic training-rollout outcomes, not
    # eval ones -- a different, incompatible population from what the
    # window means now. Reinterpreting old stochastic win/losses as eval
    # results could misfire an early promotion on stale, wrong-kind data,
    # so those get discarded (fresh eval window) rather than loaded; the
    # difficulty level itself (curriculum_index) is still valid and kept.
    if ckpt.get("curriculum_mode") == "eval":
        scheduler._results.extend(ckpt["curriculum_results"])
    elif ckpt.get("curriculum_results"):
        print(
            "  (checkpoint's curriculum_results predate eval-based promotion -- "
            "discarding, starting a fresh eval window at the same difficulty)"
        )
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


def run_eval_episode(args: argparse.Namespace, net: ActorCritic, device: torch.device, difficulty: str, update: int) -> dict:
    """Greedy (argmax) rollout to completion, dumping a watchable/verifiable
    GameRecord -- the "evaluation routine" the project's design calls for:
    always the SAME construction training uses (see GameSetup.ts), so this
    is a genuinely faithful, desync-free record of how the current policy
    actually plays, not an approximation."""
    env = make_env(args, difficulty, seed_prefix=f"eval-{update}")
    env.dump_game_record_dir = args.eval_dir
    try:
        obs, info = env.reset(seed=update)
        terminated = truncated = False
        total_reward = 0.0
        peak_territory_frac = territory_fraction(obs)
        while not (terminated or truncated):
            inputs = net_inputs(obs_to_batch([obs]), info["action_mask"][None], device)
            with torch.no_grad():
                type_logits, tile_logits, player_logits, _ = net(*inputs)
                action_type = int(torch.argmax(type_logits, dim=-1).item())
                tile_idx = int(torch.argmax(tile_logits, dim=-1).item())
                player_idx = int(torch.argmax(player_logits, dim=-1).item())
            obs, reward, terminated, truncated, info = env.step([action_type, tile_idx, player_idx])
            total_reward += reward
            peak_territory_frac = max(peak_territory_frac, territory_fraction(obs))
        return {
            "winner": info["winner"],
            "ticks": info["ticks"],
            "total_reward": total_reward,
            "self_tiles": float(obs["self_features"][SELF_TILES_IDX]),
            # Summed across opponents -- one number for "how much land the
            # opposition holds", which stays comparable as the count changes.
            "opp_tiles": float(obs["opponent_features"][:, OPPONENT_TILES_IDX].sum()),
            # Peak, not final -- an episode can reach a high territorial
            # share and then lose ground before ending (see the peak-vs-
            # final distinction in analysis/territorial_extent/), and
            # "achieves 20% territorial extent" reads as an achievement
            # reached at any point, not a requirement to still hold it at
            # episode end.
            "peak_territory_frac": peak_territory_frac,
        }
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    map_desc = args.map if args.map else f"random from SMALL_MAPS ({len(SMALL_MAPS)} maps)"
    print(f"device={device} num_envs={args.num_envs} map={map_desc}")

    scheduler = CurriculumScheduler(window=args.curriculum_window)
    vec_env = VecEnv(
        [
            (lambda i=i: make_env(args, scheduler.difficulty, seed_prefix=f"train-{i}"))
            for i in range(args.num_envs)
        ]
    )
    vec_env.set_difficulty(scheduler.difficulty)

    net = ActorCritic().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    latest_path = checkpoint_dir / "latest.pt"
    start_update = 0
    if latest_path.exists():
        start_update = load_checkpoint(latest_path, net, optimizer, scheduler)
        vec_env.set_difficulty(scheduler.difficulty)
        print(f"resumed from {latest_path} at update {start_update}, difficulty={scheduler.difficulty}")

    # First-time-beats-{medium,hard,impossible} and first-reaches-{20,40,60}%
    # territorial-extent milestone notifications -- persisted alongside the
    # checkpoint so a restart doesn't re-fire one already hit in a prior run.
    notified = load_notified_state(checkpoint_dir)

    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_is_new = not log_path.exists()
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if log_is_new:
        log_writer.writerow(
            ["update", "difficulty", "win_rate", "mean_reward", "policy_loss",
             "value_loss", "entropy", "approx_kl", "stopped_early", "episodes", "seconds",
             "type_entropy", *[f"type_prob_{a}" for a in ACTIONS]]
        )

    obs_list, info_list = vec_env.reset()
    batch = obs_to_batch(obs_list)
    action_mask = mask_batch(info_list)

    update_iter = range(start_update, args.updates) if args.updates > 0 else itertools.count(start_update)
    for update in update_iter:
        t0 = time.time()
        buffer = RolloutBuffer()
        episode_outcomes: list[bool] = []  # True = AGENT won
        reward_sum = 0.0

        for _ in range(args.rollout_length):
            actions_t, log_probs_t, values_t = net.act(*net_inputs(batch, action_mask, device))
            actions_np = actions_t.cpu().numpy()

            next_obs_list, rewards, terms, truncs, infos = vec_env.step(actions_np)
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
                    episode_outcomes.append(info["episode_info"]["winner"] == "AGENT")

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

        stats = ppo_update(
            net,
            optimizer,
            buffer,
            last_values,
            device,
            gamma=args.gamma,
            lam=args.gae_lambda,
            clip_eps=args.clip_eps,
            entropy_coef=args.entropy_coef,
            epochs=args.epochs,
            minibatch_size=args.minibatch_size,
            target_kl=args.target_kl,
        )

        win_rate = (sum(episode_outcomes) / len(episode_outcomes)) if episode_outcomes else float("nan")
        mean_reward = reward_sum / (args.rollout_length * args.num_envs)
        elapsed = time.time() - t0
        early_flag = " EARLY-STOP" if stats["stopped_early"] else ""
        type_probs_str = " ".join(f"{a}={p:.2f}" for a, p in zip(ACTIONS, type_probs))
        print(
            f"update {update:5d} difficulty={scheduler.difficulty:10s} "
            f"episodes={len(episode_outcomes):3d} win_rate={win_rate:.2f} "
            f"mean_reward={mean_reward:+.4f} policy_loss={stats['policy_loss']:+.4f} "
            f"value_loss={stats['value_loss']:.4f} entropy={stats['entropy']:.3f} "
            f"kl={stats['approx_kl']:.4f} ({elapsed:.1f}s){early_flag}\n"
            f"  type_probs[{type_probs_str}] type_entropy={type_entropy:.3f}"
        )
        log_writer.writerow(
            [update, scheduler.difficulty, win_rate, mean_reward, stats["policy_loss"],
             stats["value_loss"], stats["entropy"], stats["approx_kl"], stats["stopped_early"],
             len(episode_outcomes), elapsed, type_entropy, *type_probs]
        )
        log_file.flush()

        if (update + 1) % args.checkpoint_every == 0:
            save_checkpoint(latest_path, net, optimizer, scheduler, update + 1)
            print(f"  checkpoint saved: {latest_path}")

        if (update + 1) % args.eval_every == 0:
            # Captured before record_episode() below, which can promote and
            # change scheduler.difficulty as a side effect -- both the
            # milestone check and the printed "eval vs X" line must refer to
            # the difficulty this particular eval was actually run against,
            # not whatever the scheduler ends up at afterward.
            eval_difficulty = scheduler.difficulty
            result = run_eval_episode(args, net, device, eval_difficulty, update + 1)
            print(f"  eval vs {eval_difficulty}: {result}")

            if (
                result["winner"] == "AGENT"
                and eval_difficulty in MILESTONE_DIFFICULTIES
                and eval_difficulty not in notified["wins"]
            ):
                notified["wins"].add(eval_difficulty)
                save_notified_state(checkpoint_dir, notified)
                send_ntfy(
                    f"OpenFront RL milestone: agent beat a {eval_difficulty} opponent "
                    f"for the first time (eval at update {update + 1}, ticks={result['ticks']})."
                )
                print(f"  *** milestone: first eval win vs {eval_difficulty} -- ntfy sent ***")

            # First-time-reaches-{20,40,60}% peak territorial extent,
            # per difficulty (20% against impossible is a much bigger deal
            # than 20% against easy, so these don't share one flag across
            # difficulties -- same reasoning as the per-difficulty win
            # milestones above). Loops ascending so an eval that jumps past
            # more than one threshold in one go still notifies each of them,
            # not just the highest.
            territory_hit = notified["territory"].setdefault(eval_difficulty, set())
            for threshold in TERRITORY_MILESTONES:
                if result["peak_territory_frac"] >= threshold and threshold not in territory_hit:
                    territory_hit.add(threshold)
                    save_notified_state(checkpoint_dir, notified)
                    send_ntfy(
                        f"OpenFront RL milestone: agent reached {threshold:.0%} peak territorial "
                        f"extent vs {eval_difficulty} for the first time (eval at update {update + 1}, "
                        f"peak={result['peak_territory_frac']:.1%})."
                    )
                    print(f"  *** milestone: first {threshold:.0%} peak territory vs {eval_difficulty} -- ntfy sent ***")

            # Curriculum promotion is driven by eval (greedy/deterministic)
            # outcomes, not the stochastic training-rollout episodes above --
            # see curriculum.py's docstring for why: the stochastic win rate
            # can look much better than the actual (deployed, watched)
            # greedy policy, especially early against a new difficulty, and
            # since promotion is one-way (no demotion), that gap matters a
            # lot more than it used to.
            curriculum_outcome = scheduler.record_episode(result["winner"] == "AGENT")
            if curriculum_outcome != "holding":
                vec_env.set_difficulty(scheduler.difficulty)
                print(f"  curriculum {curriculum_outcome} -> {scheduler.difficulty}")
                if curriculum_outcome == "promoted":
                    # Promotion is inherently one-time per difficulty (the
                    # scheduler only ever moves forward -- see
                    # curriculum.py), so this needs no separate persisted
                    # "already notified" state the way the win/territory
                    # milestones above do.
                    send_ntfy(
                        f"OpenFront RL curriculum: promoted to {scheduler.difficulty} "
                        f"at update {update + 1}."
                    )
            state = scheduler.state()
            print(
                f"  curriculum state: difficulty={state['difficulty']} "
                f"eval_win_rate={state['win_rate']} episodes_at_level={state['episodes_at_level']}/{scheduler.window}"
            )

    save_checkpoint(latest_path, net, optimizer, scheduler, args.updates)
    vec_env.close()
    log_file.close()


if __name__ == "__main__":
    main()
