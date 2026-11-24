"""PPO (clipped surrogate objective) -- rollout storage, GAE, and the
update step. Deliberately plain/textbook (Schulman et al. 2017): no
distributed training, no mixed precision, nothing exotic -- this project's
compute budget is one local GPU plus occasional Kaggle bursts, so the win
is in keeping the network small (see models/network.py) and iterating
fast, not in a sophisticated trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math

import numpy as np
import torch
import torch.nn as nn


@dataclass
class RolloutBuffer:
    """Stores one rollout of T steps x N envs, as numpy arrays (H,W varies
    by map, so tile_grid is a python list of (N,H,W) arrays, one per step)."""

    tile_grids: list = field(default_factory=list)  # T x (N, H, W)
    self_features: list = field(default_factory=list)  # T x (N, NUM_SELF_FEATURES)
    opponent_features: list = field(default_factory=list)  # T x (N, slots, NUM_OPPONENT_FEATURES)
    opponent_masks: list = field(default_factory=list)  # T x (N, slots)
    action_masks: list = field(default_factory=list)  # T x (N, num_actions)
    tile_masks: list = field(default_factory=list)  # T x (N, MACRO_GRID, MACRO_GRID)
    attack_masks: list = field(default_factory=list)  # T x (N, slots)
    # T x (N, 4) -- [action_type, macro_tile_idx, player_idx, attack_frac_idx].
    # Slot 3 is the attack-size head's (network.py); with the head off it
    # holds the default index and nothing reads it. There was no reserved
    # slot before it (the roadmap's "unit_type_idx slot" was never added), so
    # Phase 2's unit_type_idx needs a new slot 4 rather than this one.
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)  # T x (N,) -- already-composed scalar, see network.py
    values: list = field(default_factory=list)  # T x (N,)
    rewards: list = field(default_factory=list)  # T x (N,)
    dones: list = field(default_factory=list)  # T x (N,) -- terminated OR truncated

    def add(
        self, tile_grid, self_feat, opponent_feat, opponent_mask, action_mask,
        tile_mask, attack_mask, action, log_prob, value, reward, done,
    ):
        self.tile_grids.append(tile_grid)
        self.self_features.append(self_feat)
        self.opponent_features.append(opponent_feat)
        self.opponent_masks.append(opponent_mask)
        self.action_masks.append(action_mask)
        self.tile_masks.append(tile_mask)
        self.attack_masks.append(attack_mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.rewards)


def compute_gae(
    rewards: np.ndarray,  # (T, N)
    values: np.ndarray,  # (T, N)
    dones: np.ndarray,  # (T, N)
    last_values: np.ndarray,  # (N,) -- bootstrap value for the state after the last step
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation. Returns (advantages, returns), both (T, N)."""
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_value = last_values if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


KL_STOP_MODES = ("minibatch", "epoch")
ADV_NORM_MODES = ("batch", "minibatch")

# Debiased-EMA decay of ValueNormalizer's return statistics, per PPO update
# (one rollout). A time constant of ~100 updates: a single rollout's returns
# swing with whether it happened to contain a win, so the statistics should
# track the return distribution, not one rollout.
VALUE_NORM_BETA = 0.99
# Floor on the normalizer's std: the PopArt rescale divides the value head by
# it, so a near-constant batch of returns must not blow the head up.
VALUE_NORM_MIN_STD = 1e-2


class ValueNormalizer:
    """PopArt-style value-target normalization (van Hasselt et al. 2016),
    train.py's --value-norm.

    Keeps a debiased EMA of the mean and second moment of the GAE returns.
    The value head then predicts NORMALIZED values; ppo_update() denormalizes
    them for GAE and fits the head to normalized targets, so the value loss
    stays O(1) whatever the scale of the returns (which gamma 0.999 and the
    +/-1 terminal reward made large and drifting -- see the 2026-09-13
    archive's noop-collapse write-up).

    Output-preserving ("Preserving Outputs Precisely"): whenever the
    statistics move from (mu, sigma) to (mu', sigma'), the head's last layer
    is rescaled -- w' = w sigma/sigma', b' = (sigma b + mu - mu')/sigma' --
    so the DENORMALIZED prediction for every state is unchanged by the
    statistics update itself; only the gradient step changes it.

    The identity state (mean 0, std 1, nothing seen) is what a checkpoint
    without stored statistics loads as: the head's raw output is then read as
    a raw value, i.e. the warm-started critic is untouched. The first update
    moves the statistics to that rollout's returns (debiasing) and rescales
    the head to match.
    """

    def __init__(self, head: nn.Linear, beta: float = VALUE_NORM_BETA):
        self.head = head
        self.beta = beta
        self.m1 = 0.0  # EMA of E[R]
        self.m2 = 0.0  # EMA of E[R^2]
        self.w = 0.0  # EMA weight accumulated, for debiasing; 0 = identity

    @property
    def mean(self) -> float:
        return self.m1 / self.w if self.w > 0 else 0.0

    @property
    def std(self) -> float:
        if self.w == 0:
            return 1.0
        var = self.m2 / self.w - self.mean**2
        return max(math.sqrt(max(var, 0.0)), VALUE_NORM_MIN_STD)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    @torch.no_grad()
    def update(self, returns: np.ndarray) -> None:
        """Fold one rollout's returns into the statistics, then rescale the
        head so its denormalized output is exactly what it was."""
        old_mean, old_std = self.mean, self.std
        r = np.asarray(returns, dtype=np.float64)
        self.m1 = self.beta * self.m1 + (1 - self.beta) * float(r.mean())
        self.m2 = self.beta * self.m2 + (1 - self.beta) * float((r * r).mean())
        self.w = self.beta * self.w + (1 - self.beta)
        new_mean, new_std = self.mean, self.std
        self.head.weight.mul_(old_std / new_std)
        self.head.bias.mul_(old_std).add_(old_mean - new_mean).div_(new_std)

    def state_dict(self) -> dict:
        return {"m1": self.m1, "m2": self.m2, "w": self.w, "beta": self.beta}

    def load_state_dict(self, state: dict) -> None:
        self.m1, self.m2, self.w = float(state["m1"]), float(state["m2"]), float(state["w"])
        self.beta = float(state.get("beta", self.beta))


def minibatch_bounds(num_samples: int, minibatch_size: int, kl_stop: str) -> list[tuple[int, int]]:
    """[start, end) of each minibatch within one epoch's permutation.

    "minibatch" (the original behaviour) keeps a trailing partial minibatch,
    which can be a handful of samples -- with a per-minibatch KL check, its
    noisy estimate alone could end the update. "epoch" folds the remainder
    into the last full minibatch instead, so every gradient step sees at
    least minibatch_size samples and nothing is dropped."""
    if kl_stop == "minibatch":
        return [(s, min(s + minibatch_size, num_samples)) for s in range(0, num_samples, minibatch_size)]
    n = max(num_samples // minibatch_size, 1)
    bounds = [(i * minibatch_size, (i + 1) * minibatch_size) for i in range(n)]
    bounds[-1] = (bounds[-1][0], num_samples)
    return bounds


def ppo_update(
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    last_values: np.ndarray,
    device: torch.device,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    epochs: int = 4,
    minibatch_size: int = 256,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.03,
    kl_stop: str = "minibatch",
    adv_norm: str = "batch",
    value_normalizer: ValueNormalizer | None = None,
) -> dict[str, float]:
    """One PPO update over a full rollout. Returns a dict of scalar stats for logging.

    kl_stop -- when the KL early stop may end the update:
      "minibatch" (default, the original): after any minibatch whose k1
        estimate mean(old_logp - logp) exceeds 1.5 x target_kl, including a
        trailing partial one. Fired on 93-100% of updates in the runs up to
        run5 while mean approx_kl was 0.004-0.011 -- noise alone ends it.
      "epoch": the k3 estimator mean((ratio - 1) - log ratio) -- unbiased and
        always >= 0, unlike k1 -- averaged over a whole epoch, checked only
        BETWEEN epochs against target_kl itself. No partial minibatch.
    adv_norm -- "batch" (default): advantages normalized once over the whole
      rollout; "minibatch": per minibatch instead.
    value_normalizer -- None (default): the value head predicts raw returns.
      Otherwise its outputs are normalized values (see ValueNormalizer).

    Consumes buffer.tile_grids: each step's grid is copied to `device` and
    dropped from the list as it goes, so the rollout's dominant array is
    never held twice in host memory (np.stack()ing it was a second full copy,
    ~2.6GB at 10 envs x T=1000). The buffer is not usable afterwards."""
    if kl_stop not in KL_STOP_MODES:
        raise ValueError(f"kl_stop must be one of {KL_STOP_MODES}, got {kl_stop!r}")
    if adv_norm not in ADV_NORM_MODES:
        raise ValueError(f"adv_norm must be one of {ADV_NORM_MODES}, got {adv_norm!r}")
    T, N = len(buffer), buffer.rewards[0].shape[0]

    rewards = np.stack(buffer.rewards)  # (T, N)
    values = np.stack(buffer.values)  # (T, N) -- the head's outputs
    dones = np.stack(buffer.dones).astype(np.float32)  # (T, N)
    if value_normalizer is None:
        advantages, returns = compute_gae(rewards, values, dones, last_values, gamma, lam)
        value_targets, old_values = returns, values
    else:
        raw_values = value_normalizer.denormalize(values).astype(np.float32)
        raw_last = value_normalizer.denormalize(last_values).astype(np.float32)
        advantages, returns = compute_gae(rewards, raw_values, dones, raw_last, gamma, lam)
        # Statistics move, head rescales (outputs preserved); then targets
        # AND the collection-time values go to the NEW normalized space, which
        # is where the rescaled head now predicts -- so value clipping below
        # compares like with like.
        value_normalizer.update(returns)
        value_targets = value_normalizer.normalize(returns).astype(np.float32)
        old_values = value_normalizer.normalize(raw_values).astype(np.float32)
    if adv_norm == "batch":
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Flatten (T, N, ...) -> (T*N, ...) for minibatching. tile_grid shape can
    # vary per-map but is constant within one training run, so a plain stack
    # is safe here.
    def flatten(steps: list) -> np.ndarray:
        """T x (N, ...) -> (T*N, ...), preserving every trailing dim."""
        return np.stack(steps).reshape(T * N, *steps[0].shape[1:])

    # Row t*N + n is env n at step t, exactly as flatten() lays it out.
    first = torch.as_tensor(buffer.tile_grids[0])
    tile_grid_t = torch.empty((T * N, *first.shape[1:]), dtype=first.dtype, device=device)
    del first
    for t in range(T):
        tile_grid_t[t * N : (t + 1) * N] = torch.as_tensor(buffer.tile_grids[t], device=device)
        buffer.tile_grids[t] = None
    flat_self_features = flatten(buffer.self_features)
    flat_opponent_features = flatten(buffer.opponent_features)
    flat_opponent_masks = flatten(buffer.opponent_masks)
    flat_masks = flatten(buffer.action_masks)
    flat_tile_masks = flatten(buffer.tile_masks)
    flat_attack_masks = flatten(buffer.attack_masks)
    flat_actions = flatten(buffer.actions)
    flat_log_probs = np.stack(buffer.log_probs).reshape(T * N)
    flat_advantages = advantages.reshape(T * N)
    flat_returns = value_targets.reshape(T * N)
    flat_old_values = old_values.reshape(T * N)

    self_features_t = torch.as_tensor(flat_self_features, dtype=torch.float32, device=device)
    opponent_features_t = torch.as_tensor(
        flat_opponent_features, dtype=torch.float32, device=device
    )
    opponent_masks_t = torch.as_tensor(flat_opponent_masks, dtype=torch.bool, device=device)
    masks_t = torch.as_tensor(flat_masks, dtype=torch.bool, device=device)
    tile_masks_t = torch.as_tensor(flat_tile_masks, dtype=torch.bool, device=device)
    attack_masks_t = torch.as_tensor(flat_attack_masks, dtype=torch.bool, device=device)
    actions_t = torch.as_tensor(flat_actions, dtype=torch.long, device=device)
    old_log_probs_t = torch.as_tensor(flat_log_probs, dtype=torch.float32, device=device)
    advantages_t = torch.as_tensor(flat_advantages, dtype=torch.float32, device=device)
    returns_t = torch.as_tensor(flat_returns, dtype=torch.float32, device=device)
    old_values_t = torch.as_tensor(flat_old_values, dtype=torch.float32, device=device)

    num_samples = T * N
    bounds = minibatch_bounds(num_samples, minibatch_size, kl_stop)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
             "approx_kl_k3": 0.0, "updates": 0}
    stopped_early = False
    epochs_completed = 0
    last_epoch_kl = float("nan")

    for epoch in range(epochs):
        if stopped_early:
            break
        perm = torch.randperm(num_samples, device=device)
        epoch_kl_sum, epoch_kl_n = 0.0, 0
        for start, end in bounds:
            idx = perm[start:end]

            log_probs, entropy, value = net.evaluate_actions(
                tile_grid_t[idx],
                self_features_t[idx],
                opponent_features_t[idx],
                opponent_masks_t[idx],
                tile_masks_t[idx],
                attack_masks_t[idx],
                masks_t[idx],
                actions_t[idx],
            )
            ratio = torch.exp(log_probs - old_log_probs_t[idx])
            adv = advantages_t[idx]
            if adv_norm == "minibatch":
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # PPO2-style value clipping: bound how far a single update can
            # move the value prediction from its collection-time estimate,
            # same spirit as the policy ratio clip -- prevents one large
            # value-target error (routine on a large, unbounded-tile-count
            # map before potential() was normalized -- see EnvServer.ts)
            # from producing an outsized gradient through the trunk the
            # policy head shares. Under value normalization, everything here
            # (value, old values, targets, and so the clip width) is in
            # normalized units.
            value_clipped = old_values_t[idx] + torch.clamp(
                value - old_values_t[idx], -clip_eps, clip_eps
            )
            value_loss_unclipped = (value - returns_t[idx]) ** 2
            value_loss_clipped = (value_clipped - returns_t[idx]) ** 2
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
            entropy_loss = entropy.mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs_t[idx] - log_probs).mean().item()
                log_ratio = (log_probs - old_log_probs_t[idx]).detach()
                kl_k3 = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy_loss.item()
            stats["approx_kl"] += approx_kl
            stats["approx_kl_k3"] += kl_k3
            stats["updates"] += 1
            epoch_kl_sum += kl_k3 * (end - start)
            epoch_kl_n += end - start

            # Early-stop the whole update (remaining minibatches/epochs) if
            # the policy has already moved far outside a healthy trust
            # region -- confirmed empirically as a real failure mode: one
            # update's approx_kl spiking to ~0.27 (healthy PPO stays under
            # ~0.02-0.05) was directly followed by policy entropy collapsing
            # to ~0 a few updates later.
            if kl_stop == "minibatch" and approx_kl > 1.5 * target_kl:
                stopped_early = True
                break
        else:
            epochs_completed += 1
            last_epoch_kl = epoch_kl_sum / epoch_kl_n
            # The same trust-region guard, on a whole epoch's k3 mean instead
            # of one minibatch's k1, and only between epochs. After the last
            # epoch there is nothing left to stop.
            if kl_stop == "epoch" and last_epoch_kl > target_kl and epoch < epochs - 1:
                stopped_early = True

    n = max(stats["updates"], 1)
    return {
        "policy_loss": stats["policy_loss"] / n,
        "value_loss": stats["value_loss"] / n,
        "entropy": stats["entropy"] / n,
        "approx_kl": stats["approx_kl"] / n,
        "approx_kl_k3": stats["approx_kl_k3"] / n,
        "last_epoch_kl": last_epoch_kl,
        "stopped_early": float(stopped_early),
        "grad_steps": stats["updates"],
        "epochs_completed": epochs_completed,
        "minibatches_per_epoch": len(bounds),
    }
