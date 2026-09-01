"""PPO (clipped surrogate objective) -- rollout storage, GAE, and the
update step. Deliberately plain/textbook (Schulman et al. 2017): no
distributed training, no mixed precision, nothing exotic -- this project's
compute budget is one local GPU plus occasional Kaggle bursts, so the win
is in keeping the network small (see models/network.py) and iterating
fast, not in a sophisticated trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class RolloutBuffer:
    """Stores one rollout of T steps x N envs, as numpy arrays (H,W varies
    by map, so tile_grid is a python list of (N,H,W) arrays, one per step)."""

    tile_grids: list = field(default_factory=list)  # T x (N, H, W)
    scalars: list = field(default_factory=list)  # T x (N, 6)
    action_masks: list = field(default_factory=list)  # T x (N, num_actions)
    actions: list = field(default_factory=list)  # T x (N,)
    log_probs: list = field(default_factory=list)  # T x (N,)
    values: list = field(default_factory=list)  # T x (N,)
    rewards: list = field(default_factory=list)  # T x (N,)
    dones: list = field(default_factory=list)  # T x (N,) -- terminated OR truncated

    def add(self, tile_grid, scalars, action_mask, action, log_prob, value, reward, done):
        self.tile_grids.append(tile_grid)
        self.scalars.append(scalars)
        self.action_masks.append(action_mask)
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
) -> dict[str, float]:
    """One PPO update over a full rollout. Returns a dict of scalar stats for logging."""
    T, N = len(buffer), buffer.rewards[0].shape[0]

    rewards = np.stack(buffer.rewards)  # (T, N)
    values = np.stack(buffer.values)  # (T, N)
    dones = np.stack(buffer.dones).astype(np.float32)  # (T, N)
    advantages, returns = compute_gae(rewards, values, dones, last_values, gamma, lam)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Flatten (T, N, ...) -> (T*N, ...) for minibatching. tile_grid shape can
    # vary per-map but is constant within one training run, so a plain stack
    # is safe here.
    flat_tile_grid = np.stack(buffer.tile_grids).reshape(T * N, *buffer.tile_grids[0].shape[1:])
    flat_scalars = np.stack(buffer.scalars).reshape(T * N, -1)
    flat_masks = np.stack(buffer.action_masks).reshape(T * N, -1)
    flat_actions = np.stack(buffer.actions).reshape(T * N)
    flat_log_probs = np.stack(buffer.log_probs).reshape(T * N)
    flat_advantages = advantages.reshape(T * N)
    flat_returns = returns.reshape(T * N)
    flat_old_values = values.reshape(T * N)

    tile_grid_t = torch.as_tensor(flat_tile_grid, device=device)
    scalars_t = torch.as_tensor(flat_scalars, dtype=torch.float32, device=device)
    masks_t = torch.as_tensor(flat_masks, dtype=torch.bool, device=device)
    actions_t = torch.as_tensor(flat_actions, dtype=torch.long, device=device)
    old_log_probs_t = torch.as_tensor(flat_log_probs, dtype=torch.float32, device=device)
    advantages_t = torch.as_tensor(flat_advantages, dtype=torch.float32, device=device)
    returns_t = torch.as_tensor(flat_returns, dtype=torch.float32, device=device)
    old_values_t = torch.as_tensor(flat_old_values, dtype=torch.float32, device=device)

    num_samples = T * N
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "updates": 0}
    stopped_early = False

    for _ in range(epochs):
        if stopped_early:
            break
        perm = torch.randperm(num_samples, device=device)
        for start in range(0, num_samples, minibatch_size):
            idx = perm[start : start + minibatch_size]

            log_probs, entropy, value = net.evaluate_actions(
                tile_grid_t[idx], scalars_t[idx], masks_t[idx], actions_t[idx]
            )
            ratio = torch.exp(log_probs - old_log_probs_t[idx])
            adv = advantages_t[idx]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # PPO2-style value clipping: bound how far a single update can
            # move the value prediction from its collection-time estimate,
            # same spirit as the policy ratio clip -- prevents one large
            # value-target error (routine on a large, unbounded-tile-count
            # map before potential() was normalized -- see EnvServer.ts)
            # from producing an outsized gradient through the trunk the
            # policy head shares.
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

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy_loss.item()
            stats["approx_kl"] += approx_kl
            stats["updates"] += 1

            # Early-stop the whole update (remaining minibatches/epochs) if
            # the policy has already moved far outside a healthy trust
            # region -- confirmed empirically as a real failure mode: one
            # update's approx_kl spiking to ~0.27 (healthy PPO stays under
            # ~0.02-0.05) was directly followed by policy entropy collapsing
            # to ~0 a few updates later.
            if approx_kl > 1.5 * target_kl:
                stopped_early = True
                break

    n = max(stats["updates"], 1)
    return {
        "policy_loss": stats["policy_loss"] / n,
        "value_loss": stats["value_loss"] / n,
        "entropy": stats["entropy"] / n,
        "approx_kl": stats["approx_kl"] / n,
        "stopped_early": float(stopped_early),
    }
