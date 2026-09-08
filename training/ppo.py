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
    self_features: list = field(default_factory=list)  # T x (N, NUM_SELF_FEATURES)
    opponent_features: list = field(default_factory=list)  # T x (N, slots, NUM_OPPONENT_FEATURES)
    opponent_masks: list = field(default_factory=list)  # T x (N, slots)
    action_masks: list = field(default_factory=list)  # T x (N, num_actions)
    tile_masks: list = field(default_factory=list)  # T x (N, MACRO_GRID, MACRO_GRID)
    attack_masks: list = field(default_factory=list)  # T x (N, slots)
    # T x (N, 3) -- [action_type, macro_tile_idx, player_idx]
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
    def flatten(steps: list) -> np.ndarray:
        """T x (N, ...) -> (T*N, ...), preserving every trailing dim."""
        return np.stack(steps).reshape(T * N, *steps[0].shape[1:])

    flat_tile_grid = flatten(buffer.tile_grids)
    flat_self_features = flatten(buffer.self_features)
    flat_opponent_features = flatten(buffer.opponent_features)
    flat_opponent_masks = flatten(buffer.opponent_masks)
    flat_masks = flatten(buffer.action_masks)
    flat_tile_masks = flatten(buffer.tile_masks)
    flat_attack_masks = flatten(buffer.attack_masks)
    flat_actions = flatten(buffer.actions)
    flat_log_probs = np.stack(buffer.log_probs).reshape(T * N)
    flat_advantages = advantages.reshape(T * N)
    flat_returns = returns.reshape(T * N)
    flat_old_values = values.reshape(T * N)

    tile_grid_t = torch.as_tensor(flat_tile_grid, device=device)
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
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "updates": 0}
    stopped_early = False

    for _ in range(epochs):
        if stopped_early:
            break
        perm = torch.randperm(num_samples, device=device)
        for start in range(0, num_samples, minibatch_size):
            idx = perm[start : start + minibatch_size]

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
