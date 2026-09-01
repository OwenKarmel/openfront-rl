"""Actor-critic network for OpenFrontEnv.

Small by design -- the compute budget is one local GTX 1660 (6GB) plus
occasional Kaggle GPU bursts, not a cluster. A CNN over the tile-ownership
grid (one-hot: neutral/self/opponent) reduced through strided convs to a
fixed-size feature vector (AdaptiveAvgPool2d, so it works across different
map sizes without changing the architecture), concatenated with a small MLP
over the six scalar player-stat features (tiles/troops/gold for both
sides), feeding shared trunk -> policy head (masked over ACTIONS) + value
head. Total params are a few hundred thousand, not millions -- this is
meant to train fast on modest hardware, not to be state-of-the-art.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ACTIONS = 3  # noop, expand, attack_opponent -- see ACTIONS in openfront_env.py
NUM_SCALAR_FEATURES = 6  # self_tiles, self_troops, self_gold, opp_tiles, opp_troops, opp_gold


class ActorCritic(nn.Module):
    def __init__(self, conv_channels: int = 32, trunk_dim: int = 256):
        super().__init__()
        # 3 input channels: one-hot(neutral, self, opponent). Four stride-2
        # convs roughly halve spatial size each time (e.g. 512 -> 32),
        # then AdaptiveAvgPool2d fixes the output size regardless of the
        # map's actual dimensions.
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        conv_out_dim = conv_channels * 4 * 4

        self.scalar_mlp = nn.Sequential(
            nn.Linear(NUM_SCALAR_FEATURES, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        self.trunk = nn.Sequential(
            nn.Linear(conv_out_dim + 64, trunk_dim),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(trunk_dim, NUM_ACTIONS)
        self.value_head = nn.Linear(trunk_dim, 1)

    def _features(self, tile_grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """tile_grid: (B, H, W) int8/long in {0,1,2}. scalars: (B, 6) float."""
        one_hot = F.one_hot(tile_grid.long(), num_classes=3).permute(0, 3, 1, 2).float()
        conv_feat = self.conv(one_hot).flatten(1)
        # log1p compresses the huge dynamic range of troops/gold (can run into
        # the tens/hundreds of thousands) into something a small MLP can use.
        scalar_feat = self.scalar_mlp(torch.log1p(scalars.clamp(min=0)))
        return self.trunk(torch.cat([conv_feat, scalar_feat], dim=1))

    def forward(self, tile_grid: torch.Tensor, scalars: torch.Tensor, action_mask: torch.Tensor):
        """Returns (logits, value). action_mask: (B, NUM_ACTIONS) bool, True = legal."""
        feat = self._features(tile_grid, scalars)
        logits = self.policy_head(feat)
        logits = logits.masked_fill(~action_mask, -1e9)
        value = self.value_head(feat).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, tile_grid: torch.Tensor, scalars: torch.Tensor, action_mask: torch.Tensor):
        """Sample an action for rollout collection. Returns (action, log_prob, value)."""
        logits, value = self.forward(tile_grid, scalars, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(
        self,
        tile_grid: torch.Tensor,
        scalars: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
    ):
        """Used during PPO updates: log_prob/entropy of given actions, plus value."""
        logits, value = self.forward(tile_grid, scalars, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
