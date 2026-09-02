"""Actor-critic network for OpenFrontEnv.

Small by design -- the compute budget is one local GTX 1660 (6GB) plus
occasional Kaggle GPU bursts, not a cluster. A CNN over the tile-ownership
grid (one-hot: neutral/self/opponent/water) reduced through strided convs to
a fixed-size feature vector (AdaptiveAvgPool2d, so it works across different
map sizes without changing the architecture), concatenated with a small MLP
over the six scalar player-stat features (tiles/troops/gold for both
sides), feeding a shared trunk -> action-type head (masked over ACTIONS) +
value head. Total params are a few hundred thousand, not millions -- this
is meant to train fast on modest hardware, not to be state-of-the-art.

Action factorization: as of the naval/boat_attack action, this is no longer
a single flat categorical. There's a small action-TYPE head (as before) plus
one PARAMETER head (a spatial tile-logit map) that only matters when the
sampled type is boat_attack. log_prob/entropy for a transition are the sum
of the type term and (only when boat_attack was the sampled type) the tile
term -- a "masked sum" composition that keeps log_prob/entropy scalar-per-
transition exactly as before, so ppo.py's GAE/clip/entropy-bonus math needs
no changes at all. See the project's action-space-expansion plan for the
full design rationale; this is deliberately the minimal version of it (just
type + tile) rather than pre-building unused unit-type/player-index heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ACTIONS = 4  # noop, expand, attack_opponent, boat_attack -- see ACTIONS in openfront_env.py
BOAT_ATTACK_IDX = 3  # must match ACTIONS.index("boat_attack") in openfront_env.py
NUM_SCALAR_FEATURES = 6  # self_tiles, self_troops, self_gold, opp_tiles, opp_troops, opp_gold
NUM_TILE_CLASSES = 4  # neutral land, self, opponent, water -- see tileGrid() in EnvServer.ts
MACRO_GRID = 32  # must match MACRO_GRID in EnvServer.ts / openfront_env.py


class ActorCritic(nn.Module):
    def __init__(self, conv_channels: int = 32, trunk_dim: int = 256):
        super().__init__()
        # Four stride-2 convs roughly halve spatial size each time (e.g.
        # 512 -> 32). Split into a body (shared by both the pooled trunk
        # path and the spatial tile-head path) and the pooling step, since
        # the tile head needs the pre-pool feature map -- AdaptiveAvgPool2d
        # collapses (x,y) structure entirely, which is fine for the type/
        # value heads but would make a spatial action unrepresentable.
        self.conv_body = nn.Sequential(
            nn.Conv2d(NUM_TILE_CLASSES, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)
        conv_out_dim = conv_channels * 4 * 4

        # 1x1 conv -> one logit per cell of the pre-pool feature map, then
        # adaptive-pooled to a fixed MACRO_GRID x MACRO_GRID regardless of
        # the feature map's actual size (which varies with the real map's
        # pixel dimensions) -- same size-invariance trick as `pool` above,
        # just producing a spatial map instead of a single vector.
        self.tile_head_conv = nn.Conv2d(conv_channels, 1, kernel_size=1)

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
        self.type_head = nn.Linear(trunk_dim, NUM_ACTIONS)
        self.value_head = nn.Linear(trunk_dim, 1)

    def _features(self, tile_grid: torch.Tensor, scalars: torch.Tensor):
        """tile_grid: (B, H, W) int8/long in {0,1,2,3}. scalars: (B, 6) float.
        Returns (trunk_feat, feat_map) -- feat_map is the pre-pool conv
        output, needed by the tile head."""
        one_hot = F.one_hot(tile_grid.long(), num_classes=NUM_TILE_CLASSES).permute(0, 3, 1, 2).float()
        feat_map = self.conv_body(one_hot)
        conv_feat = self.pool(feat_map).flatten(1)
        # log1p compresses the huge dynamic range of troops/gold (can run into
        # the tens/hundreds of thousands) into something a small MLP can use.
        scalar_feat = self.scalar_mlp(torch.log1p(scalars.clamp(min=0)))
        trunk_feat = self.trunk(torch.cat([conv_feat, scalar_feat], dim=1))
        return trunk_feat, feat_map

    def forward(
        self,
        tile_grid: torch.Tensor,
        scalars: torch.Tensor,
        tile_target_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        """Returns (type_logits, tile_logits, value).
        action_mask: (B, NUM_ACTIONS) bool, True = legal action type.
        tile_target_mask: (B, MACRO_GRID, MACRO_GRID) bool, True = legal
        boat_attack destination (only meaningful when boat_attack is legal;
        otherwise ignored downstream by the type/tile log_prob composition
        in act()/evaluate_actions())."""
        trunk_feat, feat_map = self._features(tile_grid, scalars)

        type_logits = self.type_head(trunk_feat)
        type_logits = type_logits.masked_fill(~action_mask, -1e9)

        tile_logits = self.tile_head_conv(feat_map)  # (B, 1, H', W')
        tile_logits = F.adaptive_avg_pool2d(tile_logits, MACRO_GRID).flatten(1)  # (B, MACRO_GRID^2)
        tile_mask_flat = tile_target_mask.reshape(tile_logits.shape[0], -1).bool()
        tile_logits = tile_logits.masked_fill(~tile_mask_flat, -1e9)

        value = self.value_head(trunk_feat).squeeze(-1)
        return type_logits, tile_logits, value

    @torch.no_grad()
    def act(
        self,
        tile_grid: torch.Tensor,
        scalars: torch.Tensor,
        tile_target_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        """Sample an action for rollout collection.
        Returns (action, log_prob, value) -- action is (B, 2) int:
        [action_type, macro_tile_idx]. macro_tile_idx is only meaningful
        (and only contributes to log_prob) when action_type == boat_attack;
        otherwise it's still a validly-sampled index (harmless, ignored
        server-side and by the composed log_prob/entropy below)."""
        type_logits, tile_logits, value = self.forward(tile_grid, scalars, tile_target_mask, action_mask)
        type_dist = torch.distributions.Categorical(logits=type_logits)
        tile_dist = torch.distributions.Categorical(logits=tile_logits)
        action_type = type_dist.sample()
        tile_idx = tile_dist.sample()

        is_boat = (action_type == BOAT_ATTACK_IDX).float()
        log_prob = type_dist.log_prob(action_type) + is_boat * tile_dist.log_prob(tile_idx)

        action = torch.stack([action_type, tile_idx], dim=-1)
        return action, log_prob, value

    def evaluate_actions(
        self,
        tile_grid: torch.Tensor,
        scalars: torch.Tensor,
        tile_target_mask: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
    ):
        """Used during PPO updates: log_prob/entropy of given actions, plus
        value. actions: (B, 2) int, [action_type, macro_tile_idx]."""
        action_type = actions[:, 0]
        tile_idx = actions[:, 1]
        type_logits, tile_logits, value = self.forward(tile_grid, scalars, tile_target_mask, action_mask)
        type_dist = torch.distributions.Categorical(logits=type_logits)
        tile_dist = torch.distributions.Categorical(logits=tile_logits)

        is_boat = (action_type == BOAT_ATTACK_IDX).float()
        log_prob = type_dist.log_prob(action_type) + is_boat * tile_dist.log_prob(tile_idx)
        entropy = type_dist.entropy() + is_boat * tile_dist.entropy()
        return log_prob, entropy, value
