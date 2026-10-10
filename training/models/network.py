"""Actor-critic network for OpenFrontEnv.

Small by design -- the compute budget is one local GTX 1660 (6GB) plus
occasional Kaggle GPU bursts, not a cluster. A CNN over the tile-ownership
grid (one-hot: neutral/self/ANY-opponent/water) reduced through strided
convs to a fixed-size feature vector (AdaptiveAvgPool2d, so it works across
different map sizes without changing the architecture), concatenated with a
small MLP over AGENT's own scalar stats and with a permutation-invariant
summary of the opponent set (below), feeding a shared trunk -> action-type
head (masked over ACTIONS) + value head. Total params are a few hundred
thousand, not millions -- this is meant to train fast on modest hardware,
not to be state-of-the-art.

Variable opponent count (DeepSets): opponents arrive as a padded, masked
array of slots rather than a flat opp_tiles/opp_troops/opp_gold triple. A
single shared MLP (`opponent_mlp`, phi) embeds each slot independently, and
those embeddings are masked and SUM-pooled into one fixed-size vector. Because
the same weights apply to every slot and sum-pooling ignores order, the
network is permutation-invariant and agnostic to how many opponents an
episode actually has -- no weight shape anywhere depends on the slot count,
so raising it needs no architectural change (Zaheer et al.'s DeepSets
construction). This matters because slot assignment is arbitrary and the
count varies; the network literally cannot learn "slot #3 is special."

Action factorization: this is not a single flat categorical. There's a small
action-TYPE head plus two PARAMETER heads, each of which only matters for
one type: a spatial tile-logit map (boat_attack's destination) and a pointer
head over opponent slots (attack_opponent's target). The pointer head reads
the PRE-pool per-slot embeddings -- the pooled summary has already discarded
which slot is which, so it cannot address one. log_prob/entropy for a
transition are the sum of the type term plus whichever parameter term the
sampled type actually used -- a "masked sum" composition that keeps both
scalar-per-transition, so ppo.py's GAE/clip/entropy-bonus math needs no
changes at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ACTIONS = 4  # noop, expand, attack_opponent, boat_attack -- see ACTIONS in openfront_env.py
BOAT_ATTACK_IDX = 3  # must match ACTIONS.index("boat_attack") in openfront_env.py
ATTACK_OPPONENT_IDX = 2  # must match ACTIONS.index("attack_opponent") in openfront_env.py
NUM_SELF_FEATURES = 4  # tiles, troops, troops_ratio, gold -- see SELF_FEATURES in openfront_env.py
# alive, tiles, troops, troops_ratio, gold, shares_border -- see
# OPPONENT_FEATURES in openfront_env.py. This is the per-slot feature width,
# NOT the number of slots: nothing here depends on how many opponents there
# are, which is what makes the set encoder count-agnostic.
NUM_OPPONENT_FEATURES = 6
NUM_TILE_CLASSES = 4  # neutral land, self, ANY opponent, water -- see tileGrid() in EnvServer.ts
MACRO_GRID = 32  # must match MACRO_GRID in EnvServer.ts / openfront_env.py


class BigTileHead(nn.Module):
    """boat_attack's target head with global context and a whole-map view.

    The default head is one 1x1 conv: each cell's logit sees only that cell's
    local CNN features, nothing about the rest of the map or the game state.
    Measured on the connected-first runs, it never learned to aim -- 95-97%
    of its maximum entropy after 600+ updates, with or without the entropy
    bonus on it (--param-entropy-norm made no difference). Picking a landing
    needs both what is where (unclaimed land, own coast, the opponent) and
    the global state (troops, territory), so this head gets:

      - per cell: the CNN features resampled to the MACRO_GRID target grid,
        the legal-target mask, and x/y coordinates
      - global: the trunk's game-state vector, as a per-channel FiLM
        scale/shift on every cell
      - reach: residual blocks of dilated 3x3 convs (dilations 1,2,4,8 -> a
        61-cell receptive field), so every cell sees the whole 32x32 grid

    The output conv is zero-initialized, so it starts uniform over legal
    cells like the default head does."""

    def __init__(self, in_channels: int, trunk_dim: int, width: int = 64, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.inp = nn.Conv2d(in_channels + 3, width, kernel_size=1)  # + mask, x, y
        self.film = nn.Linear(trunk_dim, 2 * width)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, width, 3, padding=d, dilation=d),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=d, dilation=d),
            )
            for d in dilations
        )
        self.out = nn.Conv2d(width, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        ramp = torch.linspace(-1.0, 1.0, MACRO_GRID)
        self.register_buffer("coords", torch.stack(torch.meshgrid(ramp, ramp, indexing="ij")), persistent=False)

    def forward(self, feat_map: torch.Tensor, trunk_feat: torch.Tensor, tile_target_mask: torch.Tensor):
        """-> (B, MACRO_GRID^2) unmasked logits."""
        b = feat_map.shape[0]
        x = F.adaptive_avg_pool2d(feat_map, MACRO_GRID)
        mask = tile_target_mask.reshape(b, 1, MACRO_GRID, MACRO_GRID).float()
        h = F.relu(self.inp(torch.cat([x, mask, self.coords.expand(b, -1, -1, -1)], dim=1)))
        scale, shift = self.film(trunk_feat).chunk(2, dim=-1)
        h = h * (1 + scale[..., None, None]) + shift[..., None, None]
        for block in self.blocks:
            h = F.relu(h + block(h))
        return self.out(h).flatten(1)


class ActorCritic(nn.Module):
    def __init__(
        self,
        conv_channels: int = 32,
        trunk_dim: int = 256,
        param_entropy_norm: bool = False,
        tile_head: str = "small",
    ):
        super().__init__()
        # Divide each PARAMETER head's entropy (tile, player) by log(number of
        # legal choices) in evaluate_actions(), i.e. the entropy bonus sees it
        # in [0, 1]. Off by default: the raw sum is unchanged. See
        # evaluate_actions() for why. Not a weight, so not in the state_dict --
        # train.py's --param-entropy-norm sets it every run.
        self.param_entropy_norm = param_entropy_norm
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
        # tile_head="big" swaps in BigTileHead instead (train.py's --tile-head);
        # the default keeps this exact module, so existing checkpoints load.
        if tile_head not in ("small", "big"):
            raise ValueError(f"tile_head must be 'small' or 'big', got {tile_head!r}")
        self.tile_head = tile_head
        if tile_head == "small":
            self.tile_head_conv = nn.Conv2d(conv_channels, 1, kernel_size=1)
        else:
            self.tile_head_big = BigTileHead(conv_channels, trunk_dim)

        self.scalar_mlp = nn.Sequential(
            nn.Linear(NUM_SELF_FEATURES, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # DeepSets phi: applied identically to every opponent slot. nn.Linear
        # maps over the trailing dim of a (B, slots, F) input, so the same
        # weights see every slot with no reshape and no per-slot parameters --
        # that weight sharing is exactly what makes this permutation-invariant
        # and slot-count-agnostic.
        self.opponent_mlp = nn.Sequential(
            nn.Linear(NUM_OPPONENT_FEATURES, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        # Pointer head for attack_opponent's target: one logit per slot, from
        # that slot's own pre-pool embedding. Shared across slots for the same
        # reason phi is -- it scores "how attractive is this opponent", which
        # must not depend on which arbitrary index it landed in.
        self.player_head = nn.Linear(64, 1)

        self.trunk = nn.Sequential(
            nn.Linear(conv_out_dim + 64 + 64, trunk_dim),
            nn.ReLU(inplace=True),
        )
        self.type_head = nn.Linear(trunk_dim, NUM_ACTIONS)
        self.value_head = nn.Linear(trunk_dim, 1)

    def _features(
        self,
        tile_grid: torch.Tensor,
        self_features: torch.Tensor,
        opponent_features: torch.Tensor,
        opponent_mask: torch.Tensor,
    ):
        """tile_grid: (B, H, W) int8/long in {0,1,2,3}. self_features:
        (B, NUM_SELF_FEATURES) float. opponent_features: (B, slots,
        NUM_OPPONENT_FEATURES) float. opponent_mask: (B, slots) bool, True
        where the slot is a real opponent rather than padding.

        Returns (trunk_feat, feat_map, opponent_emb) -- feat_map is the
        pre-pool conv output, needed by the tile head; opponent_emb is the
        pre-pool per-slot embedding, needed by the pointer head."""
        one_hot = F.one_hot(tile_grid.long(), num_classes=NUM_TILE_CLASSES).permute(0, 3, 1, 2).float()
        feat_map = self.conv_body(one_hot)
        conv_feat = self.pool(feat_map).flatten(1)
        # log1p compresses the huge dynamic range of troops/gold (can run into
        # the tens/hundreds of thousands) into something a small MLP can use.
        # Applied uniformly, including to the already-bounded ratio/flag
        # features: it's monotonic, so it only rescales them and the following
        # Linear can undo it -- not worth per-feature branching.
        self_feat = self.scalar_mlp(torch.log1p(self_features.clamp(min=0)))

        opponent_emb = self.opponent_mlp(torch.log1p(opponent_features.clamp(min=0)))
        # Zero out padding slots BEFORE summing so they contribute nothing.
        # Sum (not mean) is the standard DeepSets pooling and keeps "how many
        # opponents there are" recoverable from the summary, which a mean
        # would average away.
        opponent_summary = (opponent_emb * opponent_mask.unsqueeze(-1).float()).sum(dim=1)

        trunk_feat = self.trunk(torch.cat([conv_feat, self_feat, opponent_summary], dim=1))
        return trunk_feat, feat_map, opponent_emb

    def forward(
        self,
        tile_grid: torch.Tensor,
        self_features: torch.Tensor,
        opponent_features: torch.Tensor,
        opponent_mask: torch.Tensor,
        tile_target_mask: torch.Tensor,
        attack_target_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        """Returns (type_logits, tile_logits, player_logits, value).
        action_mask: (B, NUM_ACTIONS) bool, True = legal action type.
        tile_target_mask: (B, MACRO_GRID, MACRO_GRID) bool, True = legal
        boat_attack destination. attack_target_mask: (B, slots) bool, True =
        legal attack_opponent target. Both parameter masks are only
        meaningful when their own action type is legal; otherwise they're
        ignored downstream by the masked-sum log_prob composition in
        act()/evaluate_actions(). An all-False parameter mask is fine: every
        logit becomes -1e9, so the distribution is uniform and finite rather
        than NaN, and its term is gated out anyway."""
        trunk_feat, feat_map, opponent_emb = self._features(
            tile_grid, self_features, opponent_features, opponent_mask
        )

        type_logits = self.type_head(trunk_feat)
        type_logits = type_logits.masked_fill(~action_mask, -1e9)

        if self.tile_head == "big":
            tile_logits = self.tile_head_big(feat_map, trunk_feat, tile_target_mask)
        else:
            tile_logits = self.tile_head_conv(feat_map)  # (B, 1, H', W')
            tile_logits = F.adaptive_avg_pool2d(tile_logits, MACRO_GRID).flatten(1)  # (B, MACRO_GRID^2)
        tile_mask_flat = tile_target_mask.reshape(tile_logits.shape[0], -1).bool()
        tile_logits = tile_logits.masked_fill(~tile_mask_flat, -1e9)

        player_logits = self.player_head(opponent_emb).squeeze(-1)  # (B, slots)
        player_logits = player_logits.masked_fill(~attack_target_mask, -1e9)

        value = self.value_head(trunk_feat).squeeze(-1)
        return type_logits, tile_logits, player_logits, value

    @torch.no_grad()
    def act(
        self,
        tile_grid: torch.Tensor,
        self_features: torch.Tensor,
        opponent_features: torch.Tensor,
        opponent_mask: torch.Tensor,
        tile_target_mask: torch.Tensor,
        attack_target_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        """Sample an action for rollout collection.
        Returns (action, log_prob, value) -- action is (B, 3) int:
        [action_type, macro_tile_idx, player_idx]. Each parameter is only
        meaningful (and only contributes to log_prob) for its own action
        type -- macro_tile_idx for boat_attack, player_idx for
        attack_opponent; otherwise it's still a validly-sampled index
        (harmless, ignored server-side and by the composition below)."""
        type_logits, tile_logits, player_logits, value = self.forward(
            tile_grid, self_features, opponent_features, opponent_mask,
            tile_target_mask, attack_target_mask, action_mask,
        )
        type_dist = torch.distributions.Categorical(logits=type_logits)
        tile_dist = torch.distributions.Categorical(logits=tile_logits)
        player_dist = torch.distributions.Categorical(logits=player_logits)
        action_type = type_dist.sample()
        tile_idx = tile_dist.sample()
        player_idx = player_dist.sample()

        log_prob = self._compose_log_prob(
            action_type, tile_idx, player_idx, type_dist, tile_dist, player_dist
        )

        action = torch.stack([action_type, tile_idx, player_idx], dim=-1)
        return action, log_prob, value

    @staticmethod
    def _compose_log_prob(action_type, tile_idx, player_idx, type_dist, tile_dist, player_dist):
        """Masked-sum composition: the type term always counts, each
        parameter term only when its own type was the one sampled. Keeps
        log_prob a single scalar per transition, so ppo.py's ratio/clip math
        is unchanged by having more parameter heads."""
        is_boat = (action_type == BOAT_ATTACK_IDX).float()
        is_attack = (action_type == ATTACK_OPPONENT_IDX).float()
        return (
            type_dist.log_prob(action_type)
            + is_boat * tile_dist.log_prob(tile_idx)
            + is_attack * player_dist.log_prob(player_idx)
        )

    def evaluate_actions(
        self,
        tile_grid: torch.Tensor,
        self_features: torch.Tensor,
        opponent_features: torch.Tensor,
        opponent_mask: torch.Tensor,
        tile_target_mask: torch.Tensor,
        attack_target_mask: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
    ):
        """Used during PPO updates: log_prob/entropy of given actions, plus
        value. actions: (B, 3) int, [action_type, macro_tile_idx,
        player_idx]."""
        action_type = actions[:, 0]
        tile_idx = actions[:, 1]
        player_idx = actions[:, 2]
        type_logits, tile_logits, player_logits, value = self.forward(
            tile_grid, self_features, opponent_features, opponent_mask,
            tile_target_mask, attack_target_mask, action_mask,
        )
        type_dist = torch.distributions.Categorical(logits=type_logits)
        tile_dist = torch.distributions.Categorical(logits=tile_logits)
        player_dist = torch.distributions.Categorical(logits=player_logits)

        log_prob = self._compose_log_prob(
            action_type, tile_idx, player_idx, type_dist, tile_dist, player_dist
        )
        is_boat = (action_type == BOAT_ATTACK_IDX).float()
        is_attack = (action_type == ATTACK_OPPONENT_IDX).float()
        tile_entropy = tile_dist.entropy()
        player_entropy = player_dist.entropy()
        if self.param_entropy_norm:
            # The raw tile entropy runs up to ln(1024) = 6.9 nats against the
            # type head's ln(4) = 1.4, so under one shared entropy coefficient
            # the bonus pulls hardest on keeping boat TARGETS uniform. Measured
            # on the connected-first run at update 305: the tile head sat at
            # 88-93% of its maximum entropy over 240-510 legal cells -- boats
            # to effectively random cells, 20% of the army each. Normalized,
            # each head's term is a fraction of ITS OWN maximum. n <= 1 legal
            # choice has zero entropy; the clamp only avoids 0/0.
            tile_n = tile_target_mask.reshape(tile_entropy.shape[0], -1).sum(-1).float()
            player_n = attack_target_mask.sum(-1).float()
            tile_entropy = tile_entropy / torch.log(tile_n.clamp_min(2.0))
            player_entropy = player_entropy / torch.log(player_n.clamp_min(2.0))
        entropy = type_dist.entropy() + is_boat * tile_entropy + is_attack * player_entropy
        return log_prob, entropy, value

    @torch.no_grad()
    def type_diagnostics(
        self,
        tile_grid: torch.Tensor,
        self_features: torch.Tensor,
        opponent_features: torch.Tensor,
        opponent_mask: torch.Tensor,
        tile_target_mask: torch.Tensor,
        attack_target_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ):
        """Type-head-only entropy/probabilities, decoupled from the combined
        type+tile entropy the PPO loss uses. The logged training `entropy`
        is dominated by the tile head's much larger range (up to ~ln(1024))
        whenever boat_attack is sampled, which can mask whether the type
        head's own preference among {noop, expand, attack_opponent,
        boat_attack} is actually sharpening -- e.g. a persistent eval-time
        bias toward noop could be invisible in the aggregate entropy number.
        Returns (type_probs_mean: (NUM_ACTIONS,), type_entropy_mean: scalar)."""
        type_logits, _, _, _ = self.forward(
            tile_grid, self_features, opponent_features, opponent_mask,
            tile_target_mask, attack_target_mask, action_mask,
        )
        probs = F.softmax(type_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        return probs.mean(dim=0), dist.entropy().mean()
