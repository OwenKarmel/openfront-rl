"""Smoke test for the variable-opponent-count (DeepSets) architecture.

The environment currently runs exactly one opponent (MAX_OPPONENTS = 1), so
training alone never exercises the padding/masking path -- opponent_mask is
all-True and there is only ever one slot to point at. This test covers what
training cannot: that ONE network instance handles 1, 5, and 10 opponent
slots without any weight reshaping, that padded slots contribute nothing,
that the set encoder is permutation-invariant, and that the pointer head
never targets a slot the attack mask forbids.

Those are the properties that make raising MAX_OPPONENTS a config change
rather than an architecture change, so they're worth checking before the
count is ever raised -- a bug here would otherwise stay invisible until the
first multi-opponent run and then be indistinguishable from "multi-opponent
learning is just hard".

Run: python smoke_test_multiopp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from envs.openfront_env import ACTIONS, OpenFrontEnv
from models.network import (
    ATTACK_OPPONENT_IDX,
    MACRO_GRID,
    NUM_ACTIONS,
    NUM_OPPONENT_FEATURES,
    NUM_SELF_FEATURES,
    ActorCritic,
)

GRID = 64  # smaller than a real map; the conv path is size-invariant


def make_inputs(
    slots: int,
    real: int,
    batch: int = 2,
    attack_legal: list[int] | None = None,
    seed: int = 0,
) -> tuple:
    """One batch of synthetic observations with `slots` opponent slots, of
    which the first `real` are genuine opponents and the rest padding."""
    g = torch.Generator().manual_seed(seed)
    opponent_features = torch.rand(batch, slots, NUM_OPPONENT_FEATURES, generator=g) * 1000
    opponent_mask = torch.zeros(batch, slots, dtype=torch.bool)
    opponent_mask[:, :real] = True
    # Padding rows are zeroed by the env-bridge (see PADDING_OPPONENT in
    # EnvServer.ts); mirror that here rather than leaving them random, so
    # this test checks the masking rather than accidentally depending on it.
    opponent_features[~opponent_mask] = 0.0

    attack_target_mask = torch.zeros(batch, slots, dtype=torch.bool)
    for i in attack_legal if attack_legal is not None else range(real):
        attack_target_mask[:, i] = True

    return (
        torch.randint(0, 4, (batch, GRID, GRID), generator=g),
        torch.rand(batch, NUM_SELF_FEATURES, generator=g) * 1000,
        opponent_features,
        opponent_mask,
        torch.ones(batch, MACRO_GRID, MACRO_GRID, dtype=torch.bool),
        attack_target_mask,
        torch.ones(batch, NUM_ACTIONS, dtype=torch.bool),
    )


def check_variable_slot_counts(net: ActorCritic) -> None:
    """The same net, unmodified, across three different opponent counts."""
    for slots in (1, 5, 10):
        inputs = make_inputs(slots, real=slots)
        type_logits, tile_logits, player_logits, value = net(*inputs)
        assert type_logits.shape == (2, NUM_ACTIONS), type_logits.shape
        assert tile_logits.shape == (2, MACRO_GRID * MACRO_GRID), tile_logits.shape
        assert player_logits.shape == (2, slots), player_logits.shape
        assert value.shape == (2,), value.shape
        assert torch.isfinite(value).all()
        print(f"  slots={slots:2d}: player_logits={tuple(player_logits.shape)} value={tuple(value.shape)}")


def check_padding_ignored(net: ActorCritic) -> None:
    """One real opponent must look identical whether it sits alone or in a
    10-wide array with 9 padding slots -- i.e. the sum-pool really is masked.

    The wide case is derived from the narrow one by appending slots (rather
    than generated independently) so every other input is identical by
    construction. The padding is deliberately filled with large garbage
    values, not the zeros the env-bridge really sends: zeros would pass this
    test even with the mask removed entirely, which would defeat the point."""
    narrow = list(make_inputs(slots=1, real=1, seed=7))
    batch, pad = narrow[2].shape[0], 9
    garbage = torch.full((batch, pad, NUM_OPPONENT_FEATURES), 1e6)

    wide = list(narrow)
    wide[2] = torch.cat([narrow[2], garbage], dim=1)
    wide[3] = torch.cat([narrow[3], torch.zeros(batch, pad, dtype=torch.bool)], dim=1)
    wide[5] = torch.cat([narrow[5], torch.zeros(batch, pad, dtype=torch.bool)], dim=1)

    narrow_type, _, _, narrow_value = net(*narrow)
    wide_type, _, _, wide_value = net(*wide)
    assert torch.allclose(narrow_type, wide_type, atol=1e-5), (narrow_type, wide_type)
    assert torch.allclose(narrow_value, wide_value, atol=1e-5), (narrow_value, wide_value)
    print(f"  padded-to-10 matches unpadded: max value delta="
          f"{(narrow_value - wide_value).abs().max().item():.2e}")


def check_permutation_invariance(net: ActorCritic) -> None:
    """Reordering the real opponents must not change what the network thinks
    of the situation -- slot assignment is arbitrary. The pointer head is
    equivariant rather than invariant: its logits follow the permutation."""
    inputs = list(make_inputs(slots=5, real=3, seed=11))
    perm = [2, 0, 1, 3, 4]  # permutes the 3 real slots, leaves padding last
    permuted = list(inputs)
    permuted[2] = inputs[2][:, perm]
    permuted[3] = inputs[3][:, perm]
    permuted[5] = inputs[5][:, perm]

    type_a, _, player_a, value_a = net(*inputs)
    type_b, _, player_b, value_b = net(*permuted)
    assert torch.allclose(type_a, type_b, atol=1e-5)
    assert torch.allclose(value_a, value_b, atol=1e-5)
    assert torch.allclose(player_a[:, perm], player_b, atol=1e-5)
    print(f"  order-independent: max value delta={(value_a - value_b).abs().max().item():.2e}; "
          f"pointer logits follow the permutation")


def check_attack_mask_respected(net: ActorCritic) -> None:
    """act() must never point at a padded or mask-forbidden slot. Forced to
    attack_opponent every step (only that type is legal) so the player_idx
    branch is actually taken."""
    legal_slots = [1, 4]
    inputs = list(make_inputs(slots=8, real=5, attack_legal=legal_slots, seed=3))
    action_mask = torch.zeros(2, NUM_ACTIONS, dtype=torch.bool)
    action_mask[:, ATTACK_OPPONENT_IDX] = True
    inputs[6] = action_mask

    sampled = set()
    for _ in range(200):
        action, log_prob, _ = net.act(*inputs)
        assert (action[:, 0] == ATTACK_OPPONENT_IDX).all()
        sampled.update(action[:, 2].tolist())
        assert torch.isfinite(log_prob).all()
    assert sampled <= set(legal_slots), f"sampled illegal slots: {sampled - set(legal_slots)}"
    print(f"  200 forced attack_opponent samples stayed inside legal slots {legal_slots}: {sorted(sampled)}")


def check_log_prob_composition(net: ActorCritic) -> None:
    """player_idx must affect log_prob for attack_opponent and ONLY for it --
    the masked-sum composition's whole point."""
    inputs = make_inputs(slots=6, real=6, attack_legal=[0, 1, 2, 3, 4, 5], seed=5)
    expand_idx = ACTIONS.index("expand")

    def log_prob_for(action_type: int, player_idx: int) -> torch.Tensor:
        actions = torch.tensor([[action_type, 0, player_idx]] * 2)
        log_prob, _, _ = net.evaluate_actions(*inputs, actions)
        return log_prob

    assert torch.allclose(log_prob_for(expand_idx, 0), log_prob_for(expand_idx, 3), atol=1e-6), \
        "player_idx changed log_prob for a non-attack action"
    attack_a = log_prob_for(ATTACK_OPPONENT_IDX, 0)
    attack_b = log_prob_for(ATTACK_OPPONENT_IDX, 3)
    assert not torch.allclose(attack_a, attack_b, atol=1e-6), \
        "player_idx did not affect log_prob for attack_opponent"
    print(f"  expand: player_idx ignored; attack_opponent: player_idx counts "
          f"(delta={(attack_a - attack_b).abs().max().item():.4f})")


def check_env_player_idx_round_trip() -> None:
    """The playerIdx wire field must reach EnvServer.ts and resolve to a real
    opponent, and an out-of-range slot must fail loudly rather than silently
    attacking the wrong player."""
    env = OpenFrontEnv(
        map_name="onion", seed="smoke-multiopp", difficulty="hard",
        max_steps=40, ticks_per_step=10,
    )
    try:
        obs, info = env.reset(seed=1)
        assert obs["opponent_features"].shape[0] == obs["opponent_mask"].shape[0]
        assert obs["opponent_mask"].all(), "every slot should be a real opponent at MAX_OPPONENTS=1"
        for _ in range(12):
            obs, _, _, _, info = env.step([ACTIONS.index("expand"), 0, 0])

        # Sent regardless of attack_target_mask: this checks the wire path,
        # not the legality gate (the engine no-ops an attack with no shared
        # border, which is a valid outcome, not an error).
        _, reward, _, _, info = env.step([ACTIONS.index("attack_opponent"), 0, 0])
        assert np.isfinite(reward)
        print(f"  attack_opponent(playerIdx=0) round-tripped, ticks={info['ticks']}")

        try:
            env.step([ACTIONS.index("attack_opponent"), 0, 99])
            raise AssertionError("out-of-range playerIdx was accepted")
        except RuntimeError as err:
            assert "not a real opponent" in str(err), err
            print("  out-of-range playerIdx rejected by the env-bridge")
    finally:
        env.close()


def main() -> None:
    torch.manual_seed(0)
    net = ActorCritic()
    net.eval()
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"ActorCritic: {params:,} trainable parameters\n")

    print("variable slot counts (one net, no reshaping):")
    check_variable_slot_counts(net)
    print("\npadding contributes nothing:")
    check_padding_ignored(net)
    print("\npermutation invariance:")
    check_permutation_invariance(net)
    print("\nattack target mask respected:")
    check_attack_mask_respected(net)
    print("\nlog_prob composition:")
    check_log_prob_composition(net)
    print("\nenv round trip:")
    check_env_player_idx_round_trip()

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
