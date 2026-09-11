# Roadmap: building/structures, diplomacy

## Status

Two earlier phases are **implemented and shipped**, and their designs have
been pruned from this file — the README's Architecture / Action space /
Status sections are the live, accurate spec, not this document:

- **Naval/boat actions** — land/water tile classes, spatially-targeted
  `boat_attack` over a 32×32 macro-tile grid, the `tile_head`/masked-sum
  log_prob-entropy composition in `training/models/network.py`, and the real
  `targetTransportTile()`/`closestReachableShore()` legality check (deferred
  to fire only when `boat_attack` is actually the sampled type, not on every
  step, for performance).
- **Multi-opponent groundwork + variable-N architecture** (the former
  Phase 1) — the DeepSets `opponent_mlp` (φ) applied per padded opponent
  slot with masked sum-pooling, the `player_head` pointer read-out that
  makes `attack_opponent` target a `player_idx`, `attack_target_mask`, and
  the `self_troops_ratio` / per-opponent `troops_ratio` observations.

**`MAX_OPPONENTS` is still 1.** The shipped environment is the same 1v1
matchup as before; what changed is that neither the wire format nor any
weight shape depends on that number anymore, so raising it is a config
change rather than an architecture change. That distinction matters for
Phase 3 below, which needs the count actually raised, not merely raisable.

This file now covers only what is **not yet built**. Phase numbering is kept
as-is (2, 3) rather than renumbered, so it stays consistent with the
README's Status section.

## Phase 2 — Building/structure actions

Does not depend on multi-opponent support; it is ordered second by priority
call. The targeting conventions it reuses have all shipped, so this phase
can start whenever.

Add `build`/`upgrade`/`delete` action types reusing the existing macro-tile
head and masking machinery (`boat_attack`'s `tile_head`, generalized to
also serve as the build-location picker). New `unit_type_head` (small
categorical) over an initial subset — recommend `{Port, Warship, City,
DefensePost}` first (Port unlocks `warshipSpawn`'s water-component
requirement, directly synergizing with naval; City/DefensePost cover basic
economy/defense), deferring nukes/MIRV/Train/Factory/SAMLauncher.
Build-tile legality sourced from `PlayerImpl.buildableUnits(tile, types)`
(the same batch API the real game's build menu uses), computed lazily only
for currently-affordable unit types to bound per-step cost. Needs a second
observation channel (`structureGridB64`) so the agent can see existing
structures (own and opponent's) — necessary for `upgrade`/`delete` to be
learnable at all. Same "no new reward term" discipline as everywhere else
in this project.

The `unit_type_idx` slot `RolloutBuffer`'s action struct already reserved
is used here for real — the same story as the `player_idx` slot, which the
shipped pointer head now uses.

## Phase 3 — Diplomatic actions (gated on raising `MAX_OPPONENTS` above 1)

Gated on there actually being more than one opponent, not on the
architecture: the variable-N encoding has shipped, but `MAX_OPPONENTS` is
still 1, and diplomacy is close to meaningless against a single opponent.
Raising it is a config change (see Status).

Alliance request/accept/reject/extend, break-alliance, donate gold/troops,
embargo, target-hostile — all `player_idx`-parameterized, reusing the
shipped `player_head` pointer read-out rather than adding new architecture.
New per-opponent relation/alliance-state observation block (`relation()`,
`isAlliedWith()`, `allianceInfo()`, `betrayals()`) folds naturally into the
shipped `opponent_mlp` (φ) per-opponent feature vector as more input
features, rather than a separate encoder.
Explicitly **no reward term for diplomacy** — instrumental value already
flows through the existing territory-margin/terminal signal; this is the
single most important reward-discipline call in the roadmap given how
directly "diplomacy deserves its own reward" invites repeating this
project's past mistakes.

### Critical files (for reference across all phases)

- `env-bridge/src/EnvServer.ts`, `env-bridge/src/GameSetup.ts`
- `training/envs/openfront_env.py`, `training/models/network.py`, `training/ppo.py`, `training/train.py`
- `OpenFrontIO/src/core/configuration/Config.ts` (`maxTroops`, `troopIncreaseRate` — read-only; the basis for the shipped `troops_ratio` observations)
- `OpenFrontIO/src/core/game/PlayerImpl.ts` (`buildableUnits` — read-only reference for Phase 2)
- (read-only references throughout — no OpenFrontIO engine changes are needed anywhere in this roadmap, the same as for the naval and multi-opponent work that already shipped)
