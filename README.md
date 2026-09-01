# openfront-rl

Training a deep RL agent to beat OpenFrontIO's hardest built-in AI
("Impossible"-difficulty Nation bot) via self-play, under a tight compute
budget (one local GTX 1660 + free-tier Kaggle GPU/TPU hours).

See `.claude`-generated plan for full design rationale (PPO over
MuZero/EfficientZero, action/observation space, phased milestones).

## Repo layout

- `OpenFrontIO/` — git submodule, pinned to a specific commit of
  [openfrontio/OpenFrontIO](https://github.com/openfrontio/OpenFrontIO). This
  is the actual game simulation being trained against; never edited here.
- `env-bridge/` — TypeScript. Drives the real headless simulation
  (`GameRunner` + `Executor`, the same pipeline the browser client and
  server use) via Node, so an RL agent can control a player through the
  same `Intent` protocol as a human. `src/runEpisode.ts` is currently a
  Phase-1 smoke test / determinism check (no policy yet — see below).
- `training/`, `kaggle/` — not yet built (Phase 2+: PyTorch PPO, self-play
  league, Kaggle burst-training notebooks).

## Setup

```bash
git clone --recurse-submodules <this-repo-url>
cd openfront-rl/OpenFrontIO && npm ci && cd ..
cd env-bridge && npm install
```

Requires Node (any recent LTS — developed against v24) and network access
for the initial installs.

## Phase 1: env-bridge smoke test

```bash
cd env-bridge
npm run smoke -- --map plains --seed smoke-1 --ticks 200
```

Runs a full headless game with two Human players ("AGENT", "OPPONENT") on a
tiny test map from `OpenFrontIO/tests/testdata/maps/`, both spawning and then
periodically attacking neutral land, then re-runs it with the same seed and
asserts the final game-state hash is byte-identical. This is the foundation
the actual env (reset/step API, observation/action encoding) will be built
on in Phase 2 — it proves the headless pipeline is deterministic and crash-free
before any RL code is written.

Flags: `--map <name>` (any dir under `OpenFrontIO/tests/testdata/maps/`,
smallest is `ocean_and_land`), `--seed <string>`, `--ticks <n>`,
`--spawn-turns <n>`, `--act-every <n>`.

## Status

- [x] Phase 1: env-bridge scaffold, headless episode runner, determinism
      verified on `plains` and `ocean_and_land`.
- [ ] Phase 2: Gym-like reset/step API + Python wrapper, curriculum vs.
      Easy/Medium/Hard bots.
- [ ] Phase 3: beat the Impossible-difficulty Nation bot 1v1 (v1 milestone).
- [ ] Phase 4 (stretch): scale-up, self-play league.
