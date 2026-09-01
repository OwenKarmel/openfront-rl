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
  same `Intent` protocol as a human.
  - `src/GameSetup.ts` — shared episode setup (`Episode` class): builds AGENT
    (a controllable Human player) and OPPONENT (a real built-in Nation-AI
    player, `NationExecution`, at a selectable `Difficulty`, drawn from the
    map's real manifest) on a real production map. Construction mirrors
    OpenFrontIO's own `createGameRunner()` step-for-step — real `Config`
    (not a simplified test stand-in), real `GameType.Singleplayer`, real
    player/nation id derivation via `PseudoRandom.nextID()` — so an episode
    reconstructs identically wherever it's loaded (see "Why real
    construction matters" below).
  - `src/runEpisode.ts` — headless smoke test / determinism check, with a
    `--dump-game-record <dir>` flag to write a full `GameRecord`.
  - `src/EnvServer.ts` — long-lived process speaking a newline-JSON
    reset/step protocol over stdin/stdout (see file header for the wire
    format and action space).
  - `src/ReplayServer.ts` — tiny local HTTP server (defaults to port 8787,
    which a plain `npm run dev` client already checks by default) serving
    `GET /game/:id` from a directory of `Episode.toGameRecord()` dumps, so
    an episode can be watched in the real OpenFrontIO browser client — see
    "Watching a game" below.
- `training/` — Python.
  - `envs/openfront_env.py` — `gymnasium.Env` wrapping `EnvServer.ts` over a
    subprocess. Single-agent: AGENT is controlled by `step(action)`;
    OPPONENT (the Nation AI) needs no action, it plays itself.
  - `curriculum.py` — `CurriculumScheduler`: windowed win-rate promotion/
    demotion through Easy → Medium → Hard → Impossible, now driven by real
    training via `train.py`.
  - `smoke_test.py`, `curriculum_demo.py`, `dump_random_policy_log.py` —
    round-trip checks and mechanism demos using a random (legal-action-
    masked) policy; none of them are the actual training loop.
  - `models/network.py` — the actor-critic network: a small CNN (strided
    convs + adaptive pooling, so it works across map sizes unchanged) over
    the one-hot tile-ownership grid, concatenated with an MLP over the six
    scalar player-stat features, feeding a shared trunk into masked policy
    and value heads. Deliberately small (a few hundred thousand params) —
    the compute budget is one local GTX 1660 plus occasional Kaggle bursts,
    not a cluster.
  - `envs/vector_env.py` — a minimal synchronous multi-env wrapper (N
    `OpenFrontEnv` instances, auto-resetting on episode end) for more
    diverse rollout data; not yet OS-parallel (see file docstring).
  - `ppo.py` — the PPO algorithm itself: rollout buffer, GAE, clipped
    surrogate update. Plain/textbook, no distributed training.
  - `train.py` — **the actual training loop** (Phase 3): collects rollouts
    against the real Nation-AI opponent, runs PPO updates, feeds episode
    outcomes into `CurriculumScheduler`, checkpoints (resumable — safe to
    stop and restart across, e.g., Kaggle's 12h session cap), and
    periodically runs a greedy eval episode that dumps a real, verifiable
    `GameRecord` of how the current policy actually plays (same
    construction as training — see `GameSetup.ts` — so it's watchable in
    the real client and verifiable via `npm run replay:game`, not an
    approximation of what training saw).
- `kaggle/` — not yet built (Kaggle burst-training notebooks).

## Why real construction matters (train/deploy fidelity)

Earlier versions of this project used a `TestConfig`-derived config (fast,
deterministic attack attrition — a flat 1 troop lost per side per tick,
regardless of army size or terrain) and a hand-built synthetic opponent
that bypassed the map's real nation list. That was fast to iterate on, but
it meant training happened under game mechanics that barely resembled
production: real `Config.attackLogic()` scales troop loss and conquest
speed with troop density, terrain, relative army size, and outnumbered
ratio — nothing like a flat constant — and real spawn immunity is 50 ticks,
not 0. A policy trained under the fake mechanics would face serious
distributional shift the moment it played a real game.

`Episode` now uses real `Config` and mirrors `createGameRunner()` (the
literal function both live games and the client's own archived-game replay
path use) exactly: same player/nation construction order, same PRNG draw
sequence, same `GameType.Singleplayer` a real solo-vs-AI game uses. This
buys two things at once: training happens under real mechanics, and a
dumped episode is byte-for-byte reproducible by OpenFrontIO's own tooling
and the real client — there's exactly one construction path to keep
faithful, not a fast one and a separate "faithful" one to keep in sync.
Concretely this also means:

- `resolveMap()` now only supports real production maps (`resources/maps/`)
  for training/eval — `tests/testdata/` fixtures have no real `GameMapType`
  or manifest nations, both of which the real construction needs. (Fixture
  support still exists in the code but isn't used by default; it would only
  make sense for a throwaway wiring test that doesn't care about fidelity.)
- OPPONENT's identity is resolved dynamically per episode
  (`game.nations()[0].playerInfo.id`) rather than a fixed constant — which
  real nation you get (name, flag, spawn location) depends deterministically
  on the map and seed, exactly like a real game.
- The spawn phase isn't artificially shortened — real `Singleplayer` games
  end it the instant the human spawns anyway (confirmed empirically:
  episodes still start fast), so there was no actual fidelity/speed
  trade-off to make here.

## Action space

`env-bridge/src/EnvServer.ts`'s `ACTIONS`, mirrored in
`training/envs/openfront_env.py`'s `ACTIONS`:

- `noop` — do nothing this decision step
- `expand` — attack neutral (unowned) land bordering AGENT's territory
- `attack_opponent` — attack OPPONENT directly (only legal once they share a border)

Intentionally still small: richer intents (boat attacks, structure builds,
alliances) need a spatial/entity action head to *target* them, which is
Phase 3 network work, not env-bridge plumbing. `legal_actions`/`action_mask`
are provided every `reset()`/`step()` so a policy never needs to guess
legality.

## Watching a game

- **Quick debugging** (works today): any observation's `tile_grid` (Python)
  / `obs.tileGrid` (Node) is a flat ownership array (0=neutral, 1=agent,
  2=opponent) you can plot directly (e.g. `matplotlib.pyplot.imshow`).
- **Headless replay verification** (works today, via OpenFrontIO's own
  unmodified tooling): dump a `GameRecord` (`--dump-game-record <dir>` /
  `dump_game_record_dir=<dir>`), then from `OpenFrontIO/`:
  `npm run replay:game -- <path/to/dumped/record.json>`. Reports "Replay is
  IN SYNC with the recorded game" — confirmed working (0 mismatches across
  every hash checkpoint) once `Episode`'s construction was made to match
  `createGameRunner()` exactly; earlier versions of this project needed a
  bespoke verifier (`verifyRecord.ts`, now deleted) because their episodes
  couldn't be reconstructed by OpenFrontIO's own tooling.
- **Full visual playback in the actual browser client** (built and
  verified as far as this environment allows):
  ```bash
  # 1. Dump a full, strictly schema-valid GameRecord for an episode
  cd env-bridge
  npx tsx --tsconfig ../OpenFrontIO/tsconfig.json src/runEpisode.ts \
    --map onion --difficulty impossible --dump-game-record ../training/replays
  # -> prints the watch URL, e.g. .../game/c86a25e4

  # 2. Serve it (defaults to port 8787 — the address a plain `npm run dev`
  #    client already checks with zero configuration)
  npx tsx --tsconfig ../OpenFrontIO/tsconfig.json src/ReplayServer.ts

  # 3. In OpenFrontIO/: npm run dev, then open the printed URL, e.g.
  #    http://localhost:9000/game/c86a25e4
  ```
  `EnvServer.ts`'s `reset` also takes `dumpGameRecordDir` (Python:
  `OpenFrontEnv(dump_game_record_dir=...)`) to do the same from a training
  run. Built via `Episode.toGameRecord()`, which reuses
  `createPartialGameRecord` (the same helper the real server uses to
  archive games). Confirmed working end-to-end against a real running dev
  client: no desync popup (previously reproduced and fixed — see below),
  `checkArchivedGame()` fetches/accepts the record and proceeds into
  `handleJoinLobby`.
  - **The desync bug that motivated the real-construction rework**: an
    earlier version's episodes desynced the moment the real client tried to
    replay them (`GameRecord` reconstruction failed at turn 110 with a
    hash mismatch). Root cause was two-fold: (1) the live simulation was
    seeded from a different `gameID` than what got written into the dumped
    record, so the client's `PseudoRandom` reseeded from the wrong value
    immediately; (2) episodes bypassed real player/nation construction
    (`createNationsForGame`, `PseudoRandom.nextID()`-derived ids) with
    fixed constants and a hand-built opponent, which the client's
    reconstruction — the same `createGameRunner()` code path used for both
    live play and replay — could never regenerate to match. Both are fixed
    now: `Episode` uses one canonical gameID everywhere and mirrors real
    construction exactly (see "Why real construction matters" above).
  - `ReplayServer.ts`'s CORS headers must echo the request's `Origin` (not
    `Access-Control-Allow-Origin: *`) and set
    `Access-Control-Allow-Credentials: true`, since several of the client's
    other `apiBase`-directed calls (auth refresh, cosmetics, news) use
    `credentials: "include"`, which a wildcard origin can't satisfy.
  - **Not independently confirmed**: pixels actually on screen. This dev
    environment's headless Chromium hits OpenFrontIO's `WebGLGate`
    (`src/client/components/WebGLGate.ts`) — a deliberate hard block on
    software-rendered WebGL2 (SwiftShader/llvmpipe), added for real users
    hitting the same issue (see its `#4357` reference) — which also blocks
    the repo's own pre-existing `game.mjs` smoke test in this environment
    (confirmed by running it unmodified: it times out the same way,
    independent of anything built here). In a normal desktop browser with
    real GPU acceleration, the gate doesn't trigger and the confirmed
    server-side chain above (record generated → validated → fetched →
    accepted by the client, no desync) is exactly what feeds the renderer —
    and this has been spot-checked from a real browser (via port-forwarding
    into this dev environment), confirming rendering does work.

## Setup

```bash
git clone --recurse-submodules <this-repo-url>
cd openfront-rl/OpenFrontIO && npm ci && cd ..
cd env-bridge && npm install
cd ../training && pip install -r requirements.txt
```

Requires Node (any recent LTS — developed against v24) and network access
for the initial installs.

## Try it

```bash
# TypeScript: episode vs. a real Nation-AI opponent, determinism check
cd env-bridge
npm run smoke -- --map onion --seed smoke-1 --ticks 300 --difficulty hard

# Python: gymnasium round-trip against the same real opponent
cd ../training
python smoke_test.py

# Curriculum scheduler mechanism demo (random policy — expect it to stay
# at "easy"; only a real policy earns promotions)
python curriculum_demo.py

# Headless replay verification via OpenFrontIO's own stock tooling
cd ../env-bridge
npx tsx --tsconfig ../OpenFrontIO/tsconfig.json src/runEpisode.ts \
  --map onion --difficulty impossible --dump-game-record ../training/replays
cd ../OpenFrontIO
npm run replay:game -- ../training/replays/<gameID>.json
```

## Status

- [x] Phase 1: env-bridge scaffold, headless episode runner, determinism
      verified.
- [x] Phase 2: `EnvServer.ts` reset/step protocol + `openfront_env.py`
      gymnasium wrapper; a real built-in Nation-AI opponent (selectable
      Easy/Medium/Hard/Impossible, the same `NationExecution` code driving
      production games) wired in as OPPONENT; a 3-action space with legal-
      action masking; a windowed win-rate curriculum scheduler. All
      verified end-to-end, including through the full Python↔subprocess↔sim
      path.
- [x] Faithful train/eval construction: `Episode` now mirrors
      `createGameRunner()` exactly (real `Config`, real
      `GameType.Singleplayer`, real manifest-drawn opponent, one canonical
      gameID) instead of a simplified `TestConfig`-based stand-in — closing
      a real distributional-shift gap between training mechanics and
      production mechanics, and making dumped episodes verifiable by
      OpenFrontIO's own unmodified `npm run replay:game` and watchable in
      the real client with no desync.
- [x] Visual playback: `Episode.toGameRecord()` + `ReplayServer.ts` let the
      real OpenFrontIO browser client load and watch an episode
      (`GET /game/:id`, strictly schema-validated) with no desync — verified
      both headlessly (stock `replay:game`, IN SYNC) and live against a real
      running dev client (via a real desktop browser, port-forwarded in).
- [x] Phase 3 scaffold: `train.py` — a real PPO training loop (small
      CNN+MLP actor-critic, GAE, clipped surrogate update, multi-env
      rollouts, curriculum-driven difficulty, resumable checkpointing,
      periodic verified eval episodes). Verified end-to-end at both smoke
      scale and realistic default scale (4 envs × 64-step rollouts) on the
      local GTX 1660: runs, checkpoints, resumes correctly from a saved
      checkpoint, and produces eval `GameRecord`s that pass
      `npm run replay:game` IN SYNC even on a full ~3000-tick episode.
      **Not yet done**: actual training to convergence — this is the
      infrastructure, not a trained agent. Nothing here has run for more
      than a few minutes/updates.
- [ ] v1 milestone: train to consistently beat the Impossible-difficulty
      Nation bot 1v1 — the actual multi-hour-plus training run(s), likely
      spanning local + Kaggle sessions per the original compute plan.
- [ ] Phase 4 (stretch): scale-up, self-play league.

## Training

```bash
cd training
python train.py                          # real defaults: 4 envs, 64-step rollouts
# resumes automatically from checkpoints/latest.pt if present -- safe to
# Ctrl-C and rerun. Progress logs to checkpoints/train_log.csv, eval
# episodes (every --eval-every updates) land in replays/ as watchable/
# verifiable GameRecords -- see "Watching a game" above.
```

Key flags: `--num-envs`, `--rollout-length`, `--updates`, `--map`,
`--checkpoint-dir`, `--checkpoint-every`, `--eval-every` (see `train.py
--help` for the full list — learning rate, GAE/clip/entropy coefficients,
epochs, minibatch size are all exposed for tuning once real training
starts).
