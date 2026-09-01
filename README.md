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
  - `src/GameSetup.ts` — shared episode setup/teardown (`Episode` class):
    builds AGENT (a controllable Human player) and OPPONENT (a real
    built-in Nation-AI player, `NationExecution`, at a selectable
    `Difficulty`) on either a tiny test map or a real production map.
  - `src/EnvConfig.ts` — deterministic combat config (fixed attack
    attrition, zero spawn immunity) with a settable short spawn phase, so
    RL episodes are reproducible and don't burn ticks on setup.
  - `src/runEpisode.ts` — headless smoke test / determinism check, with a
    `--dump-record <path>` flag to write a replayable turn log.
  - `src/EnvServer.ts` — long-lived process speaking a newline-JSON
    reset/step protocol over stdin/stdout (see file header for the wire
    format and action space).
  - `src/verifyRecord.ts` — headless replay verification for a dumped turn
    log (re-simulates it and diffs hash checkpoints).
- `training/` — Python.
  - `envs/openfront_env.py` — `gymnasium.Env` wrapping `EnvServer.ts` over a
    subprocess. Single-agent: AGENT is controlled by `step(action)`;
    OPPONENT (the Nation AI) needs no action, it plays itself.
  - `curriculum.py` — `CurriculumScheduler`: windowed win-rate promotion/
    demotion through Easy → Medium → Hard → Impossible. Scheduling policy
    only — plug a real training loop's per-episode win/loss into it.
  - `smoke_test.py`, `curriculum_demo.py`, `dump_random_policy_log.py` —
    round-trip checks and mechanism demos using a random (legal-action-
    masked) policy; none of them are the actual training loop.
  - PPO/network code: not yet built (Phase 3).
- `kaggle/` — not yet built (Kaggle burst-training notebooks, Phase 3+).

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
- **Headless replay verification** (works today): pass a real map —
  `--map onion` (smallest production map) instead of a `tests/testdata/`
  fixture — to `runEpisode.ts`/`OpenFrontEnv`, plus `--dump-record <path>` /
  `dump_record=<path>`, to write a turn log with real hash checkpoints.
  Verify it with `env-bridge`'s own `npx tsx src/verifyRecord.ts <path>`
  (re-simulates the turns through a fresh episode and diffs hashes).
  - **Not** OpenFrontIO's own `npm run replay:game`: that script
    reconstructs players via `random.nextID()` and a production `Config`,
    neither of which matches how env-bridge builds episodes (fixed
    `AGENT`/`OPPONENT` ids, `EnvConfig`'s deterministic combat) — the two
    diverge from tick 0 even though both are internally deterministic. Our
    own turn logs verify correctly with `verifyRecord.ts` instead.
- **Full visual playback in the actual browser client** (investigated, not
  built): the client only ever loads a `GameRecord` one way —
  `JoinLobbyModal.checkArchivedGame()` does `GET {apiBase}/game/{gameID}`
  and requires the response to pass `GameRecordSchema.safeParse` *strictly*
  (unlike the headless replay tool, there's no lenient fallback), plus a
  `gitCommit` match (or a DEV-build client, which skips that check). Getting
  a training episode on screen in the real client would need: (1) a tiny
  local HTTP server serving our dumped record at that route, with
  `getApiBase()` pointed at it, and (2) the record actually filled out to
  the full schema — real `GameEndInfo`/`PlayerRecord`/stats fields, not the
  loose shape `writeReplayRecord()` produces today. Not started; a bigger
  lift than headless verification was, and orthogonal to the training
  milestone, so scoped separately.

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
npm run smoke -- --map plains --seed smoke-1 --ticks 300 --difficulty hard

# Python: gymnasium round-trip against the same real opponent
cd ../training
python smoke_test.py

# Curriculum scheduler mechanism demo (random policy — expect it to stay
# at "easy"; only a real policy earns promotions)
python curriculum_demo.py

# Headless replay verification on a real map
cd ../env-bridge
npx tsx --tsconfig ../OpenFrontIO/tsconfig.json src/runEpisode.ts \
  --map onion --difficulty impossible --dump-record /tmp/record.json
npx tsx --tsconfig ../OpenFrontIO/tsconfig.json src/verifyRecord.ts /tmp/record.json
```

## Status

- [x] Phase 1: env-bridge scaffold, headless episode runner, determinism
      verified.
- [x] Phase 2: `EnvServer.ts` reset/step protocol + `openfront_env.py`
      gymnasium wrapper; a real built-in Nation-AI opponent (selectable
      Easy/Medium/Hard/Impossible, the same `NationExecution` code driving
      production games) wired in as OPPONENT; a 3-action space with legal-
      action masking; headless replay verification on real production maps
      (with a real terrain-caching bug found and fixed along the way); a
      windowed win-rate curriculum scheduler. All verified end-to-end,
      including through the full Python↔subprocess↔sim path.
- [ ] Phase 3: PPO network + training loop; beat the Impossible-difficulty
      Nation bot 1v1 (v1 milestone).
- [ ] Phase 4 (stretch): scale-up, self-play league, full visual playback
      in the browser client.
