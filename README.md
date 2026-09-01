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
  - `src/GameSetup.ts` — shared episode setup/teardown (`Episode` class).
  - `src/runEpisode.ts` — Phase-1 smoke test / determinism check, and a
    `--dump-record <path>` flag to write a replayable turn log.
  - `src/EnvServer.ts` — Phase-2 long-lived process speaking a newline-JSON
    reset/step protocol over stdin/stdout (see file header for the wire
    format). Minimal action space so far: `noop` / `expand`.
- `training/` — Python.
  - `envs/openfront_env.py` — `gymnasium.Env` wrapping `EnvServer.ts` over a
    subprocess.
  - `smoke_test.py` — random-policy round-trip check of the whole stack.
  - PPO/network/curriculum code: not yet built (Phase 3).
- `kaggle/` — not yet built (Kaggle burst-training notebooks, Phase 3+).

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
- [x] Phase 2 (core plumbing): `EnvServer.ts` reset/step protocol +
      `openfront_env.py` gymnasium wrapper, verified end-to-end with a
      random policy (`training/smoke_test.py`). Action space is still just
      `noop`/`expand` against a scripted "always expand" opponent — no real
      Nation/bot difficulty wired in yet, no PPO network yet.
- [ ] Phase 2 remainder: wire a real Nation-AI opponent (Easy → Impossible)
      into `EnvServer.ts`/`Episode`, richer action space, curriculum.
- [ ] Phase 3: PPO network + training loop; beat the Impossible-difficulty
      Nation bot 1v1 (v1 milestone).
- [ ] Phase 4 (stretch): scale-up, self-play league.
