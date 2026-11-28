# tests

Every test for this project's own code. OpenFrontIO's own test suite is not
here — it lives in `OpenFrontIO/` and is upstream's.

Run one directly, or all of them with `tests/run_all.sh`:

```bash
/home/developer/miniconda/envs/openfront-rl/bin/python tests/test_schedules.py
tests/run_all.sh          # all nineteen, in ascending cost order (~20min)
tests/run_all.sh -q       # same, only the per-file verdict
```

Each file is a plain script with its own `main()` and a non-zero exit on
failure — no pytest, no collection, no shared fixtures. They import from
`training/` via a two-line bootstrap at the top of each file:

```python
TRAINING = Path(__file__).resolve().parent.parent / "training"
sys.path.insert(0, str(TRAINING))
```

## Run them one at a time

`run_all.sh` is sequential deliberately. The env-backed tests each stand up
several Node env-bridge workers plus a torch process; run in parallel, or run
alongside a training run that is late in its gamma anneal, they compete for
the RAM that gets training OOM-killed. Check `free -m` first — see the
"Before starting a long run" section of `CLAUDE.md`, and note that `/tmp` is
tmpfs on this box.

## What each one covers

| test | needs | runtime | covers |
|---|---|---|---|
| `test_schedules.py` | — | 2s | `gamma_at`/`rollout_at`/`steps_before`/`horizon_steps`: the gamma anneal is geometric in (1−γ), the rollout equals exactly one discount horizon, and every schedule is a pure function of the update counter so a resume lands identically |
| `test_eval_cadence.py` | — | 1s | `eval_due`: step-keyed eval cadence fires on the same eval indices as the old update-keyed one when the rollout is constant, so eval index stays comparable across runs |
| `test_curriculum.py` | — | 2s | `CurriculumScheduler`/`LADDERS`: each promotion changes exactly one of difficulty and map pool, promotion at the suite-win-rate threshold and never past the last stage; `wilson_interval` at 0/n, mid and n/n; `map_pool_for` |
| `test_param_entropy_norm.py` | — | 2s | `--param-entropy-norm`: same weights, flag off vs on — log_prob and value identical, a boat row's entropy is type + tile / log(#legal cells), one legal cell gives 0 not NaN, non-parameter rows unchanged |
| `test_big_tile_head.py` | — | 16s (GPU) | `--tile-head big`: default head's module (and checkpoints) unchanged, small/big state_dicts refuse to cross-load, uniform start over legal cells, size-invariant logits, gradient only from boat rows, and ~98% supervised accuracy at aiming for the legal cell nearest AGENT's territory |
| `test_ppo_tier0.py` | — | 5s | `--kl-stop`/`--adv-norm`/`--value-norm`: flag-off bit-identity of the network (head off) and `ppo_update()` against verbatim copies of the pre-change code in `reference/`, with and without the KL stop firing; `epoch` mode runs every epoch on small KL and stops between epochs on large KL (and the old k1 check misses a KL k3 sees); `minibatch_bounds`; PopArt output preservation, identity start, std floor, and an lr=0 `ppo_update()` that leaves every denormalized prediction unchanged |
| `test_attack_size_head.py` | — | 3s | `--attack-size-head`, network side: no new parameters when off, P(20%)=0.9 at start in every state, the attack-size term in log_prob/entropy for attack types only, gradient only from attack rows, `--param-entropy-norm` divides it by log 5 |
| `test_init_from.py` | — | 3s | `--init-from`: shared weights, update count and stage carried, optimizer state moved by name (none for the new head), and the refusals (other missing keys, a different tile head, a different ladder); value-norm statistics and wall time in checkpoints, the resume refusals; `eval_seed_maps()` matches the env's own draw, map-stratified `canary_seeds()`, `wins_excl`, `per_map_line`, `open_csv` with an older header |
| `smoke_test.py` | env-bridge | 1s | one real episode end to end: obs shapes, action round trip, reward, termination |
| `smoke_test_multiopp.py` | env-bridge | 2s | variable-N opponent encoding: padding contributes nothing, permutation invariance, `attack_target_mask` respected, pointer-head log_prob composition, env round trip |
| `test_neutral_boats.py` | env-bridge | 6s | `OPENFRONT_NEUTRAL_BOATS=1` on fourislands: the boat mask grows from opponent-only to opponent-or-unowned land (same seed, flag off vs on), and a boat to another island's unowned land actually lands. Plus the `envFlags` echo and the stale-worker refusal |
| `test_eval_suite.py` | env-bridge | 20s | `run_eval_suite`: fixed seeds give the same maps every suite, workers are reused and every step is one batched `net.act()` (sampled policy), capped episodes are truncated not won, replays dumped only for seeds < `--eval-replays`, one gameID per (stage, seed) |
| `test_worker_memory_flags.py` | env-bridge | ~1min | `--recycle-every` reaches `OpenFrontEnv` and really respawns the worker; `run_eval_suite()` with 1 vs 3 workers and recycling every episode vs every 25 plays identical games |
| `test_expand_mask.py` | env-bridge, snapshot | 71s | `OPENFRONT_EXPAND_MASK=1`: reproduces eval `0ab47d4b` (the episode that conquered its island then pulsed forever) and measures where the mass goes once `expand` is masked off |
| `test_boat_mask.py` | env-bridge, snapshot | 32s | `OPENFRONT_BOAT_MASK=1`: reproduces eval `cc83d2cc` (landlocked all game) and measures the same redistribution for `boat_attack` |
| `test_selfplay.py` | env-bridge, OpenFrontIO npm | ~1min | self-play: `--expert-frac` parsing, `OpponentPool` FIFO/sampling/eviction fallback/save-load, `OpponentChooser` (expert_frac 1 is all-Nation and draws no randomness, the mix follows the stage); through a real env-bridge, the two seats' observations mirror exactly, random play by both seats ends with the right `winner`, the step guards refuse a missing/extra `opponent_action`, and the self-play GameRecord lists both humans and replays IN SYNC under `npm run replay:game` |
| `test_attack_size.py` | env-bridge | ~3min | `--attack-size-head`, wire side: the `attackSizeHead` reset echo (a stale bridge fails at reset), every `troopFraction` rejection path, 75% vs 10% actually differ; fraction 0.20 vs the head off in lockstep — boats identical to the engine's end, land attacks identical up to the first one and then off by the engine's one-tick sampling difference (see the file's docstring) |
| `test_reward_modes.py` | env-bridge | 3min | `--reward-mode`: the three modes in lockstep on one seed — dynamics identical, `shaping_and_terminal == shaping_only + terminal_only` exactly, and the shaping sum really is the tile-share delta at every prefix. Plus the four rejection paths |
| `test_potential_beta.py` | env-bridge, archive | 8min | `--potential-beta`: β 0/2/5 in lockstep, reward equals `Φ(p′) − Φ(p)` exactly against the engine's own tile counts, gains and losses weighted alike by `e^{βp}`, `Φ′(0) = 1`. Plus the rejection paths |

The two mask tests and `test_neutral_boats.py` are for behaviour that is
**env-var gated and off by default** (`OPENFRONT_EXPAND_MASK`,
`OPENFRONT_BOAT_MASK`, `OPENFRONT_NEUTRAL_BOATS`); they set the variables
themselves. No archived training run has used any of them;
`training/run_connected_first.sh` is the first launcher that turns all three on.

## Fixed checkpoints, not `latest.pt`

Three tests drive a real policy. All three load a **fixed archived
checkpoint**, never `training/checkpoints/latest.pt`:

- the mask tests load
  `training/checkpoints/archive/run_2026-09-11_bbf-adamw_gamma098_rollout50/policy_snapshot_update3000.pt`
- `test_potential_beta.py` loads
  `training/checkpoints/archive/run_2026-09-17_gamma-anneal098-0999_shaping-only/latest.pt`

`latest.pt` moves with whatever run is live, so pointing at it would silently
change what these tests cover — a freshly cold-started policy reaches neither
high territory nor sustained tile losses, and `test_potential_beta.py`'s
high-`p` and loss-weighting checks would keep passing while covering nothing.
Both paths are under `training/checkpoints/`, which is gitignored, so a fresh
clone has neither: `test_potential_beta.py` skips that arm with a note and
still runs its closed-form checks, and the mask tests exit with a message
naming the missing file.
