/**
 * Phase-1 smoke test / determinism check for the env-bridge.
 *
 * Drives a full headless OpenFrontIO game through the real intent pipeline
 * (GameRunner + Executor, exactly like the browser client and the perf
 * harness do), with two Human players ("AGENT" and "OPPONENT") on a tiny
 * test map. No policy yet: both players just spawn, then periodically issue
 * an "attack neutral land" intent so territory actually changes over time.
 *
 * This exists to prove the plumbing works before any RL code is written:
 *   - a full episode runs to completion without desync/crash
 *   - the same seed produces byte-identical game-state hashes every time
 *
 * Usage:
 *   npx tsx src/runEpisode.ts [--map plains] [--seed smoke-1] [--ticks 300]
 *                              [--spawn-turns 3] [--act-every 10]
 */
import { AGENT_CLIENT_ID, Episode, OPPONENT_CLIENT_ID } from "./GameSetup";
import { StampedIntent } from "../../OpenFrontIO/src/core/Schemas";

interface Options {
  map: string;
  seed: string;
  ticks: number;
  spawnTurns: number;
  actEvery: number;
  dumpRecord: string | undefined;
}

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    map: "plains",
    seed: "smoke-1",
    ticks: 300,
    spawnTurns: 3,
    actEvery: 10,
    dumpRecord: undefined,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`missing value for ${arg}`);
      return v;
    };
    switch (arg) {
      case "--map":
        opts.map = next();
        break;
      case "--seed":
        opts.seed = next();
        break;
      case "--ticks":
        opts.ticks = parseInt(next(), 10);
        break;
      case "--spawn-turns":
        opts.spawnTurns = parseInt(next(), 10);
        break;
      case "--act-every":
        opts.actEvery = parseInt(next(), 10);
        break;
      case "--dump-record":
        opts.dumpRecord = next();
        break;
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }
  return opts;
}

interface EpisodeResult {
  ticks: number;
  finalHash: string | undefined;
  finalHashTick: number | undefined;
  players: { name: string; alive: boolean; tiles: number; troops: number; gold: number }[];
}

async function runEpisode(opts: Options): Promise<EpisodeResult> {
  console.debug = () => {}; // silence per-tick debug logging

  const episode = await Episode.create(opts.map, opts.seed, opts.spawnTurns);

  const expandIntent = (clientID: string): StampedIntent => ({
    type: "attack",
    targetID: null,
    troops: null,
    clientID,
  });
  for (let i = 0; i < opts.ticks; i++) {
    const intents: StampedIntent[] =
      i % opts.actEvery === 0
        ? [expandIntent(AGENT_CLIENT_ID), expandIntent(OPPONENT_CLIENT_ID)]
        : [];
    if (!episode.runTick(intents) || episode.isDone()) break;
  }

  if (opts.dumpRecord) {
    episode.writeReplayRecord(opts.dumpRecord);
    console.log(`Wrote replay record to ${opts.dumpRecord}`);
  }

  return {
    ticks: episode.game.ticks(),
    finalHash: episode.lastHash?.hash?.toString(),
    finalHashTick: episode.lastHash?.tick,
    players: episode.game.players().map((p) => ({
      name: p.name(),
      alive: p.isAlive(),
      tiles: p.numTilesOwned(),
      troops: p.troops(),
      gold: Number(p.gold()),
    })),
  };
}

function printResult(label: string, r: EpisodeResult): void {
  console.log(`\n--- ${label} ---`);
  console.log(
    `ticks=${r.ticks} finalHash=${r.finalHash ?? "n/a"} (tick ${r.finalHashTick ?? "n/a"})`,
  );
  for (const p of r.players) {
    console.log(
      `  ${p.name}: alive=${p.alive} tiles=${p.tiles} troops=${p.troops.toFixed(0)} gold=${p.gold.toFixed(0)}`,
    );
  }
}

async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));
  console.log(
    `Running episode: map=${opts.map} seed=${opts.seed} ticks=${opts.ticks} ` +
      `spawnTurns=${opts.spawnTurns} actEvery=${opts.actEvery}`,
  );

  const run1 = await runEpisode(opts);
  printResult("run 1", run1);

  const run2 = await runEpisode(opts);
  printResult("run 2 (same seed)", run2);

  if (run1.finalHash === undefined || run2.finalHash === undefined) {
    throw new Error("no hash was ever recorded — determinism unverified");
  }
  if (run1.finalHash !== run2.finalHash || run1.ticks !== run2.ticks) {
    throw new Error(
      "DETERMINISM CHECK FAILED: same seed produced different results",
    );
  }
  console.log(
    `\nDeterminism check passed: identical hash (${run1.finalHash}) across two runs of seed "${opts.seed}".`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
