/**
 * Smoke test / determinism check for the env-bridge, against a real
 * built-in Nation-AI opponent (not a scripted stand-in).
 *
 * Drives a full headless OpenFrontIO game through the real intent pipeline
 * (GameRunner + Executor, exactly like the browser client and the perf
 * harness do): AGENT is a Human player that spawns and then periodically
 * expands into neutral land; OPPONENT is a real Nation-AI player
 * (NationExecution) at the given --difficulty, making its own decisions
 * every tick exactly as it would in a production game.
 *
 * This exists to prove the plumbing works before any RL code is written:
 *   - a full episode runs to completion without desync/crash
 *   - the same seed produces byte-identical game-state hashes every time
 *
 * Usage:
 *   npx tsx src/runEpisode.ts [--map plains] [--seed smoke-1] [--ticks 300]
 *                              [--spawn-turns 3] [--act-every 10]
 *                              [--difficulty medium]
 *                              [--dump-game-record ../training/replays]
 *
 * --dump-game-record <dir> writes a full GameRecord (Episode.toGameRecord())
 * to <dir>/<wireGameID>.json, servable by ReplayServer.ts for watching in
 * the actual OpenFrontIO client — see README "Watching a game".
 */
import fs from "fs";
import path from "path";
import { AGENT_CLIENT_ID, Episode } from "./GameSetup";
import { Difficulty } from "../../OpenFrontIO/src/core/game/Game";
import { StampedIntent } from "../../OpenFrontIO/src/core/Schemas";

interface Options {
  map: string;
  seed: string;
  ticks: number;
  spawnTurns: number;
  actEvery: number;
  difficulty: Difficulty;
  dumpRecord: string | undefined;
  dumpGameRecord: string | undefined;
}

function parseDifficulty(name: string): Difficulty {
  const key = Object.keys(Difficulty).find(
    (k) => k.toLowerCase() === name.toLowerCase(),
  );
  if (key === undefined) {
    throw new Error(
      `unknown difficulty "${name}": expected one of ${Object.keys(Difficulty).join(", ")}`,
    );
  }
  return Difficulty[key as keyof typeof Difficulty];
}

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    map: "plains",
    seed: "smoke-1",
    ticks: 300,
    spawnTurns: 3,
    actEvery: 10,
    difficulty: Difficulty.Medium,
    dumpRecord: undefined,
    dumpGameRecord: undefined,
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
      case "--difficulty":
        opts.difficulty = parseDifficulty(next());
        break;
      case "--dump-record":
        opts.dumpRecord = next();
        break;
      case "--dump-game-record":
        opts.dumpGameRecord = next();
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
  winner: string;
  players: { name: string; alive: boolean; tiles: number; troops: number; gold: number }[];
}

async function runEpisode(opts: Options): Promise<EpisodeResult> {
  console.debug = () => {}; // silence per-tick debug logging

  const episode = await Episode.create(
    opts.map,
    opts.seed,
    opts.spawnTurns,
    opts.difficulty,
  );

  const expandIntent: StampedIntent = {
    type: "attack",
    targetID: null,
    troops: null,
    clientID: AGENT_CLIENT_ID,
  };
  for (let i = 0; i < opts.ticks; i++) {
    const intents = i % opts.actEvery === 0 ? [expandIntent] : [];
    if (!episode.runTick(intents) || episode.isDone()) break;
  }

  if (opts.dumpRecord) {
    episode.writeReplayRecord(opts.dumpRecord);
    console.log(`Wrote replay record to ${opts.dumpRecord}`);
  }
  if (opts.dumpGameRecord) {
    fs.mkdirSync(opts.dumpGameRecord, { recursive: true });
    const gameID = episode.wireGameID();
    const outFile = path.join(opts.dumpGameRecord, `${gameID}.json`);
    fs.writeFileSync(outFile, JSON.stringify(episode.toGameRecord()));
    console.log(
      `Wrote GameRecord to ${outFile} — with ReplayServer.ts running, ` +
        `watch it at http://localhost:<vite-port>/game/${gameID}`,
    );
  }

  return {
    ticks: episode.game.ticks(),
    finalHash: episode.lastHash?.hash?.toString(),
    finalHashTick: episode.lastHash?.tick,
    winner: JSON.stringify(episode.lastWinner ?? null),
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
    `ticks=${r.ticks} finalHash=${r.finalHash ?? "n/a"} (tick ${r.finalHashTick ?? "n/a"}) winner=${r.winner}`,
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
      `spawnTurns=${opts.spawnTurns} actEvery=${opts.actEvery} difficulty=${opts.difficulty}`,
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
