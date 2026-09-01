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
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { Executor } from "../../OpenFrontIO/src/core/execution/ExecutionManager";
import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  PlayerInfo,
  PlayerType,
} from "../../OpenFrontIO/src/core/game/Game";
import { createGame } from "../../OpenFrontIO/src/core/game/GameImpl";
import {
  GameUpdateType,
  HashUpdate,
} from "../../OpenFrontIO/src/core/game/GameUpdates";
import {
  genTerrainFromBin,
  MapManifest,
} from "../../OpenFrontIO/src/core/game/TerrainMapLoader";
import { UserSettings } from "../../OpenFrontIO/src/core/game/UserSettings";
import { GameRunner } from "../../OpenFrontIO/src/core/GameRunner";
import { GameConfig, StampedIntent, Turn } from "../../OpenFrontIO/src/core/Schemas";
import { EnvConfig } from "./EnvConfig";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TESTDATA_MAPS = path.join(
  __dirname,
  "../../OpenFrontIO/tests/testdata/maps",
);

const AGENT_CLIENT_ID = "AGENT";
const OPPONENT_CLIENT_ID = "OPPONENT";

interface Options {
  map: string;
  seed: string;
  ticks: number;
  spawnTurns: number;
  actEvery: number;
}

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    map: "plains",
    seed: "smoke-1",
    ticks: 300,
    spawnTurns: 3,
    actEvery: 10,
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

/** Scans outward in a square spiral from (x0, y0) for the nearest land tile. */
function findLandTile(
  game: ReturnType<typeof createGame>,
  x0: number,
  y0: number,
): number {
  const map = game.map();
  const w = game.width();
  const h = game.height();
  for (let r = 0; r < Math.max(w, h); r++) {
    for (let dx = -r; dx <= r; dx++) {
      for (let dy = -r; dy <= r; dy++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
        const x = x0 + dx;
        const y = y0 + dy;
        if (x < 0 || x >= w || y < 0 || y >= h) continue;
        const ref = map.ref(x, y);
        if (map.isLand(ref)) return ref;
      }
    }
  }
  throw new Error("no land tile found on map");
}

async function runEpisode(opts: Options): Promise<EpisodeResult> {
  console.debug = () => {}; // silence per-tick debug logging

  const mapDir = path.join(TESTDATA_MAPS, opts.map);
  const mapBinBuffer = fs.readFileSync(path.join(mapDir, "map.bin"));
  const miniMapBinBuffer = fs.readFileSync(path.join(mapDir, "map4x.bin"));
  const manifest = JSON.parse(
    fs.readFileSync(path.join(mapDir, "manifest.json"), "utf8"),
  ) as MapManifest;

  const gameMap = await genTerrainFromBin(manifest.map, mapBinBuffer);
  const miniGameMap = await genTerrainFromBin(manifest.map4x, miniMapBinBuffer);

  const gameConfig: GameConfig = {
    // Placeholder: irrelevant once terrain is loaded directly from testdata,
    // but the field is required by the schema.
    gameMap: GameMapType.Asia,
    gameMapSize: GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Public,
    difficulty: Difficulty.Medium,
    nations: "disabled",
    donateGold: false,
    donateTroops: false,
    bots: 0,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: false,
  };

  const humans = [
    new PlayerInfo("Agent", PlayerType.Human, AGENT_CLIENT_ID, AGENT_CLIENT_ID),
    new PlayerInfo(
      "Opponent",
      PlayerType.Human,
      OPPONENT_CLIENT_ID,
      OPPONENT_CLIENT_ID,
    ),
  ];

  const config = new EnvConfig(gameConfig, new UserSettings(), opts.spawnTurns);
  const game = createGame(humans, [], gameMap, miniGameMap, config);

  let lastHash: HashUpdate | undefined;
  let fatalError: string | undefined;
  const runner = new GameRunner(
    game,
    new Executor(game, opts.seed, undefined),
    (gu) => {
      if ("errMsg" in gu) {
        fatalError = `${gu.errMsg}\n${gu.stack ?? ""}`;
        return;
      }
      const hashes = gu.updates[GameUpdateType.Hash] as HashUpdate[];
      if (hashes.length > 0) {
        lastHash = hashes[hashes.length - 1];
      }
    },
  );
  runner.init();

  let turnNumber = 0;
  const runTick = (intents: StampedIntent[] = []): boolean => {
    const turn: Turn = { turnNumber: turnNumber++, intents };
    runner.addTurn(turn);
    const ok = runner.executeNextTick();
    if (fatalError !== undefined) {
      throw new Error(`game errored at tick ${game.ticks()}:\n${fatalError}`);
    }
    return ok;
  };

  // Spawn: both players pick tiles on the very first turn.
  const w = game.width();
  const h = game.height();
  const agentTile = findLandTile(game, Math.floor(w * 0.15), Math.floor(h * 0.5));
  const opponentTile = findLandTile(
    game,
    Math.floor(w * 0.85),
    Math.floor(h * 0.5),
  );
  runTick([
    { type: "spawn", tile: agentTile, clientID: AGENT_CLIENT_ID },
    { type: "spawn", tile: opponentTile, clientID: OPPONENT_CLIENT_ID },
  ]);

  const maxSpawnTurns = opts.spawnTurns + 5;
  while (game.inSpawnPhase()) {
    if (turnNumber > maxSpawnTurns) {
      throw new Error(`spawn phase did not end after ${maxSpawnTurns} turns`);
    }
    runTick();
  }

  // Main phase: periodically expand into neutral land so territory changes.
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
    if (!runTick(intents)) break;
  }

  return {
    ticks: game.ticks(),
    finalHash: lastHash?.hash?.toString(),
    finalHashTick: lastHash?.tick,
    players: game.players().map((p) => ({
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
