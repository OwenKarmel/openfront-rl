/**
 * Replays a single dumped GameRecord (training/replays/*.json) through the
 * real OpenFrontIO core simulation -- the exact same construction
 * ReplayGame.ts's `npm run replay:game` uses (real Config, real
 * createNationsForGame/createGame, real Executor) -- and reports the
 * greatest tile count AGENT (clientID "AGENTAAA") reached at any point
 * during the episode, not just at the end (an episode can peak mid-game
 * before losing ground).
 *
 * Deliberately one file per process invocation: loadTerrainMap() has a
 * module-level cache keyed by map:size that returns the same MUTABLE
 * GameMapImpl across calls in one process. ReplayGame.ts's CLI usage never
 * hits this (one file per process already); an earlier version of this
 * script looped over many files in one process and got every file after
 * the first silently corrupted with leftover tile ownership from the
 * previous game (reported 0 tiles for files that were actually fine --
 * verified by cross-checking against `npm run replay:game`, which is
 * unaffected since it's also one-file-per-process).
 *
 * NOTE ON RUNNING THIS: saved here for reference/reproducibility, but
 * Node's bare-specifier resolution (zod, etc., used transitively by the
 * imported OpenFrontIO core modules) walks up from *this file's own
 * directory* looking for node_modules, not from cwd -- there's no
 * node_modules above OpenFrontIO/ itself. To actually run it, copy it into
 * OpenFrontIO/tests/replay/ (same import paths as ReplayGame.ts, which
 * lives there) and run from OpenFrontIO/:
 *   npx tsx tests/replay/FindMaxTerritory.ts ../training/replays/<file>.json
 *
 * Usage: RESULT line to stdout:
 *   RESULT <peakTiles> <finalTiles> <ticks> <ALIVE|DEAD> <filename>
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { Config } from "../../src/core/configuration/Config";
import { Executor } from "../../src/core/execution/ExecutionManager";
import { PlayerInfo, PlayerType } from "../../src/core/game/Game";
import { createGame } from "../../src/core/game/GameImpl";
import { createNationsForGame } from "../../src/core/game/NationCreation";
import { loadTerrainMap } from "../../src/core/game/TerrainMapLoader";
import { GameRunner } from "../../src/core/GameRunner";
import { PseudoRandom } from "../../src/core/PseudoRandom";
import { GameRecord, GameRecordSchema, GameStartInfo } from "../../src/core/Schemas";
import { decompressGameRecord, simpleHash, toWireGameStartInfo } from "../../src/core/Util";
import { NodeGameMapLoader } from "../perf/fullgame/NodeGameMapLoader";

const OPENFRONTIO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const AGENT_CLIENT_ID = "AGENTAAA";

async function maxTerritoryFor(
  filePath: string,
): Promise<{ maxTiles: number; finalTiles: number; turns: number; winner: string }> {
  const raw = JSON.parse(fs.readFileSync(filePath, "utf8"));
  const parsed = GameRecordSchema.safeParse(raw);
  const record: GameRecord = decompressGameRecord(parsed.success ? parsed.data : (raw as GameRecord));
  const info = record.info;

  const gameStart: GameStartInfo = toWireGameStartInfo({
    gameID: info.gameID,
    lobbyCreatedAt: info.lobbyCreatedAt,
    config: info.config,
    players: info.players,
    tribes: info.tribes,
  });

  const config = new Config(info.config, null, false);
  const mapLoader = new NodeGameMapLoader(path.join(OPENFRONTIO_ROOT, "resources/maps"));
  const terrain = await loadTerrainMap(info.config.gameMap, info.config.gameMapSize, mapLoader, false);
  const random = new PseudoRandom(simpleHash(gameStart.gameID));
  const humans = gameStart.players.map(
    (p) =>
      new PlayerInfo(
        p.username,
        PlayerType.Human,
        p.clientID,
        random.nextID(),
        p.isLobbyCreator ?? false,
        p.clanTag,
        p.friends ?? [],
        p.teamIndex ?? null,
      ),
  );
  const nations = createNationsForGame(gameStart, terrain.nations, terrain.additionalNations, humans.length, random);
  const game = createGame(humans, nations, terrain.gameMap, terrain.miniGameMap, config, terrain.teamGameSpawnAreas);

  const runner = new GameRunner(
    game,
    new Executor(game, gameStart.gameID, undefined, gameStart.tribes?.map((t) => t.name)),
    () => {},
  );
  runner.init();

  // AGENT doesn't exist as a Player until its spawn intent (turn 0) is
  // processed, so playerByClientID() returns null until then -- look it up
  // lazily inside the loop rather than before any turns run.
  let maxTiles = 0;
  let agent = game.playerByClientID(AGENT_CLIENT_ID);
  for (const turn of record.turns) {
    runner.addTurn(turn);
    if (!runner.executeNextTick()) break;
    agent ??= game.playerByClientID(AGENT_CLIENT_ID);
    if (agent) {
      const tiles = agent.numTilesOwned();
      if (tiles > maxTiles) maxTiles = tiles;
    }
  }
  return {
    maxTiles,
    finalTiles: agent?.numTilesOwned() ?? 0,
    turns: record.turns.length,
    winner: agent?.isAlive() ? "ALIVE" : "DEAD",
  };
}

async function main(): Promise<void> {
  console.debug = () => {};
  console.warn = () => {}; // silence expected "cannot send ship"/"cannot spawn" engine noise
  const file = process.argv[2];
  if (!file) throw new Error("usage: FindMaxTerritory.ts <record.json>");
  const r = await maxTerritoryFor(file);
  console.log(`RESULT ${r.maxTiles} ${r.finalTiles} ${r.turns} ${r.winner} ${path.basename(file)}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
