/**
 * Replays a single dumped GameRecord (training/replays/*.json) through the
 * real OpenFrontIO core simulation -- same construction as
 * FindMaxTerritory.ts / ReplayGame.ts's `npm run replay:game` (real Config,
 * real createNationsForGame/createGame, real Executor) -- and writes a
 * per-tick time series of AGENT (clientID "AGENTAAA") tile count to a CSV,
 * instead of just the peak/final summary FindMaxTerritory.ts reports.
 *
 * Same one-file-per-process discipline as FindMaxTerritory.ts, for the same
 * reason: loadTerrainMap() caches a mutable GameMapImpl at module scope,
 * keyed by map:size, so looping over many files in one process corrupts
 * every file after the first.
 *
 * NOTE ON RUNNING THIS: saved here for reference/reproducibility, but
 * Node's bare-specifier resolution walks up from *this file's own
 * directory* looking for node_modules, not from cwd -- there's no
 * node_modules above OpenFrontIO/ itself. To actually run it, copy it into
 * OpenFrontIO/tests/replay/ (same import paths as ReplayGame.ts/
 * FindMaxTerritory.ts, which live there) and run from OpenFrontIO/:
 *   npx tsx tests/replay/TerritoryTimeSeries.ts ../training/replays/<file>.json <out.csv>
 *
 * Writes <out.csv> with header "tick,tiles" and one row per simulated tick.
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

async function tileSeriesFor(filePath: string): Promise<{ tick: number; tiles: number }[]> {
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

  const series: { tick: number; tiles: number }[] = [];
  let agent = game.playerByClientID(AGENT_CLIENT_ID);
  let tick = 0;
  for (const turn of record.turns) {
    runner.addTurn(turn);
    if (!runner.executeNextTick()) break;
    tick++;
    agent ??= game.playerByClientID(AGENT_CLIENT_ID);
    series.push({ tick, tiles: agent?.numTilesOwned() ?? 0 });
  }
  return series;
}

async function main(): Promise<void> {
  console.debug = () => {};
  console.warn = () => {}; // silence expected "cannot send ship"/"cannot spawn" engine noise
  const file = process.argv[2];
  const outCsv = process.argv[3];
  if (!file || !outCsv) throw new Error("usage: TerritoryTimeSeries.ts <record.json> <out.csv>");
  const series = await tileSeriesFor(file);
  const lines = ["tick,tiles", ...series.map((s) => `${s.tick},${s.tiles}`)];
  fs.writeFileSync(outCsv, lines.join("\n") + "\n");
  console.log(`Wrote ${series.length} rows to ${outCsv}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
