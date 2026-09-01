import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { Executor } from "../../OpenFrontIO/src/core/execution/ExecutionManager";
import {
  Difficulty,
  Game,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  PlayerInfo,
  PlayerType,
} from "../../OpenFrontIO/src/core/game/Game";
import { createGame } from "../../OpenFrontIO/src/core/game/GameImpl";
import { GameMap } from "../../OpenFrontIO/src/core/game/GameMap";
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
import {
  GameConfig,
  StampedIntent,
  Turn,
  Winner,
} from "../../OpenFrontIO/src/core/Schemas";
import { EnvConfig } from "./EnvConfig";
import { NodeGameMapLoader } from "../../OpenFrontIO/tests/perf/fullgame/NodeGameMapLoader";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const TESTDATA_MAPS = path.join(
  __dirname,
  "../../OpenFrontIO/tests/testdata/maps",
);
const PRODUCTION_MAPS_DIR = path.join(
  __dirname,
  "../../OpenFrontIO/resources/maps",
);

export const AGENT_CLIENT_ID = "AGENT";
export const OPPONENT_CLIENT_ID = "OPPONENT";

interface ResolvedMap {
  gameMap: GameMap;
  miniGameMap: GameMap;
  /** The real GameMapType when mapName resolves to a production map under
   *  resources/maps/, or the Asia placeholder for a tests/testdata/ fixture
   *  (those aren't a real GameMapType, so OpenFrontIO's own replay/client
   *  tooling can't resolve matching terrain for them from gameConfig alone —
   *  fine for training, not for producing a watchable/replayable record). */
  gameMapType: GameMapType;
}

/**
 * Resolves `mapName` to terrain data. Tries tests/testdata/maps/<mapName>
 * first (tiny fixtures, fast — the default for training throughput), then
 * falls back to resources/maps/<mapName> (real production maps, matching a
 * GameMapType) — larger and slower, but their gameConfig.gameMap is real,
 * so episodes on them are replayable/watchable with OpenFrontIO's own
 * tooling. See training-map vs. replay-map guidance in the README.
 */
async function resolveMap(mapName: string): Promise<ResolvedMap> {
  const testMapDir = path.join(TESTDATA_MAPS, mapName);
  if (fs.existsSync(testMapDir)) {
    const mapBinBuffer = fs.readFileSync(path.join(testMapDir, "map.bin"));
    const miniMapBinBuffer = fs.readFileSync(
      path.join(testMapDir, "map4x.bin"),
    );
    const manifest = JSON.parse(
      fs.readFileSync(path.join(testMapDir, "manifest.json"), "utf8"),
    ) as MapManifest;
    return {
      gameMap: await genTerrainFromBin(manifest.map, mapBinBuffer),
      miniGameMap: await genTerrainFromBin(manifest.map4x, miniMapBinBuffer),
      gameMapType: GameMapType.Asia, // placeholder — not a real map
    };
  }

  const key = Object.keys(GameMapType).find(
    (k) => k.toLowerCase() === mapName.toLowerCase(),
  );
  if (key === undefined) {
    throw new Error(
      `unknown map "${mapName}": no tests/testdata/maps/${mapName}/ fixture ` +
        `and no matching GameMapType under resources/maps/`,
    );
  }
  const gameMapType = GameMapType[key as keyof typeof GameMapType];

  // Deliberately not TerrainMapLoader's loadTerrainMap(): it caches the
  // parsed GameMapImpl itself (module-level, keyed by map+size), and each
  // GameMapImpl's mutable per-tile ownership state is allocated once at
  // construction — reusing the cached object across two episodes leaks one
  // episode's conquered territory into the next (confirmed: a second
  // episode on the same map in one process desynced from tick 0, with both
  // players unable to spawn onto tiles the first episode already owned).
  // Re-parsing from the raw map bytes each episode keeps every episode's
  // ownership state isolated; only the raw bytes/manifest read is worth
  // caching here, and Node's own fs layer already caches small file reads.
  const mapData = new NodeGameMapLoader(PRODUCTION_MAPS_DIR).getMapData(
    gameMapType,
  );
  const manifest = await mapData.manifest();
  return {
    gameMap: await genTerrainFromBin(manifest.map, await mapData.mapBin()),
    miniGameMap: await genTerrainFromBin(
      manifest.map4x,
      await mapData.map4xBin(),
    ),
    gameMapType,
  };
}

/** Scans outward in a square spiral from (x0, y0) for the nearest land tile. */
export function findLandTile(game: Game, x0: number, y0: number): number {
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

export class Episode {
  runner: GameRunner;
  game: Game;
  lastHash: HashUpdate | undefined;
  lastWinner: Winner | undefined;
  fatalError: string | undefined;
  /** Every turn played, in order — enough to reconstruct/replay the episode. */
  turns: Turn[] = [];
  private turnNumber = 0;
  private seed: string;
  private gameConfig: GameConfig;
  private spawnTurns: number;

  private constructor(
    runner: GameRunner,
    seed: string,
    gameConfig: GameConfig,
    spawnTurns: number,
  ) {
    this.runner = runner;
    this.game = runner.game;
    this.seed = seed;
    this.gameConfig = gameConfig;
    this.spawnTurns = spawnTurns;
  }

  /**
   * Builds the game + runner (map, config, the two Human players) but plays
   * no turns — shared by create() (which then auto-spawns) and fromRecord()
   * (which replays a recorded turn log, spawn intents included, instead).
   */
  private static async build(
    mapName: string,
    seed: string,
    spawnTurns: number,
  ): Promise<Episode> {
    const { gameMap, miniGameMap, gameMapType } = await resolveMap(mapName);

    const gameConfig: GameConfig = {
      gameMap: gameMapType,
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
      new PlayerInfo(
        "Agent",
        PlayerType.Human,
        AGENT_CLIENT_ID,
        AGENT_CLIENT_ID,
      ),
      new PlayerInfo(
        "Opponent",
        PlayerType.Human,
        OPPONENT_CLIENT_ID,
        OPPONENT_CLIENT_ID,
      ),
    ];

    const config = new EnvConfig(gameConfig, new UserSettings(), spawnTurns);
    const game = createGame(humans, [], gameMap, miniGameMap, config);

    const episode = new Episode(
      new GameRunner(game, new Executor(game, seed, undefined), (gu) => {
        if ("errMsg" in gu) {
          episode.fatalError = `${gu.errMsg}\n${gu.stack ?? ""}`;
          return;
        }
        const hashes = gu.updates[GameUpdateType.Hash] as HashUpdate[];
        if (hashes.length > 0) episode.lastHash = hashes[hashes.length - 1];
        const wins = gu.updates[GameUpdateType.Win] as { winner: Winner }[];
        if (wins.length > 0) episode.lastWinner = wins[wins.length - 1].winner;
      }),
      seed,
      gameConfig,
      spawnTurns,
    );
    episode.runner.init();
    return episode;
  }

  static async create(
    mapName: string,
    seed: string,
    spawnTurns: number,
  ): Promise<Episode> {
    const episode = await Episode.build(mapName, seed, spawnTurns);
    const game = episode.game;

    // Spawn: both players pick a tile on the far sides of the map, first turn.
    const w = game.width();
    const h = game.height();
    const agentTile = findLandTile(
      game,
      Math.floor(w * 0.15),
      Math.floor(h * 0.5),
    );
    const opponentTile = findLandTile(
      game,
      Math.floor(w * 0.85),
      Math.floor(h * 0.5),
    );
    episode.runTick([
      { type: "spawn", tile: agentTile, clientID: AGENT_CLIENT_ID },
      { type: "spawn", tile: opponentTile, clientID: OPPONENT_CLIENT_ID },
    ]);

    const maxSpawnTurns = spawnTurns + 5;
    while (episode.game.inSpawnPhase()) {
      if (episode.turnNumber > maxSpawnTurns) {
        throw new Error(
          `spawn phase did not end after ${maxSpawnTurns} turns`,
        );
      }
      episode.runTick();
    }

    return episode;
  }

  /**
   * Replays a record written by writeReplayRecord() through a *fresh* game
   * built the same way create() builds one (same EnvConfig/spawnTurns/player
   * construction — see writeReplayRecord for why this can't reuse
   * OpenFrontIO's own ReplayGame.ts unmodified), and checks every recorded
   * turn.hash against the freshly recomputed hash at that tick.
   */
  static async fromRecord(record: {
    info: { gameID: string; config: GameConfig; envSpawnTurns: number };
    turns: Turn[];
  }): Promise<{
    episode: Episode;
    compared: number;
    matches: number;
    firstMismatch: number | null;
  }> {
    // Must reuse the exact original seed: SpawnExecution seeds its own
    // PseudoRandom from simpleHash(playerInfo.id) + simpleHash(gameID), which
    // shapes the conquered spawn area even for an explicit target tile — a
    // different seed here would diverge the very first hash checkpoint.
    const episode = await Episode.build(
      record.info.config.gameMap,
      record.info.gameID,
      record.info.envSpawnTurns,
    );

    let compared = 0;
    let matches = 0;
    let firstMismatch: number | null = null;
    for (const turn of record.turns) {
      const ok = episode.runTick(turn.intents);
      if (!ok) break;
      if (turn.hash !== undefined && turn.hash !== null) {
        compared++;
        const computed = episode.turns[episode.turns.length - 1].hash;
        if (computed === turn.hash) {
          matches++;
        } else if (firstMismatch === null) {
          firstMismatch = turn.turnNumber;
        }
      }
    }
    return { episode, compared, matches, firstMismatch };
  }

  /** Advances one tick, applying the given intents this turn. Throws on a fatal sim error. */
  runTick(intents: StampedIntent[] = []): boolean {
    const turn: Turn = { turnNumber: this.turnNumber++, intents };
    this.turns.push(turn);
    this.runner.addTurn(turn);
    const ok = this.runner.executeNextTick();
    if (this.fatalError !== undefined) {
      throw new Error(
        `game errored at tick ${this.game.ticks()}:\n${this.fatalError}`,
      );
    }
    // GameImpl emits a Hash update every 10 ticks (see GameImpl.executeNextTick),
    // stamped with the pre-increment tick count — game.ticks() has already
    // moved one past it by the time executeNextTick() returns here. Stamp it
    // onto this turn so a dumped record carries the same hashes ReplayGame.ts
    // checks the recomputed simulation against.
    if (
      this.lastHash !== undefined &&
      this.lastHash.tick === this.game.ticks() - 1
    ) {
      turn.hash = this.lastHash.hash;
    }
    return ok;
  }

  isDone(): boolean {
    return (
      this.lastWinner !== undefined ||
      !this.game.players().some((p) => p.isAlive())
    );
  }

  /**
   * Writes a JSON turn log of the episode so far. Verify it headlessly with
   * `Episode.fromRecord()` (see verifyRecord.ts) — deliberately NOT meant to
   * be loaded by OpenFrontIO's own `npm run replay:game`: that script always
   * reconstructs players via `random.nextID()` and a production `Config`,
   * neither of which matches how env-bridge builds an episode (fixed
   * "AGENT"/"OPPONENT" ids, EnvConfig's deterministic combat), so the two
   * would diverge from tick 0 even though both are internally deterministic.
   * `info`/`turns` are still GameRecord-*shaped* (same field names) so this
   * stays close to something OpenFrontIO's tooling could consume if the
   * player-construction mismatch is ever closed.
   */
  writeReplayRecord(filePath: string): void {
    const record = {
      info: {
        gameID: this.seed,
        lobbyCreatedAt: 0,
        config: this.gameConfig,
        envSpawnTurns: this.spawnTurns,
        players: [AGENT_CLIENT_ID, OPPONENT_CLIENT_ID].map((id) => ({
          clientID: id,
          username: id === AGENT_CLIENT_ID ? "Agent" : "Opponent",
          clanTag: null,
        })),
      },
      gitCommit: "DEV",
      version: "v0.0.2",
      turns: this.turns,
    };
    fs.writeFileSync(filePath, JSON.stringify(record));
  }
}
