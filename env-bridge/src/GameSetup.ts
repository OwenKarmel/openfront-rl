import crypto from "crypto";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { Executor } from "../../OpenFrontIO/src/core/execution/ExecutionManager";
import {
  Cell,
  Difficulty,
  Game,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Nation,
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
  GameRecord,
  PlayerRecord,
  StampedIntent,
  Turn,
  Winner,
} from "../../OpenFrontIO/src/core/Schemas";
import { createPartialGameRecord } from "../../OpenFrontIO/src/core/Util";
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

// Exactly 8 alphanumeric chars each — Schemas.ts's ID/MappedID (used for
// player clientIDs and gameIDs throughout, including StampedIntent.clientID
// on every recorded turn) requires `GAME_ID_REGEX = /^[A-Za-z0-9]{8}$/`.
// Only matters for toGameRecord()'s output (the real client strictly
// schema-validates it); writeReplayRecord()/verifyRecord.ts don't care, but
// using valid IDs everywhere avoids needing two different ID schemes.
export const AGENT_CLIENT_ID = "AGENTAAA";
export const OPPONENT_CLIENT_ID = "OPPONENT";

/**
 * Deterministically derives a schema-valid 8-char alphanumeric gameID (see
 * GAME_ID_REGEX) from an arbitrary episode seed string, so a training
 * episode's own seed (e.g. "onion-42", not 8 chars, has a hyphen) can still
 * be used as OpenFrontIO's wire gameID in toGameRecord()/ReplayServer.ts.
 */
export function wireGameID(seed: string): string {
  return crypto.createHash("sha256").update(seed).digest("hex").slice(0, 8);
}

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

/**
 * Scans outward in a square spiral from (x0, y0) for the nearest land tile.
 * Takes a bare GameMap (not Game) so it works both before game creation
 * (picking the opponent Nation's spawnCell hint, which the Nation
 * constructor needs) and after (picking the AGENT's spawn tile) — Game
 * extends GameMap, so a live Game satisfies this either way.
 */
export function findLandTile(map: GameMap, x0: number, y0: number): number {
  const w = map.width();
  const h = map.height();
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
  private readonly startTimeMs = Date.now();

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
   * Builds the game + runner (map, config, the AGENT human player and the
   * OPPONENT Nation-AI player) but plays no turns — shared by create()
   * (which then spawns AGENT and lets OPPONENT self-spawn) and fromRecord()
   * (which replays a recorded turn log, spawn intents included, instead).
   *
   * OPPONENT is a real built-in Nation AI (NationExecution — the same code
   * driving Nation bots in production games), not a scripted stand-in: it
   * decides its own attacks/builds/alliance behavior every tick based on
   * `difficulty`, exactly like the in-game "Impossible" opponent the whole
   * project is ultimately trying to beat. GameRunner.init() wires its
   * Execution automatically once game.config().spawnNations() is true (see
   * gameConfig.nations below) — see ExecutionManager.nationExecutions().
   */
  private static async build(
    mapName: string,
    seed: string,
    spawnTurns: number,
    difficulty: Difficulty,
  ): Promise<Episode> {
    const { gameMap, miniGameMap, gameMapType } = await resolveMap(mapName);

    const gameConfig: GameConfig = {
      gameMap: gameMapType,
      gameMapSize: GameMapSize.Normal,
      gameMode: GameMode.FFA,
      gameType: GameType.Public,
      difficulty,
      // Anything but "disabled" — see Config.spawnNations(). Count is moot:
      // we build the OPPONENT Nation ourselves below instead of letting
      // createNationsForGame() draw nations from the map manifest.
      nations: "default",
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
    ];
    const opponentSpawnCell = new Cell(
      Math.floor(gameMap.width() * 0.85),
      Math.floor(gameMap.height() * 0.5),
    );
    const nations = [
      new Nation(
        opponentSpawnCell,
        new PlayerInfo(
          "Opponent",
          PlayerType.Nation,
          null,
          OPPONENT_CLIENT_ID,
        ),
      ),
    ];

    const config = new EnvConfig(gameConfig, new UserSettings(), spawnTurns);
    const game = createGame(humans, nations, gameMap, miniGameMap, config);

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
    difficulty: Difficulty,
  ): Promise<Episode> {
    const episode = await Episode.build(mapName, seed, spawnTurns, difficulty);
    const game = episode.game;

    // AGENT picks a spawn tile on the opposite side of the map from the
    // Nation's spawnCell hint above; OPPONENT self-spawns via NationExecution.
    const agentTile = findLandTile(
      game,
      Math.floor(game.width() * 0.15),
      Math.floor(game.height() * 0.5),
    );
    episode.runTick([
      { type: "spawn", tile: agentTile, clientID: AGENT_CLIENT_ID },
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
      record.info.config.difficulty,
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

  /** The gameID toGameRecord() writes into the record — visit /game/<this> in the client. */
  wireGameID(): string {
    return wireGameID(this.seed);
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
        // OPPONENT is a Nation (added via createGame's `nations` array, not
        // the wire player list) — only AGENT is a "player" on this shape.
        players: [
          { clientID: AGENT_CLIENT_ID, username: "Agent", clanTag: null },
        ],
      },
      gitCommit: "DEV",
      version: "v0.0.2",
      turns: this.turns,
    };
    fs.writeFileSync(filePath, JSON.stringify(record));
  }

  /**
   * Builds a strictly schema-valid GameRecord (GameRecordSchema in
   * Schemas.ts) using the same `createPartialGameRecord` helper the real
   * server uses to archive games. Unlike writeReplayRecord()'s loose shape,
   * this is what the actual OpenFrontIO browser client requires: its only
   * path for loading an archived game (JoinLobbyModal.checkArchivedGame)
   * does `GET {apiBase}/game/{gameID}` and runs the strict
   * `GameRecordSchema.safeParse` on the response with no fallback — see
   * ReplayServer.ts, which serves records built by this method so a
   * training episode can actually be watched in the real client.
   */
  toGameRecord(): GameRecord {
    const players: PlayerRecord[] = [
      {
        clientID: AGENT_CLIENT_ID,
        username: "Agent",
        clanTag: null,
        persistentID: null,
        stats: {},
      },
    ];
    const partial = createPartialGameRecord(
      wireGameID(this.seed),
      this.gameConfig,
      players,
      this.turns,
      this.startTimeMs,
      Date.now(),
      this.lastWinner,
    );
    // gitCommit: "DEV" makes JoinLobbyModal.checkArchivedGame skip the
    // build-matches-record check entirely (it only compares when the
    // client's own build is non-DEV) — exactly matches a `npm run dev`
    // client, which also reports "DEV".
    return { ...partial, gitCommit: "DEV" };
  }
}
