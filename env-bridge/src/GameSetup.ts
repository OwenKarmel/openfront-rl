import crypto from "crypto";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { Config } from "../../OpenFrontIO/src/core/configuration/Config";
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
import { GameMap } from "../../OpenFrontIO/src/core/game/GameMap";
import { createGame } from "../../OpenFrontIO/src/core/game/GameImpl";
import {
  GameUpdateType,
  HashUpdate,
} from "../../OpenFrontIO/src/core/game/GameUpdates";
import { createNationsForGame } from "../../OpenFrontIO/src/core/game/NationCreation";
import {
  AdditionalNation,
  genTerrainFromBin,
  MapManifest,
  Nation as ManifestNation,
} from "../../OpenFrontIO/src/core/game/TerrainMapLoader";
import { GameRunner } from "../../OpenFrontIO/src/core/GameRunner";
import { PseudoRandom } from "../../OpenFrontIO/src/core/PseudoRandom";
import {
  GameConfig,
  GameRecord,
  GameStartInfo,
  PlayerRecord,
  StampedIntent,
  Turn,
  Winner,
} from "../../OpenFrontIO/src/core/Schemas";
import { simpleHash } from "../../OpenFrontIO/src/core/Util";
import { createPartialGameRecord } from "../../OpenFrontIO/src/core/Util";
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

// Exactly 8 alphanumeric chars — Schemas.ts's GAME_ID_REGEX requires this
// for gameIDs. Never used to seed anything (only the internal small id
// PseudoRandom.nextID() returns feeds determinism — see wireGameID below),
// so a fixed constant is fine.
export const AGENT_CLIENT_ID = "AGENTAAA";

interface ResolvedMap {
  gameMap: GameMap;
  miniGameMap: GameMap;
  nations: ManifestNation[];
  additionalNations: AdditionalNation[];
  /**
   * The real GameMapType when mapName resolves to a production map under
   * resources/maps/, or the Asia placeholder for a tests/testdata/ fixture
   * (those aren't a real GameMapType, and have no manifest nations, so
   * they're only usable for fast wiring smoke-tests — never for training
   * or eval episodes, which need a real map for both correct nation
   * construction and for OpenFrontIO's own replay/client tooling to
   * resolve matching terrain from gameConfig alone).
   */
  gameMapType: GameMapType;
}

/**
 * Resolves `mapName` to terrain + manifest data. Tries
 * tests/testdata/maps/<mapName> first (tiny fixtures, no real GameMapType
 * or manifest nations — smoke-test only), then resources/maps/<mapName>
 * (real production maps, matching a GameMapType, with real manifest
 * nations) — the only maps that should back a training or eval episode.
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
      nations: manifest.nations ?? [],
      additionalNations: manifest.additionalNations ?? [],
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
    nations: manifest.nations ?? [],
    additionalNations: manifest.additionalNations ?? [],
    gameMapType,
  };
}

/** Scans outward in a square spiral from (x0, y0) for the nearest land tile. */
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

/**
 * Deterministically derives a schema-valid 8-char alphanumeric gameID (see
 * GAME_ID_REGEX in Schemas.ts) from an arbitrary episode seed string (e.g.
 * "onion-42" — not 8 chars, has a hyphen). This is the ONE gameID used
 * everywhere for a given episode: it seeds the live simulation's PRNG
 * *and* is what gets written into the dumped GameRecord — using two
 * different values for those (as an earlier version of this file did) means
 * the client reseeds its own PRNG from a different value than the
 * simulation actually ran on and immediately diverges.
 */
export function wireGameID(seed: string): string {
  return crypto.createHash("sha256").update(seed).digest("hex").slice(0, 8);
}

export class Episode {
  runner: GameRunner;
  game: Game;
  /**
   * Internal PlayerID game-state lookups need (game.player(id),
   * sharesBorderWith, etc.) — NOT the same as AGENT_CLIENT_ID, which is the
   * wire clientID intents are stamped with (Executor resolves those via
   * playerByClientID separately). Production derives this via
   * PseudoRandom.nextID(), same as here, so it's not a fixed constant.
   */
  agentId: string;
  /** Resolved once, after construction: game.nations()[0]'s internal player id. */
  opponentId: string;
  lastHash: HashUpdate | undefined;
  lastWinner: Winner | undefined;
  fatalError: string | undefined;
  /** Every turn played, in order — enough to reconstruct/replay the episode. */
  turns: Turn[] = [];
  private turnNumber = 0;
  private gameID: string;
  private gameConfig: GameConfig;
  private readonly startTimeMs = Date.now();

  private constructor(
    runner: GameRunner,
    gameID: string,
    gameConfig: GameConfig,
    agentId: string,
    opponentId: string,
  ) {
    this.runner = runner;
    this.game = runner.game;
    this.gameID = gameID;
    this.gameConfig = gameConfig;
    this.agentId = agentId;
    this.opponentId = opponentId;
  }

  /**
   * Builds the game + runner (map, config, the AGENT human player, and one
   * Nation-AI opponent drawn from the map's real manifest) but plays no
   * turns — create() then spawns AGENT and lets OPPONENT self-spawn.
   *
   * Mirrors OpenFrontIO's own createGameRunner() (src/core/GameRunner.ts)
   * step-for-step — the literal function the real client uses for both
   * live play and loading an archived GameRecord — rather than
   * hand-approximating it, so a dumped episode reconstructs identically
   * (map, Config, player/nation ids, PRNG draw order) wherever it's loaded.
   * Not a direct call to createGameRunner()/loadTerrainMap() because that
   * caches the parsed (and mutable) GameMapImpl across calls — see
   * resolveMap().
   */
  private static async build(
    mapName: string,
    seed: string,
    difficulty: Difficulty,
  ): Promise<Episode> {
    const gameID = wireGameID(seed);
    const { gameMap, miniGameMap, nations, additionalNations, gameMapType } =
      await resolveMap(mapName);

    const gameConfig: GameConfig = {
      gameMap: gameMapType,
      gameMapSize: GameMapSize.Normal,
      gameMode: GameMode.FFA,
      // The real mode a solo player uses against AI in production
      // (SinglePlayerModal.startGame()) — not cosmetic: it's what makes
      // Config.numSpawnPhaseTurns() return the real 100-tick value (vs.
      // 300 for Public) and what makes GameRunner.init() skip adding a
      // SpawnTimerExecution a real solo game never has either.
      gameType: GameType.Singleplayer,
      difficulty,
      // Numeric, not "default": exactly one Nation-AI opponent, matching
      // this project's v1 scope (beat one bot 1v1). createNationsForGame
      // draws it (deterministically, given the seed) from the map's real
      // manifest nations below.
      nations: 1,
      donateGold: false,
      donateTroops: false,
      bots: 0,
      infiniteGold: false,
      infiniteTroops: false,
      instantBuild: false,
      randomSpawn: false,
    };

    const gameStart: GameStartInfo = {
      gameID,
      lobbyCreatedAt: 0,
      config: gameConfig,
      players: [
        {
          clientID: AGENT_CLIENT_ID,
          username: "Agent",
          clanTag: null,
        },
      ],
    };

    // Exact construction order from createGameRunner() (GameRunner.ts) —
    // order matters, since both draws come from the same PRNG stream.
    const random = new PseudoRandom(simpleHash(gameID));
    const humans = gameStart.players.map(
      (p) =>
        new PlayerInfo(
          p.username,
          PlayerType.Human,
          p.clientID,
          random.nextID(),
        ),
    );
    const nationList = createNationsForGame(
      gameStart,
      nations,
      additionalNations,
      humans.length,
      random,
    );

    const config = new Config(gameConfig, null, false);
    const game = createGame(humans, nationList, gameMap, miniGameMap, config);
    const agentId = humans[0].id;
    const opponentId = game.nations()[0].playerInfo.id;

    const episode = new Episode(
      new GameRunner(game, new Executor(game, gameID, undefined), (gu) => {
        if ("errMsg" in gu) {
          episode.fatalError = `${gu.errMsg}\n${gu.stack ?? ""}`;
          return;
        }
        const hashes = gu.updates[GameUpdateType.Hash] as HashUpdate[];
        if (hashes.length > 0) episode.lastHash = hashes[hashes.length - 1];
        const wins = gu.updates[GameUpdateType.Win] as { winner: Winner }[];
        if (wins.length > 0) episode.lastWinner = wins[wins.length - 1].winner;
      }),
      gameID,
      gameConfig,
      agentId,
      opponentId,
    );
    episode.runner.init();
    return episode;
  }

  static async create(
    mapName: string,
    seed: string,
    difficulty: Difficulty,
  ): Promise<Episode> {
    const episode = await Episode.build(mapName, seed, difficulty);
    const game = episode.game;

    // AGENT picks a spawn tile on one side of the map; OPPONENT self-spawns
    // via NationExecution, near its manifest-derived spawnCell.
    const agentTile = findLandTile(
      game,
      Math.floor(game.width() * 0.15),
      Math.floor(game.height() * 0.5),
    );
    episode.runTick([
      { type: "spawn", tile: agentTile, clientID: AGENT_CLIENT_ID },
    ]);

    // Real numSpawnPhaseTurns() (100 for Singleplayer) — not shortened.
    // Nothing strategic happens during it beyond idle waiting (EnvServer.ts
    // already forces legalActions() = ["noop"] while inSpawnPhase()), so
    // this only costs cheap headless executeNextTick() calls, not real
    // agent-decision overhead.
    const maxSpawnTurns = episode.game.config().numSpawnPhaseTurns() + 10;
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
    // onto this turn so a dumped record carries the same hashes
    // OpenFrontIO's replay tooling checks the recomputed simulation against.
    if (
      this.lastHash !== undefined &&
      this.lastHash.tick === this.game.ticks() - 1
    ) {
      turn.hash = this.lastHash.hash;
    }
    return ok;
  }

  /** The gameID this episode ran on — also what's written into toGameRecord(). */
  wireGameID(): string {
    return this.gameID;
  }

  isDone(): boolean {
    return (
      this.lastWinner !== undefined ||
      !this.game.players().some((p) => p.isAlive())
    );
  }

  /**
   * Builds a strictly schema-valid GameRecord (GameRecordSchema in
   * Schemas.ts) using the same `createPartialGameRecord` helper the real
   * server uses to archive games. This is what the actual OpenFrontIO
   * browser client requires: its only path for loading an archived game
   * (JoinLobbyModal.checkArchivedGame) does `GET {apiBase}/game/{gameID}`
   * and runs the strict `GameRecordSchema.safeParse` on the response with
   * no fallback — see ReplayServer.ts, which serves records built by this
   * method so a training/eval episode can be watched in the real client.
   * Since construction now matches createGameRunner() exactly, this same
   * record also verifies correctly with OpenFrontIO's own unmodified
   * `npm run replay:game`.
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
      this.gameID,
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
