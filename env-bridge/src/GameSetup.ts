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

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const TESTDATA_MAPS = path.join(
  __dirname,
  "../../OpenFrontIO/tests/testdata/maps",
);

export const AGENT_CLIENT_ID = "AGENT";
export const OPPONENT_CLIENT_ID = "OPPONENT";

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

  private constructor(runner: GameRunner, seed: string, gameConfig: GameConfig) {
    this.runner = runner;
    this.game = runner.game;
    this.seed = seed;
    this.gameConfig = gameConfig;
  }

  static async create(
    mapName: string,
    seed: string,
    spawnTurns: number,
  ): Promise<Episode> {
    const mapDir = path.join(TESTDATA_MAPS, mapName);
    const mapBinBuffer = fs.readFileSync(path.join(mapDir, "map.bin"));
    const miniMapBinBuffer = fs.readFileSync(path.join(mapDir, "map4x.bin"));
    const manifest = JSON.parse(
      fs.readFileSync(path.join(mapDir, "manifest.json"), "utf8"),
    ) as MapManifest;

    const gameMap = await genTerrainFromBin(manifest.map, mapBinBuffer);
    const miniGameMap = await genTerrainFromBin(manifest.map4x, miniMapBinBuffer);

    const gameConfig: GameConfig = {
      gameMap: GameMapType.Asia, // placeholder; terrain is loaded directly above
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
    );
    episode.runner.init();

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
    return ok;
  }

  isDone(): boolean {
    return (
      this.lastWinner !== undefined ||
      !this.game.players().some((p) => p.isAlive())
    );
  }

  /**
   * Writes a GameRecord-shaped JSON of every turn played so far, loadable
   * with OpenFrontIO's own `npm run replay:game -- <path>` (headless replay
   * + hash verification against the hashes recorded here). Not a strictly
   * schema-valid GameRecord (no end-of-game stats), but ReplayGame.ts falls
   * back to replaying an object shape like this "as-is" when strict parsing
   * fails — good enough to inspect/verify what an episode actually did.
   */
  writeReplayRecord(filePath: string): void {
    const record = {
      info: {
        gameID: this.seed,
        lobbyCreatedAt: 0,
        config: this.gameConfig,
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
