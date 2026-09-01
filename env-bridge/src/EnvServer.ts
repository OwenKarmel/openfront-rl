/**
 * Env-bridge: a long-lived Node process that drives one headless OpenFrontIO
 * episode at a time and speaks a tiny newline-delimited JSON protocol over
 * stdin/stdout, so a Python `gymnasium.Env` can reset/step it like any other
 * RL environment. See training/envs/openfront_env.py.
 *
 * Single-agent: AGENT is the controlled player; OPPONENT is a real built-in
 * Nation-AI (NationExecution — the same code driving Nation bots in
 * production games, difficulty selectable per episode, drawn from the map's
 * real manifest — see GameSetup.ts) making its own decisions every tick.
 * There is nothing to send OPPONENT actions for.
 *
 * Action space (still intentionally small — richer intents come once the
 * network has spatial/entity action heads to point them with, e.g. to pick
 * an attack's boat destination or a structure's build tile):
 *   "noop"             — do nothing this decision step
 *   "expand"           — attack neutral (unowned) land bordering AGENT's territory
 *   "attack_opponent"  — attack OPPONENT directly across a shared border
 *
 * Decision cadence: one JSON "step" call advances `ticksPerStep` simulation
 * ticks (default 10), applying AGENT's chosen intent on the first of those
 * ticks — this mirrors how the built-in Nation AI reacts every few dozen
 * ticks rather than every tick, and keeps episode length manageable. The
 * Nation AI itself still ticks (and can act) every simulation tick in
 * between, on its own schedule, regardless of this cadence.
 *
 * Protocol (one JSON object per line each direction):
 *   -> {"cmd":"reset","seed":"...","map":"onion","difficulty":"impossible",
 *       "ticksPerStep":10,"dumpGameRecordDir":"/path/to/replays"}
 *   <- {"obs":{...},"legalActions":[...],"done":false}
 *   -> {"cmd":"step","action":"expand"}
 *   <- {"obs":{...},"reward":0.1,"done":false,"legalActions":[...],
 *       "info":{"ticks":123,"winner":null}}
 *   -> {"cmd":"close"}
 *
 * `info.winner` is "AGENT"/"OPPONENT"/null (still playing, or a truncated-
 * without-a-winner episode) — set only once `done` is true.
 *
 * `dumpGameRecordDir` (optional, on reset) writes a full, strictly
 * schema-valid GameRecord (Episode.toGameRecord()) to
 * <dir>/<wireGameID>.json once the episode ends (done:true from step) or
 * the session is closed, whichever comes first — watchable in the real
 * OpenFrontIO client via ReplayServer.ts, and verifiable headlessly with
 * OpenFrontIO's own `npm run replay:game` — see README.
 */
import fs from "fs";
import path from "path";
import readline from "readline";
import { Difficulty, Game, Player } from "../../OpenFrontIO/src/core/game/Game";
import { StampedIntent } from "../../OpenFrontIO/src/core/Schemas";
import { AGENT_CLIENT_ID, Episode } from "./GameSetup";

const ACTIONS = ["noop", "expand", "attack_opponent"] as const;
type Action = (typeof ACTIONS)[number];

interface ResetCmd {
  cmd: "reset";
  seed: string;
  map: string;
  difficulty?: string;
  ticksPerStep?: number;
  dumpGameRecordDir?: string;
}
interface StepCmd {
  cmd: "step";
  action: Action;
}
interface CloseCmd {
  cmd: "close";
}
type Cmd = ResetCmd | StepCmd | CloseCmd;

function parseDifficulty(name: string | undefined): Difficulty {
  if (name === undefined) return Difficulty.Medium;
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

interface PlayerObs {
  alive: boolean;
  tiles: number;
  troops: number;
  gold: number;
}

function player(game: Game, playerId: string): Player {
  return game.player(playerId);
}

/** Ownership grid: 0 = unowned/water, 1 = AGENT, 2 = OPPONENT. */
function tileGrid(episode: Episode): number[] {
  const game = episode.game;
  const map = game.map();
  const w = game.width();
  const h = game.height();
  const agent = player(game, episode.agentId);
  const opponent = player(game, episode.opponentId);
  const out = new Array<number>(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const ref = map.ref(x, y);
      const owner = game.owner(ref);
      out[y * w + x] = owner === agent ? 1 : owner === opponent ? 2 : 0;
    }
  }
  return out;
}

function playerObs(game: Game, playerId: string): PlayerObs {
  const p = player(game, playerId);
  return {
    alive: p.isAlive(),
    tiles: p.numTilesOwned(),
    troops: p.troops(),
    gold: Number(p.gold()),
  };
}

function observation(episode: Episode, width: number, height: number) {
  return {
    width,
    height,
    tileGrid: tileGrid(episode),
    players: {
      AGENT: playerObs(episode.game, episode.agentId),
      OPPONENT: playerObs(episode.game, episode.opponentId),
    },
  };
}

function legalActions(episode: Episode): Action[] {
  const agent = player(episode.game, episode.agentId);
  if (!agent.isAlive()) return ["noop"];
  if (episode.game.inSpawnPhase()) return ["noop"];
  const opponent = player(episode.game, episode.opponentId);
  const actions: Action[] = ["noop", "expand"];
  if (opponent.isAlive() && agent.sharesBorderWith(opponent)) {
    actions.push("attack_opponent");
  }
  return actions;
}

/** Potential function for reward shaping: AGENT's owned tile count. */
function potential(episode: Episode): number {
  return player(episode.game, episode.agentId).numTilesOwned();
}

function intentFor(action: Action, opponentId: string): StampedIntent | null {
  switch (action) {
    case "noop":
      return null;
    case "expand":
      return {
        type: "attack",
        targetID: null,
        troops: null,
        clientID: AGENT_CLIENT_ID,
      };
    case "attack_opponent":
      return {
        type: "attack",
        targetID: opponentId,
        troops: null,
        clientID: AGENT_CLIENT_ID,
      };
  }
}

class Session {
  episode: Episode | undefined;
  width = 0;
  height = 0;
  ticksPerStep = 10;
  prevPotential = 0;
  dumpGameRecordDir: string | undefined;

  async reset(cmd: ResetCmd): Promise<object> {
    this.episode = await Episode.create(
      cmd.map,
      cmd.seed,
      parseDifficulty(cmd.difficulty),
    );
    this.width = this.episode.game.width();
    this.height = this.episode.game.height();
    this.ticksPerStep = cmd.ticksPerStep ?? 10;
    this.dumpGameRecordDir = cmd.dumpGameRecordDir;
    this.prevPotential = potential(this.episode);
    return {
      obs: observation(this.episode, this.width, this.height),
      legalActions: legalActions(this.episode),
      done: this.episode.isDone(),
    };
  }

  step(cmd: StepCmd): object {
    if (!this.episode) throw new Error("step called before reset");
    const episode = this.episode;

    const intent = intentFor(cmd.action, episode.opponentId);
    const intents: StampedIntent[] = intent ? [intent] : [];

    let done = false;
    for (let i = 0; i < this.ticksPerStep; i++) {
      const ok = episode.runTick(i === 0 ? intents : []);
      if (!ok || episode.isDone()) {
        done = true;
        break;
      }
    }

    // Potential-based shaping (AGENT tile-count delta), plus a dominant
    // terminal win/loss term once the episode ends.
    const newPotential = potential(episode);
    let reward = (newPotential - this.prevPotential) * 0.01;
    this.prevPotential = newPotential;
    let winner: "AGENT" | "OPPONENT" | null = null;
    if (done) {
      const agentAlive = player(episode.game, episode.agentId).isAlive();
      const opponentAlive = player(episode.game, episode.opponentId).isAlive();
      if (agentAlive && !opponentAlive) {
        reward += 1;
        winner = "AGENT";
      } else if (!agentAlive && opponentAlive) {
        reward -= 1;
        winner = "OPPONENT";
      }
    }

    if (done) this.flushRecord();

    return {
      obs: observation(episode, this.width, this.height),
      reward,
      done,
      legalActions: legalActions(episode),
      info: { ticks: episode.game.ticks(), winner },
    };
  }

  /** Writes a pending GameRecord dump requested on reset. */
  flushRecord(): void {
    if (this.episode && this.dumpGameRecordDir) {
      fs.mkdirSync(this.dumpGameRecordDir, { recursive: true });
      const gameID = this.episode.wireGameID();
      const outFile = path.join(this.dumpGameRecordDir, `${gameID}.json`);
      fs.writeFileSync(outFile, JSON.stringify(this.episode.toGameRecord()));
      console.log(`Wrote GameRecord to ${outFile} (watch at /game/${gameID})`);
      this.dumpGameRecordDir = undefined;
    }
  }
}

async function main(): Promise<void> {
  console.debug = () => {}; // silence per-tick debug logging
  const session = new Session();
  const rl = readline.createInterface({ input: process.stdin });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let cmd: Cmd;
    try {
      cmd = JSON.parse(trimmed);
    } catch (err) {
      process.stdout.write(JSON.stringify({ error: `bad JSON: ${err}` }) + "\n");
      continue;
    }
    try {
      if (cmd.cmd === "reset") {
        const res = await session.reset(cmd);
        process.stdout.write(JSON.stringify(res) + "\n");
      } else if (cmd.cmd === "step") {
        const res = session.step(cmd);
        process.stdout.write(JSON.stringify(res) + "\n");
      } else if (cmd.cmd === "close") {
        session.flushRecord();
        process.exit(0);
      } else {
        process.stdout.write(
          JSON.stringify({ error: `unknown cmd: ${JSON.stringify(cmd)}` }) + "\n",
        );
      }
    } catch (err) {
      process.stdout.write(
        JSON.stringify({ error: err instanceof Error ? err.message : String(err) }) + "\n",
      );
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
