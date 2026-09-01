/**
 * Phase-2 env-bridge: a long-lived Node process that drives one headless
 * OpenFrontIO episode at a time and speaks a tiny newline-delimited JSON
 * protocol over stdin/stdout, so a Python `gymnasium.Env` can reset/step it
 * like any other RL environment. See training/envs/openfront_env.py.
 *
 * Action space (v1, intentionally minimal — just enough to prove the loop
 * works end-to-end; richer intents come once the network has spatial/entity
 * action heads to point them with):
 *   "noop"   — do nothing this decision step
 *   "expand" — attack neutral (unowned) land bordering the player's territory
 *
 * Decision cadence: one JSON "step" call advances `ticksPerStep` simulation
 * ticks (default 10), applying each player's chosen intent on the first of
 * those ticks — this mirrors how the built-in Nation AI reacts every few
 * dozen ticks rather than every tick, and keeps episode length manageable.
 *
 * Protocol (one JSON object per line each direction):
 *   -> {"cmd":"reset","seed":"...","map":"plains","spawnTurns":3,"ticksPerStep":10,
 *       "dumpRecord":"/path/to/record.json"}
 *   <- {"obs":{...},"legalActions":{...},"done":false}
 *   -> {"cmd":"step","actions":{"AGENT":"expand","OPPONENT":"noop"}}
 *   <- {"obs":{...},"reward":{...},"done":false,"legalActions":{...},"info":{...}}
 *   -> {"cmd":"close"}
 *
 * `dumpRecord` (optional, on reset) writes every turn of the episode to a
 * replayable JSON file — see Episode.writeReplayRecord — once the episode
 * ends (done:true from step) or the session is closed, whichever comes
 * first.
 */
import readline from "readline";
import { Game, Player } from "../../OpenFrontIO/src/core/game/Game";
import { StampedIntent } from "../../OpenFrontIO/src/core/Schemas";
import { AGENT_CLIENT_ID, Episode, OPPONENT_CLIENT_ID } from "./GameSetup";

type Action = "noop" | "expand";
const PLAYERS = [AGENT_CLIENT_ID, OPPONENT_CLIENT_ID] as const;

interface ResetCmd {
  cmd: "reset";
  seed: string;
  map: string;
  spawnTurns?: number;
  ticksPerStep?: number;
  dumpRecord?: string;
}
interface StepCmd {
  cmd: "step";
  actions: Record<string, Action>;
}
interface CloseCmd {
  cmd: "close";
}
type Cmd = ResetCmd | StepCmd | CloseCmd;

interface PlayerObs {
  alive: boolean;
  tiles: number;
  troops: number;
  gold: number;
}

function playerByClientId(game: Game, clientId: string): Player {
  return game.player(clientId);
}

/** Ownership grid: 0 = unowned/water, 1 = AGENT, 2 = OPPONENT. */
function tileGrid(game: Game): number[] {
  const map = game.map();
  const w = game.width();
  const h = game.height();
  const agent = playerByClientId(game, AGENT_CLIENT_ID);
  const opponent = playerByClientId(game, OPPONENT_CLIENT_ID);
  const out = new Array<number>(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const ref = map.ref(x, y);
      const owner = game.owner(ref);
      out[y * w + x] =
        owner === agent ? 1 : owner === opponent ? 2 : 0;
    }
  }
  return out;
}

function playerObs(game: Game, clientId: string): PlayerObs {
  const p = playerByClientId(game, clientId);
  return {
    alive: p.isAlive(),
    tiles: p.numTilesOwned(),
    troops: p.troops(),
    gold: Number(p.gold()),
  };
}

function observation(game: Game, width: number, height: number) {
  return {
    width,
    height,
    tileGrid: tileGrid(game),
    players: {
      AGENT: playerObs(game, AGENT_CLIENT_ID),
      OPPONENT: playerObs(game, OPPONENT_CLIENT_ID),
    },
  };
}

function legalActions(episode: Episode): Record<string, Action[]> {
  const canExpand = !episode.game.inSpawnPhase();
  const forPlayer = (clientId: string): Action[] => {
    const alive = playerByClientId(episode.game, clientId).isAlive();
    if (!alive) return ["noop"];
    return canExpand ? ["noop", "expand"] : ["noop"];
  };
  return { AGENT: forPlayer(AGENT_CLIENT_ID), OPPONENT: forPlayer(OPPONENT_CLIENT_ID) };
}

/** Potential function for reward shaping: owned tiles, one term per player. */
function potential(game: Game, clientId: string): number {
  return playerByClientId(game, clientId).numTilesOwned();
}

function intentFor(action: Action, clientID: string): StampedIntent | null {
  if (action === "noop") return null;
  return { type: "attack", targetID: null, troops: null, clientID };
}

class Session {
  episode: Episode | undefined;
  width = 0;
  height = 0;
  ticksPerStep = 10;
  prevPotential: Record<string, number> = {};
  dumpRecordPath: string | undefined;

  async reset(cmd: ResetCmd): Promise<object> {
    this.episode = await Episode.create(
      cmd.map,
      cmd.seed,
      cmd.spawnTurns ?? 3,
    );
    this.width = this.episode.game.width();
    this.height = this.episode.game.height();
    this.ticksPerStep = cmd.ticksPerStep ?? 10;
    this.dumpRecordPath = cmd.dumpRecord;
    for (const p of PLAYERS) {
      this.prevPotential[p] = potential(this.episode.game, p);
    }
    return {
      obs: observation(this.episode.game, this.width, this.height),
      legalActions: legalActions(this.episode),
      done: this.episode.isDone(),
    };
  }

  step(cmd: StepCmd): object {
    if (!this.episode) throw new Error("step called before reset");
    const episode = this.episode;

    const intents: StampedIntent[] = [];
    for (const p of PLAYERS) {
      const action = cmd.actions[p] ?? "noop";
      const intent = intentFor(action, p);
      if (intent) intents.push(intent);
    }

    let done = false;
    for (let i = 0; i < this.ticksPerStep; i++) {
      const ok = episode.runTick(i === 0 ? intents : []);
      if (!ok || episode.isDone()) {
        done = true;
        break;
      }
    }

    const reward: Record<string, number> = {};
    for (const p of PLAYERS) {
      const newPotential = potential(episode.game, p);
      // Potential-based shaping (tile-count delta), plus a dominant terminal
      // win/loss term once the episode ends.
      let r = (newPotential - this.prevPotential[p]) * 0.01;
      this.prevPotential[p] = newPotential;
      if (done) {
        const alive = playerByClientId(episode.game, p).isAlive();
        const opponentId = p === AGENT_CLIENT_ID ? OPPONENT_CLIENT_ID : AGENT_CLIENT_ID;
        const opponentAlive = playerByClientId(episode.game, opponentId).isAlive();
        if (alive && !opponentAlive) r += 1;
        else if (!alive && opponentAlive) r -= 1;
      }
      reward[p] = r;
    }

    if (done) this.flushRecord();

    return {
      obs: observation(episode.game, this.width, this.height),
      reward,
      done,
      legalActions: legalActions(episode),
      info: { ticks: episode.game.ticks() },
    };
  }

  /** Writes the pending replay record (if any dumpRecord path was set on reset). */
  flushRecord(): void {
    if (this.episode && this.dumpRecordPath) {
      this.episode.writeReplayRecord(this.dumpRecordPath);
      this.dumpRecordPath = undefined;
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
