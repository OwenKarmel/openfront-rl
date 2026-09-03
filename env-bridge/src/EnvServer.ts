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
 * Action space (still intentionally small — richer intents like structure
 * builds and diplomacy come later, per the action-space-expansion plan):
 *   "noop"             — do nothing this decision step
 *   "expand"           — attack neutral (unowned) land bordering AGENT's territory
 *   "attack_opponent"  — attack OPPONENT directly across a shared border
 *   "boat_attack"      — send a transport ship across water to invade OPPONENT's
 *                         territory (needs macroX/macroY — see StepCmd)
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
import { Difficulty, Game, Player, UnitType } from "../../OpenFrontIO/src/core/game/Game";
import { targetTransportTile } from "../../OpenFrontIO/src/core/game/TransportShipUtils";
import { StampedIntent } from "../../OpenFrontIO/src/core/Schemas";
import { AGENT_CLIENT_ID, Episode } from "./GameSetup";

// boat_attack's target macro-cell is a cheap heuristic (see
// tileGridAndBoatMask() below), not the real canBuildTransportShip check --
// an unreachable pick is an expected, harmless outcome by design (the real
// engine still validates and no-ops it), not a bug to investigate every
// time. TransportShipExecution.init() logs it via console.warn on every
// occurrence (OpenFrontIO/src/core/execution/TransportShipExecution.ts,
// "cannot send ship to ... cannot find target/start tile"), which floods
// train_stdout.log at training volume. Filtered here at the env-bridge
// boundary -- not in OpenFrontIO's own source, so this stays a training-
// harness concern, not an engine change -- rather than in Python, since
// Node's console.warn already writes straight to stderr before it ever
// reaches the Python subprocess.
const originalConsoleWarn = console.warn;
console.warn = (...args: unknown[]) => {
  if (typeof args[0] === "string" && args[0].includes("cannot send ship to")) {
    return;
  }
  originalConsoleWarn(...args);
};

/**
 * Filesystem-safe, chronologically-sortable Eastern-time timestamp (e.g.
 * "2026-09-02T10-30-27-146"), used for dumped replay filenames -- see
 * flushRecord(). Formats via the IANA "America/New_York" zone rather than a
 * fixed UTC offset, so this is correct across the EST/EDT DST transition
 * automatically (the host machine's own clock stays UTC throughout -- this
 * only affects how a timestamp is *displayed* in a filename, never any
 * timing-sensitive game/PRNG logic, which stays on the host's real clock).
 */
function easternTimestamp(date: Date): string {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  }).formatToParts(date);
  const get = (type: string) => parts.find((p) => p.type === type)?.value ?? "00";
  const ms = String(date.getMilliseconds()).padStart(3, "0");
  return `${get("year")}-${get("month")}-${get("day")}T${get("hour")}-${get("minute")}-${get("second")}-${ms}`;
}

const ACTIONS = ["noop", "expand", "attack_opponent", "boat_attack"] as const;
type Action = (typeof ACTIONS)[number];

// Fixed-size macro-tile grid a spatial action (today: boat_attack's
// destination) picks a cell from, independent of the actual map's pixel
// dimensions -- same size-invariance trick the network's AdaptiveAvgPool2d
// already relies on. Full tile-resolution targeting would be far more
// precision than the game needs (the engine's own TransportShipExecution
// already snaps to the nearest legal shore when a BoatAttackIntent
// executes) and would make the per-step legality mask below cost far more
// to compute/transmit.
const MACRO_GRID = 32;

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
  // Only meaningful (and required) when action === "boat_attack": a cell in
  // the MACRO_GRID x MACRO_GRID grid, translated to a real tile server-side
  // by macroCellToTile().
  macroX?: number;
  macroY?: number;
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

/**
 * Ownership grid (0=neutral land, 1=AGENT, 2=OPPONENT, 3=water) AND the
 * boat-destination macro-mask, computed together in one pass over every
 * tile. Water used to be lumped into the same "0" bucket as neutral land,
 * which made a naval/boat-destination action unlearnable (the agent
 * couldn't distinguish "unclaimed land I could expand into" from "open
 * sea") -- split out as its own class now that boat_attack needs it.
 *
 * The boat mask is a CHEAP heuristic (any OPPONENT-owned tile in the
 * macro-cell -> legal), not the real targetTransportTile()/
 * closestReachableShore() check -- deliberately deferred, see
 * resolveBoatTarget()'s comment for why and for where the real check
 * actually happens (once, only for the one macro-cell chosen, only on the
 * step where boat_attack is the sampled type -- not here, on every step,
 * for every candidate cell). This mask is only ever used to (a) decide
 * whether "boat_attack" appears in legalActions at all, and (b) as the
 * network's tile_target_mask for sampling *which* cell to try -- both
 * tolerant of false positives, since resolveBoatTarget() is the actual
 * authority when an attempt is made.
 *
 * Returned as Uint8Arrays (base64-encoded on the wire by observation()
 * below), not plain number[]s -- profiling found that for a 512x512 map,
 * JSON-encoding the grid as 262144 individual array elements produced a
 * ~524KB line and cost ~10ms of Python-side json.loads plus ~5ms of numpy
 * postprocessing *per decision step* (measured: ~15ms of a ~22ms step, i.e.
 * the dominant cost, well above the actual simulation-tick time). A single
 * base64 string decodes via one C-level call on both ends instead of
 * parsing hundreds of thousands of JSON tokens.
 */
function tileGridAndBoatMask(episode: Episode): { grid: Uint8Array; boatMask: Uint8Array } {
  const game = episode.game;
  const map = game.map();
  const w = game.width();
  const h = game.height();
  const agent = player(game, episode.agentId);
  const opponent = player(game, episode.opponentId);
  const grid = new Uint8Array(w * h);
  const boatMask = new Uint8Array(Math.ceil((MACRO_GRID * MACRO_GRID) / 8));
  for (let y = 0; y < h; y++) {
    const my = Math.min(MACRO_GRID - 1, Math.floor((y * MACRO_GRID) / h));
    for (let x = 0; x < w; x++) {
      const ref = map.ref(x, y);
      if (map.isWater(ref)) {
        grid[y * w + x] = 3;
        continue;
      }
      const owner = game.owner(ref);
      if (owner === agent) {
        grid[y * w + x] = 1;
      } else if (owner === opponent) {
        grid[y * w + x] = 2;
        const mx = Math.min(MACRO_GRID - 1, Math.floor((x * MACRO_GRID) / w));
        const bitIdx = my * MACRO_GRID + mx;
        boatMask[bitIdx >> 3] |= 1 << (bitIdx & 7);
      } else {
        grid[y * w + x] = 0;
      }
    }
  }
  return { grid, boatMask };
}

/**
 * Resolves a sampled (macroX, macroY) boat_attack target to a real,
 * verified landing tile -- called from intentFor(), i.e. ONLY on the one
 * step where boat_attack is actually the sampled type, and ONLY for that
 * one chosen macro-cell, never all MACRO_GRID*MACRO_GRID=1024 of them.
 *
 * This replaces an earlier version of this fix that ran the real
 * targetTransportTile()/closestReachableShore() check for every
 * OPPONENT-bordering macro-cell inside tileGridAndBoatMask(), unconditionally,
 * on every single step regardless of which action ends up chosen (needed
 * so the returned tile_target_mask/observation would already reflect real
 * legality). Benchmarked: ~7-10x slower per step in isolation, and ~3-3.5x
 * slower per training update end-to-end (measured against this project's
 * live 8-env rollout: ~23-25s/update baseline -> ~72-87s/update) -- because
 * that cost was paid on every step, not just boat_attack steps, since
 * legalActions()/tile_target_mask have to be ready before the network even
 * samples an action. Deferring to here instead means the expensive part
 * only runs on steps where the sampled type is actually boat_attack (per
 * training's own diagnostics, roughly 5-30% of steps, not 100%), and even
 * then it's exactly ONE targetTransportTile() call, not O(candidate
 * macro-cells) of them.
 *
 * The tradeoff: the mask the network samples FROM (tileGridAndBoatMask()'s
 * boatMask) stays the cheap any-opponent-tile-in-cell heuristic, so a
 * sampled cell can still fail this real check (mirroring the original,
 * pre-fix false-legality gap) -- but now that's confined to "this one
 * attempt is a no-op" (see intentFor()'s handling of a null return here),
 * not "the whole per-step mask is expensive regardless of outcome". Given
 * training's own type_probs diagnostics show boat_attack sampled well
 * under 100% of steps, this trades a bounded amount of residual no-op risk
 * for most of the real check's cost back.
 *
 * Scans only the chosen macro-cell's own pixel rectangle (bounded by
 * roughly (w/MACRO_GRID)*(h/MACRO_GRID) tiles, not the whole map) for one
 * representative OPPONENT-owned tile, then runs the same real check
 * intentFor() used to always defer to when the mask alone said "legal".
 * Returns null if the macro-cell truly has no OPPONENT tile (stale/false
 * heuristic pick) or no real reachable shore near it.
 */
function resolveBoatTarget(episode: Episode, macroX: number, macroY: number): number | null {
  const game = episode.game;
  const map = game.map();
  const w = game.width();
  const h = game.height();
  const opponent = player(game, episode.opponentId);
  const agent = player(game, episode.agentId);
  const x0 = Math.floor((macroX * w) / MACRO_GRID);
  const x1 = Math.min(w, Math.floor(((macroX + 1) * w) / MACRO_GRID));
  const y0 = Math.floor((macroY * h) / MACRO_GRID);
  const y1 = Math.min(h, Math.floor(((macroY + 1) * h) / MACRO_GRID));
  let representative: number | null = null;
  for (let y = y0; y < y1 && representative === null; y++) {
    for (let x = x0; x < x1; x++) {
      const ref = map.ref(x, y);
      if (!map.isWater(ref) && game.owner(ref) === opponent) {
        representative = ref;
        break;
      }
    }
  }
  if (representative === null) return null;
  return targetTransportTile(game, agent, representative);
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

function observation(
  episode: Episode,
  width: number,
  height: number,
  grid: Uint8Array,
  boatMask: Uint8Array,
) {
  return {
    width,
    height,
    // Base64 of the raw Uint8Array bytes -- see tileGridAndBoatMask()'s comment.
    tileGridB64: Buffer.from(grid.buffer).toString("base64"),
    boatTargetMacroMaskB64: Buffer.from(boatMask.buffer).toString("base64"),
    players: {
      AGENT: playerObs(episode.game, episode.agentId),
      OPPONENT: playerObs(episode.game, episode.opponentId),
    },
  };
}

function legalActions(episode: Episode, boatMask: Uint8Array): Action[] {
  const agent = player(episode.game, episode.agentId);
  if (!agent.isAlive()) return ["noop"];
  if (episode.game.inSpawnPhase()) return ["noop"];
  const opponent = player(episode.game, episode.opponentId);
  const actions: Action[] = ["noop", "expand"];
  if (opponent.isAlive() && agent.sharesBorderWith(opponent)) {
    actions.push("attack_opponent");
  }
  if (
    agent.unitCount(UnitType.TransportShip) < episode.game.config().boatMaxNumber() &&
    boatMask.some((byte) => byte !== 0)
  ) {
    actions.push("boat_attack");
  }
  return actions;
}

/**
 * Potential function for reward shaping: AGENT's owned tile count MINUS
 * OPPONENT's, normalized by the map's total land tiles. Two deliberate
 * choices here, both found necessary empirically (not just architecturally
 * motivated):
 *
 * 1. Relative, not just AGENT's own count: with only AGENT's count,
 *    expanding into neutral land and attacking OPPONENT are reward-
 *    equivalent per tile gained, but attacking is strictly riskier
 *    (contested combat, troop losses) with no offsetting benefit — so the
 *    reward-maximizing policy has no incentive to ever fight, only to
 *    expand into neutral land forever (confirmed empirically: policy
 *    entropy collapsed to ~0 within ~10 updates under the old single-sided
 *    reward, converging fast onto "always expand, never attack").
 * 2. Normalized by total land tiles: raw tile counts on a real production
 *    map range into the tens of thousands, so *un*normalized deltas (even
 *    after fix 1) produced huge, map-size-dependent reward swings —
 *    confirmed empirically again: value_loss spiked into the thousands
 *    within ~10-20 updates, approx_kl spiked to ~0.27 (healthy PPO stays
 *    under ~0.02-0.05) in the same window, and entropy collapsed to ~0
 *    shortly after. Mechanism: huge value-loss gradients backpropagate
 *    through the trunk the policy head shares (see models/network.py),
 *    corrupting its features and destabilizing the policy despite PPO's
 *    ratio clipping, which doesn't protect against the *feature
 *    representation itself* shifting wildly underneath it. Normalizing by
 *    numLandTiles() bounds potential to roughly [-1, 1] regardless of map
 *    size, keeping per-step reward and discounted returns in a small,
 *    consistent range — the standard reward/return-normalization fix for
 *    this class of instability.
 */
function potential(episode: Episode): number {
  const agentTiles = player(episode.game, episode.agentId).numTilesOwned();
  const opponentTiles = player(episode.game, episode.opponentId).numTilesOwned();
  const totalLandTiles = episode.game.map().numLandTiles();
  return (agentTiles - opponentTiles) / totalLandTiles;
}

function intentFor(
  action: Action,
  episode: Episode,
  macroX: number | undefined,
  macroY: number | undefined,
): StampedIntent | null {
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
        targetID: episode.opponentId,
        troops: null,
        clientID: AGENT_CLIENT_ID,
      };
    case "boat_attack": {
      if (macroX === undefined || macroY === undefined) {
        throw new Error("boat_attack requires macroX/macroY");
      }
      // dst is the REAL verified landing shore, resolved HERE (not in
      // tileGridAndBoatMask()) -- see resolveBoatTarget()'s comment for why
      // this is deliberately deferred to only the step where boat_attack is
      // actually the sampled type, and only for this one chosen macro-cell.
      // null means the cheap heuristic mask picked a cell that doesn't
      // actually have a reachable target (no OPPONENT tile in it after all,
      // or one with no reachable shore nearby) -- training must never crash
      // over an env-side edge case like this, so it stays a harmless no-op
      // (like "noop") rather than throwing, consistent with the rest of
      // this module's stance on unreachable boat picks. Logged (not
      // silenced) so a persistently high rate would still be visible.
      const dst = resolveBoatTarget(episode, macroX, macroY);
      if (dst === null) {
        console.warn(
          `boat_attack macro-cell (${macroX},${macroY}) has no verified target -- treating as noop`,
        );
        return null;
      }
      const agent = player(episode.game, episode.agentId);
      const owner = episode.game.owner(dst);
      // Unlike AttackIntent, BoatAttackIntent.troops is a required
      // (non-nullable) float on the wire schema -- boatAttackAmount() is the
      // same default-sizing the engine's own Nation-bot boats use.
      const troops = episode.game.config().boatAttackAmount(agent, owner);
      return {
        type: "boat",
        troops,
        dst,
        clientID: AGENT_CLIENT_ID,
      };
    }
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
    const { grid, boatMask } = tileGridAndBoatMask(this.episode);
    return {
      obs: observation(this.episode, this.width, this.height, grid, boatMask),
      legalActions: legalActions(this.episode, boatMask),
      done: this.episode.isDone(),
    };
  }

  step(cmd: StepCmd): object {
    if (!this.episode) throw new Error("step called before reset");
    const episode = this.episode;

    const intent = intentFor(cmd.action, episode, cmd.macroX, cmd.macroY);
    const intents: StampedIntent[] = intent ? [intent] : [];

    let done = false;
    for (let i = 0; i < this.ticksPerStep; i++) {
      const ok = episode.runTick(i === 0 ? intents : []);
      if (!ok || episode.isDone()) {
        done = true;
        break;
      }
    }

    // Potential-based shaping (relative, map-size-normalized tile-count
    // delta -- see potential()), plus a terminal win/loss term once the
    // episode ends. Coefficient recalibrated after normalizing potential()
    // to roughly [-1, 1] (was 0.01 against *unbounded* raw tile counts,
    // which produced huge, map-size-dependent returns -- see potential()'s
    // comment); 5.0 keeps a fully map-dominant swing's cumulative shaping
    // reward in the same rough order of magnitude as before (bounded now,
    // not larger) while staying a meaningful dense signal relative to the
    // terminal ±1.
    const newPotential = potential(episode);
    let reward = (newPotential - this.prevPotential) * 5.0;
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

    const { grid, boatMask } = tileGridAndBoatMask(episode);
    return {
      obs: observation(episode, this.width, this.height, grid, boatMask),
      reward,
      done,
      legalActions: legalActions(episode, boatMask),
      info: { ticks: episode.game.ticks(), winner },
    };
  }

  /**
   * Writes a pending GameRecord dump requested on reset. Filename is
   * `<timestamp>_<gameID>.json`, not just `<gameID>.json` -- gameID is a
   * sha256-derived hash (wireGameID()), so bare-gameID filenames sort
   * essentially randomly in a directory listing, not chronologically. The
   * gameID must still be recoverable from the filename (ReplayServer.ts's
   * GET /game/:gameID route needs it, and so does the real client's PRNG
   * reseeding, which is keyed on this exact gameID -- see wireGameID()'s
   * comment), so it stays as a suffix rather than being dropped.
   *
   * timestamp is Eastern local time (the host machine's clock is correctly
   * UTC -- see easternTimestamp()'s comment), not UTC, purely for
   * readability when eyeballing a directory listing.
   */
  flushRecord(): void {
    if (this.episode && this.dumpGameRecordDir) {
      fs.mkdirSync(this.dumpGameRecordDir, { recursive: true });
      const gameID = this.episode.wireGameID();
      const timestamp = easternTimestamp(new Date());
      const outFile = path.join(this.dumpGameRecordDir, `${timestamp}_${gameID}.json`);
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
