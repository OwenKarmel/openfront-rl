import { UserSettings } from "../../OpenFrontIO/src/core/game/UserSettings";
import { GameConfig } from "../../OpenFrontIO/src/core/Schemas";
import { TestConfig } from "../../OpenFrontIO/tests/util/TestConfig";

/**
 * Config for headless RL episodes: same knobs as TestConfig (deterministic
 * attack attrition, zero spawn immunity, etc.) plus a short, settable spawn
 * phase so episodes don't burn hundreds of ticks waiting for players to pick
 * a spawn tile.
 */
export class EnvConfig extends TestConfig {
  private _numSpawnPhaseTurns: number;

  constructor(
    gameConfig: GameConfig,
    userSettings: UserSettings,
    numSpawnPhaseTurns: number,
  ) {
    super(gameConfig, userSettings, false);
    this._numSpawnPhaseTurns = numSpawnPhaseTurns;
  }

  numSpawnPhaseTurns(): number {
    return this._numSpawnPhaseTurns;
  }
}
