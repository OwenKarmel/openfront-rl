/**
 * Headless replay verification for a turn log written by
 * Episode.writeReplayRecord() (see runEpisode.ts --dump-record,
 * EnvServer.ts's dumpRecord). Re-simulates the recorded turns through a
 * fresh episode and checks every recorded hash checkpoint against the
 * freshly recomputed one — the same kind of check OpenFrontIO's own
 * tests/replay/ReplayGame.ts does, but built on env-bridge's own episode
 * construction (see Episode.fromRecord for why the stock script can't be
 * reused unmodified here).
 *
 * Usage: npx tsx src/verifyRecord.ts <path/to/record.json>
 */
import fs from "fs";
import { Episode } from "./GameSetup";

async function main(): Promise<void> {
  const file = process.argv[2];
  if (!file) {
    throw new Error("usage: verifyRecord.ts <path/to/record.json>");
  }
  const record = JSON.parse(fs.readFileSync(file, "utf8"));

  console.log(
    `Replaying ${record.info.gameID}: ${record.info.config.gameMap} ` +
      `(${record.info.config.gameMapSize}), ${record.turns.length} turns`,
  );

  const { episode, compared, matches, firstMismatch } =
    await Episode.fromRecord(record);

  console.log(
    `Final state: ticks=${episode.game.ticks()} players=${episode.game
      .players()
      .map((p) => `${p.name()}(${p.numTilesOwned()} tiles)`)
      .join(", ")}`,
  );
  console.log(
    `Compared ${compared} hash checkpoints: ${matches} match, ${compared - matches} mismatch.`,
  );
  if (firstMismatch === null) {
    console.log("Replay is IN SYNC with the recorded episode.");
  } else {
    console.log(`Replay DIVERGED starting at turn ${firstMismatch}.`);
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
