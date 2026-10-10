/**
 * How much of each map can AGENT reach WITHOUT a boat?
 *
 * Land attacks stop at water: AttackExecution.addNeighbors() only walks
 * 4-neighbour land tiles. So the landmass a player spawns on (its connected
 * component of passable land) is everything it can ever take overland; the
 * rest needs a transport ship. This prints, per map:
 *
 *   land               passable land tiles (the engine's isLand && !isImpassable)
 *   components>=1%     landmasses holding at least 1% of the land
 *   largest            the biggest landmass's share of the land
 *   spawnShare         expected share of the land in the landmass a uniformly
 *                      random land tile falls in -- an approximation of
 *                      GameSetup.ts's random spawn, which draws a uniform
 *                      (x, y) and snaps it to the nearest land tile
 *
 * A map whose spawnShare is under 80% (FFA's percentageTilesOwnedToWin) cannot
 * be won overland from a typical spawn. This is what picked
 * training/envs/openfront_env.py's CONNECTED_MAPS.
 *
 * usage (from env-bridge/):
 *   npx tsx --tsconfig ../OpenFrontIO/tsconfig.json src/mapLandmasses.ts onion caucasus ...
 */
import path from "path";
import { fileURLToPath } from "url";
import { GameMapType } from "../../OpenFrontIO/src/core/game/Game";
import { genTerrainFromBin } from "../../OpenFrontIO/src/core/game/TerrainMapLoader";
import { NodeGameMapLoader } from "../../OpenFrontIO/tests/perf/fullgame/NodeGameMapLoader";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const loader = new NodeGameMapLoader(path.join(__dirname, "../../OpenFrontIO/resources/maps"));

async function landmasses(name: string): Promise<string> {
  const key = Object.keys(GameMapType).find((k) => k.toLowerCase() === name.toLowerCase());
  if (key === undefined) throw new Error(`unknown map "${name}"`);
  const data = loader.getMapData(GameMapType[key as keyof typeof GameMapType]);
  const manifest = await data.manifest();
  const map = await genTerrainFromBin(manifest.map, await data.mapBin());
  const w = map.width();
  const h = map.height();
  const passable = (t: number) => map.isLand(t) && !map.isImpassable(t);

  // Iterative flood fill -- the biggest maps have >1M land tiles, far past
  // any recursion limit.
  const seen = new Uint8Array(w * h);
  const sizes: number[] = [];
  const stack: number[] = [];
  for (let i = 0; i < w * h; i++) {
    if (seen[i] || !passable(i)) continue;
    let size = 0;
    seen[i] = 1;
    stack.push(i);
    while (stack.length > 0) {
      const t = stack.pop()!;
      size++;
      const x = t % w;
      const y = (t - x) / w;
      const nbrs = [x > 0 ? t - 1 : -1, x < w - 1 ? t + 1 : -1, y > 0 ? t - w : -1, y < h - 1 ? t + w : -1];
      for (const n of nbrs) {
        if (n < 0 || seen[n] || !passable(n)) continue;
        seen[n] = 1;
        stack.push(n);
      }
    }
    sizes.push(size);
  }
  sizes.sort((a, b) => b - a);
  const land = sizes.reduce((a, b) => a + b, 0);
  const big = sizes.filter((s) => s >= 0.01 * land);
  const spawnShare = sizes.reduce((acc, s) => acc + (s / land) ** 2, 0);
  const pct = (f: number) => `${(100 * f).toFixed(1)}%`;
  return (
    `${name.padEnd(18)} ${`${w}x${h}`.padEnd(10)} land=${String(land).padEnd(8)} ` +
    `components>=1%=${String(big.length).padEnd(3)} largest=${pct(sizes[0] / land).padEnd(6)} ` +
    `spawnShare=${pct(spawnShare)}`
  );
}

for (const name of process.argv.slice(2)) {
  console.log(await landmasses(name));
}
