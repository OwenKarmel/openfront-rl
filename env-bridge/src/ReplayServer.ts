/**
 * Tiny local HTTP server that lets the actual OpenFrontIO browser client
 * (running via `npm run dev` inside OpenFrontIO/) load a training episode
 * as a watchable replay.
 *
 * The client's only path for loading an archived game is
 * JoinLobbyModal.checkArchivedGame(): `GET {apiBase}/game/{gameID}`,
 * strictly validated with GameRecordSchema.safeParse (no fallback). This
 * server answers exactly that route from a directory of GameRecord JSON
 * files (written by Episode.toGameRecord() — see runEpisode.ts
 * --dump-game-record / EnvServer.ts's dumpGameRecord).
 *
 * getApiBase() (src/client/Api.ts) defaults to http://localhost:8787 for a
 * plain `npm run dev` client (audience "localhost", no API_DOMAIN/apiHost
 * override) — so running this server on port 8787 (the default here) means
 * the dev client finds it with zero client-side configuration. Visit
 * http://localhost:<vite-port>/game/<gameID> to watch.
 *
 * Usage: npx tsx src/ReplayServer.ts [--dir ../training/replays] [--port 8787]
 */
import fs from "fs";
import http from "http";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

interface Options {
  dir: string;
  port: number;
}

function parseArgs(argv: string[]): Options {
  const opts: Options = {
    dir: path.join(__dirname, "../../training/replays"),
    port: 8787,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`missing value for ${arg}`);
      return v;
    };
    switch (arg) {
      case "--dir":
        opts.dir = path.resolve(next());
        break;
      case "--port":
        opts.port = parseInt(next(), 10);
        break;
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }
  return opts;
}

function main(): void {
  const opts = parseArgs(process.argv.slice(2));
  fs.mkdirSync(opts.dir, { recursive: true });

  const server = http.createServer((req, res) => {
    // The client's fetches to apiBase use `credentials: "include"` (its real
    // auth cookie in production) for most routes, which the CORS spec
    // forbids combining with a wildcard Access-Control-Allow-Origin — must
    // echo back the exact requesting origin instead. The GET to /game/:id
    // itself doesn't need this (no credentials), but every *other*
    // apiBase-shaped call the client fires alongside it (auth/refresh,
    // cosmetics.json, streams.json, news.json, ...) does, and a CORS
    // rejection surfaces to the client as a hard network failure rather
    // than a handled 404 — so apply this to every route, not just /game/.
    const origin = req.headers.origin;
    if (origin) res.setHeader("Access-Control-Allow-Origin", origin);
    res.setHeader("Access-Control-Allow-Credentials", "true");
    res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    res.setHeader("Vary", "Origin");

    if (req.method === "OPTIONS") {
      res.writeHead(204).end();
      return;
    }

    const match = req.url?.match(/^\/game\/([^/?]+)/);
    if (req.method !== "GET" || !match) {
      // Everything else the client incidentally calls on apiBase (auth,
      // cosmetics, news, ...) — a clean 404 the client already handles
      // gracefully, now reachable instead of CORS-blocked.
      res.writeHead(404).end();
      return;
    }

    const gameID = match[1];
    const filePath = path.join(opts.dir, `${gameID}.json`);
    if (!filePath.startsWith(opts.dir) || !fs.existsSync(filePath)) {
      console.log(`GET /game/${gameID} -> 404 (no ${filePath})`);
      res.writeHead(404).end();
      return;
    }

    console.log(`GET /game/${gameID} -> 200`);
    res.setHeader("Content-Type", "application/json");
    res.writeHead(200).end(fs.readFileSync(filePath));
  });

  server.listen(opts.port, () => {
    console.log(`Replay server serving ${opts.dir} on http://localhost:${opts.port}`);
    console.log(`(this is the default apiHost a plain "npm run dev" client uses)`);
  });
}

main();
