import { Router, type IRouter, type Request, type Response } from "express";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const router: IRouter = Router();

// -- Helpers ------------------------------------------------------------------

/** Walk up from cwd until version.json is found -- that's the project root. */
function findProjectRoot(): string {
  let dir = process.cwd();
  for (let i = 0; i < 6; i++) {
    try { readFileSync(resolve(dir, "version.json")); return dir; } catch { /* keep going */ }
    const parent = resolve(dir, "..");
    if (parent === dir) break;
    dir = parent;
  }
  return process.cwd();
}

interface VersionFile {
  version: string;
  releaseDate?: string;
  changelog?: string[];
  history?: Array<{ version: string; releaseDate?: string; changelog?: string[] }>;
}

function readVersionFile(): VersionFile {
  try {
    const raw = readFileSync(resolve(findProjectRoot(), "version.json"), "utf-8");
    const parsed = JSON.parse(raw) as { version?: string };
    if ((parsed as VersionFile).version) return parsed as VersionFile;
  } catch { /* fall through */ }
  return { version: "unknown" };
}

/**
 * Resolve version-check URL from UPDATE_SOURCE_URL.
 *
 * Supports two formats:
 *   * GitHub repo  -- https://github.com/owner/repo
 *       version : https://raw.githubusercontent.com/owner/repo/main/version.json
 *   * Server URL   -- https://my-proxy.replit.app  (legacy / self-hosted)
 *       version : {url}/api/version
 */
function resolveVersionUrl(base: string): string {
  const clean = base.replace(/\/$/, "");
  const ghMatch = clean.match(/^https?:\/\/github\.com\/([^/]+\/[^/]+)/);
  if (ghMatch) {
    const slug = ghMatch[1].replace(/\.git$/, "");
    return `https://raw.githubusercontent.com/${slug}/main/version.json`;
  }
  return `${clean}/api/version`;
}

function parseVerSeg(seg: string): number {
  const n = parseInt(seg, 10);
  return isNaN(n) ? 0 : n;
}

// Versioning scheme: MAJOR.MINOR.PATCH[.BUILD] -- all segments are plain integers.
// Hotfixes increment BUILD (e.g. 1.1.2 -> 1.1.2.1 -> 1.1.2.2).
// Minor releases increment PATCH and reset BUILD (e.g. 1.1.3).
// Returns >0 if a is newer, <0 if b is newer, 0 if equal.
function compareVersions(a: string, b: string): number {
  if (a === b) return 0;
  const pa = a.split(".").map(parseVerSeg);
  const pb = b.split(".").map(parseVerSeg);
  const len = Math.max(pa.length, pb.length);
  for (let i = 0; i < len; i++) {
    const na = pa[i] ?? 0, nb = pb[i] ?? 0;
    if (na > nb) return 1; if (na < nb) return -1;
  }
  return 0;
}

// -- GET /api/version ---------------------------------------------------------

router.get("/version", async (_req: Request, res: Response) => {
  const local = readVersionFile();
  const updateSourceUrl = process.env["UPDATE_SOURCE_URL"] ?? "https://github.com/Akatsuki03/AI-Monorepo";

  let remote: { version: string; releaseDate?: string; changelog?: string[] } | null = null;
  let updateAvailable = false;

  if (updateSourceUrl) {
    try {
      const versionUrl = resolveVersionUrl(updateSourceUrl);
      const r = await fetch(versionUrl, { signal: AbortSignal.timeout(5000) });
      if (r.ok) {
        const body = await r.json() as { current?: typeof remote };
        remote = (body.current ?? body) as typeof remote;
        if (remote && local.version !== "unknown") {
          updateAvailable = compareVersions(remote.version ?? "0", local.version) > 0;
        }
      }
    } catch { /* remote check failed -- not a hard error */ }
  }

  res.json({ current: local, remote, updateAvailable, updateSourceUrl: updateSourceUrl ?? null });
});

// -- GET /api/source.tar.gz ---------------------------------------------------
// Serves the project source as a gzipped tarball (no auth -- code is public).
// Excludes node_modules, dist, .git, .local, secrets, and log files.

router.get("/source.tar.gz", (_req: Request, res: Response) => {
  const root = findProjectRoot();
  res.setHeader("Content-Type", "application/gzip");
  res.setHeader("Content-Disposition", "attachment; filename=source.tar.gz");

  const tar = spawn("tar", [
    "-czf", "-",
    "--exclude=node_modules",
    "--exclude=.git",
    "--exclude=dist",
    "--exclude=.local",
    "--exclude=.cache",
    "--exclude=.env",
    "--exclude=*.log",
    "--exclude=*.secret",
    "-C", root, ".",
  ]);

  tar.stdout.pipe(res);
  tar.stderr.on("data", () => { /* suppress tar warnings */ });
  tar.on("error", () => { if (!res.headersSent) res.status(500).end("tar failed"); });
  res.on("close", () => tar.kill());
});

export default router;
