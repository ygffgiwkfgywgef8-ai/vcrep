import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { logger } from "./logger";

// -- Shared update state (read by middleware to inject response headers) --------

export const updateState = {
  currentVersion: "unknown",
  latestVersion: null as string | null,
  updateAvailable: false,
};

// -- Helpers -------------------------------------------------------------------

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

function readLocalVersion(): string {
  try {
    const raw = readFileSync(resolve(findProjectRoot(), "version.json"), "utf-8");
    const parsed = JSON.parse(raw) as { version?: string };
    return parsed.version ?? "unknown";
  } catch {
    return "unknown";
  }
}

function parseVersionSegment(seg: string): number {
  const n = parseInt(seg, 10);
  return isNaN(n) ? 0 : n;
}

// Versioning scheme: MAJOR.MINOR.PATCH[.BUILD] -- all segments are plain integers.
// Hotfixes increment BUILD (e.g. 1.1.2 -> 1.1.2.1). Minor releases increment PATCH.
// Returns >0 if a is newer, <0 if b is newer, 0 if equal.
function compareVersions(a: string, b: string): number {
  if (a === b) return 0;
  const pa = a.split(".").map(parseVersionSegment);
  const pb = b.split(".").map(parseVersionSegment);
  const len = Math.max(pa.length, pb.length);
  for (let i = 0; i < len; i++) {
    const na = pa[i] ?? 0, nb = pb[i] ?? 0;
    if (na > nb) return 1; if (na < nb) return -1;
  }
  return 0;
}

// -- Check function -------------------------------------------------------------

function resolveVersionUrl(base: string): string {
  const clean = base.replace(/\/$/, "");
  const ghMatch = clean.match(/^https?:\/\/github\.com\/([^/]+\/[^/]+)/);
  if (ghMatch) {
    const slug = ghMatch[1].replace(/\.git$/, "");
    return `https://raw.githubusercontent.com/${slug}/main/version.json`;
  }
  return `${clean}/api/version`;
}

async function checkForUpdate(): Promise<void> {
  const sourceUrl = process.env["UPDATE_SOURCE_URL"] ?? "https://github.com/Akatsuki03/AI-Monorepo";
  const local = readLocalVersion();
  updateState.currentVersion = local;

  try {
    const versionUrl = resolveVersionUrl(sourceUrl);
    const r = await fetch(versionUrl, { signal: AbortSignal.timeout(8000) });
    if (!r.ok) return;
    const body = await r.json() as { current?: { version?: string }; version?: string };
    const remoteVersion = body.current?.version ?? (body as { version?: string }).version ?? null;
    if (!remoteVersion) return;

    updateState.latestVersion = remoteVersion;
    updateState.updateAvailable = local !== "unknown" && compareVersions(remoteVersion, local) > 0;

    if (updateState.updateAvailable) {
      logger.warn(
        { currentVersion: local, latestVersion: remoteVersion, portalUrl: sourceUrl },
        `[UP]  AI Monorepo update available: v${local} -> v${remoteVersion}. ` +
        `Open the portal and copy the Agent update prompt to apply the update.`
      );
    } else {
      logger.info({ version: local }, "AI Monorepo is up to date");
    }
  } catch {
    // Remote check failed -- not a hard error, silently skip
  }
}

// -- Startup + periodic checker ------------------------------------------------

const CHECK_INTERVAL_MS = 60 * 60 * 1000; // 1 hour

export function startUpdateChecker(): void {
  // Immediate check after a short delay so the server finishes starting first
  setTimeout(() => {
    void checkForUpdate();
  }, 3000);

  // Periodic checks every hour
  setInterval(() => {
    void checkForUpdate();
  }, CHECK_INTERVAL_MS);
}
