import app from "./app";
import { logger } from "./lib/logger";
import { startUpdateChecker } from "./lib/updateChecker";

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error(
    "PORT environment variable is required but was not provided.",
  );
}

const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

// Startup env checks -- warn but don't crash so the setup wizard can guide users
const missing: string[] = [];
if (!process.env["PROXY_API_KEY"]) missing.push("PROXY_API_KEY");
if (!process.env["AI_INTEGRATIONS_ANTHROPIC_API_KEY"]) missing.push("AI_INTEGRATIONS_ANTHROPIC_API_KEY");
if (!process.env["AI_INTEGRATIONS_ANTHROPIC_BASE_URL"]) missing.push("AI_INTEGRATIONS_ANTHROPIC_BASE_URL");
if (!process.env["AI_INTEGRATIONS_OPENAI_API_KEY"]) missing.push("AI_INTEGRATIONS_OPENAI_API_KEY");
if (!process.env["AI_INTEGRATIONS_OPENAI_BASE_URL"]) missing.push("AI_INTEGRATIONS_OPENAI_BASE_URL");
if (!process.env["AI_INTEGRATIONS_GEMINI_API_KEY"]) missing.push("AI_INTEGRATIONS_GEMINI_API_KEY");
if (!process.env["AI_INTEGRATIONS_GEMINI_BASE_URL"]) missing.push("AI_INTEGRATIONS_GEMINI_BASE_URL");
if (!process.env["AI_INTEGRATIONS_OPENROUTER_API_KEY"]) missing.push("AI_INTEGRATIONS_OPENROUTER_API_KEY");
if (!process.env["AI_INTEGRATIONS_OPENROUTER_BASE_URL"]) missing.push("AI_INTEGRATIONS_OPENROUTER_BASE_URL");
if (missing.length > 0) {
  logger.warn(
    { missing },
    "Missing environment variables -- visit the portal setup wizard or ask the Replit AI assistant to configure them"
  );
}

const server = app.listen(port, (err) => {
  if (err) {
    logger.error({ err }, "Error listening on port");
    process.exit(1);
  }

  logger.info({ port }, "Server listening");

  // Start background version check (logs warning if update is available)
  startUpdateChecker();
});

// Disable all server-level timeouts so long streaming responses (10k+ tokens,
// 7-10 min) are never cut short by the HTTP layer.
// SSE keepalive frames every 5 s keep the TCP connection alive through proxies.
server.headersTimeout  = 0;   // no timeout waiting for request headers
server.requestTimeout  = 0;   // no timeout for the full request/response cycle
server.timeout         = 0;   // no socket inactivity timeout
server.keepAliveTimeout = 65_000; // keep TCP alive slightly longer than a 60 s proxy
