import { type Request, type Response, type NextFunction } from "express";

export function authMiddleware(req: Request, res: Response, next: NextFunction): void {
  const proxyApiKey = process.env["PROXY_API_KEY"];
  if (!proxyApiKey) {
    res.status(500).json({ error: { message: "PROXY_API_KEY not configured", type: "server_error" } });
    return;
  }

  const authHeader = req.headers["authorization"];
  const googleKey = req.headers["x-goog-api-key"];
  const queryKey = req.query["key"];

  let providedKey: string | undefined;

  if (authHeader && authHeader.startsWith("Bearer ")) {
    providedKey = authHeader.slice(7);
  } else if (typeof googleKey === "string" && googleKey) {
    providedKey = googleKey;
  } else if (typeof queryKey === "string" && queryKey) {
    providedKey = queryKey;
  }

  if (!providedKey || providedKey !== proxyApiKey) {
    res.status(401).json({ error: { message: "Invalid or missing API key", type: "invalid_request_error", code: "invalid_api_key" } });
    return;
  }

  next();
}
