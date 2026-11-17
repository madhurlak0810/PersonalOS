/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

/** Where the API listens by default: `python -m personalos.cli api`. */
const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

/**
 * The same headers the API sends (`apps/api/security.py`, `SECURITY_HEADERS`),
 * with a CSP that admits this UI's own bundle and the API it calls. Nothing
 * inline, nothing from a CDN, nothing that can frame it.
 */
function securityHeaders(apiBaseUrl: string, { dev }: { dev: boolean }): Record<string, string> {
  const apiOrigin = new URL(apiBaseUrl).origin;
  const headers: Record<string, string> = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
  };
  // The dev server injects an inline React Refresh preamble and talks to
  // itself over a websocket, so the strict policy applies to the built UI
  // (`npm run preview`, or any server sending these headers) only.
  if (!dev) {
    headers["Content-Security-Policy"] = [
      "default-src 'none'",
      "script-src 'self'",
      "style-src 'self'",
      "img-src 'self'",
      `connect-src ${apiOrigin}`,
      "frame-ancestors 'none'",
      "base-uri 'none'",
      "form-action 'none'",
    ].join("; ");
  }
  return headers;
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  const apiBaseUrl = env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL;
  return {
    plugins: [react()],
    // `localhost:5173` is the one origin the API's CORS allow-list admits by
    // default (`API_CORS_ORIGINS`), so the port is fixed rather than bumped.
    server: {
      host: "localhost",
      port: 5173,
      strictPort: true,
      headers: securityHeaders(apiBaseUrl, { dev: true }),
    },
    preview: {
      host: "localhost",
      port: 5173,
      strictPort: true,
      headers: securityHeaders(apiBaseUrl, { dev: false }),
    },
    build: {
      sourcemap: true,
    },
    test: {
      environment: "jsdom",
      globals: true,
      setupFiles: ["./tests/setup.ts"],
      restoreMocks: true,
    },
  };
});
