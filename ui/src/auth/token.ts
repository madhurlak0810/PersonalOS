/**
 * Where the UI keeps the local API token (Phase K, #119).
 *
 * The token is the one `python -m personalos.cli auth token` prints. It is held
 * in `sessionStorage`, so it is scoped to this origin and this tab and is gone
 * when the tab closes; it is never put in a URL, a cookie, or `localStorage`.
 * Storage can be unavailable (a private window, blocked site data), so every
 * access is guarded and the UI falls back to asking again.
 */

const STORAGE_KEY = "personalos.api-token";

/** Every local token starts with this (`personalos.domain.auth.API_TOKEN_PREFIX`). */
export const API_TOKEN_PREFIX = "pos_at_";

function storage(): Storage | null {
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

export function loadToken(): string | null {
  try {
    return storage()?.getItem(STORAGE_KEY) || null;
  } catch {
    return null;
  }
}

export function saveToken(token: string): void {
  try {
    storage()?.setItem(STORAGE_KEY, token);
  } catch {
    // Not persisted; the token still works for this page load.
  }
}

export function clearToken(): void {
  try {
    storage()?.removeItem(STORAGE_KEY);
  } catch {
    // Nothing stored to clear.
  }
}

/** The pasted value, trimmed, with an accidental `Bearer ` prefix removed. */
export function normalizeToken(raw: string): string {
  return raw.trim().replace(/^Bearer\s+/i, "").trim();
}
