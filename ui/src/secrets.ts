/**
 * The UI's last line against showing a credential.
 *
 * The API redacts every audit, failure and integration payload on its way out
 * (`personalos.domain.redaction`), so a credential in one of those responses is
 * an API bug. When one turns up anyway, the UI does not render it: the value is
 * replaced with `WITHHELD` before any component sees it, and the caller is told
 * where, so the screen can say that something was withheld (and the bug can be
 * reported) without ever saying what it was.
 *
 * Two checks, mirroring the API's redactor:
 *
 * - **Sensitive keys.** A value under `access_token`, `Authorization`,
 *   `client_secret` and the like is withheld unless it is the API's own
 *   `[REDACTED]` marker (or empty).
 * - **Token shapes.** A string that contains a bearer header, a PEM private
 *   key, a `key=value` credential, or a well-known token format is withheld
 *   whole. A string the API already redacted (`Bearer [REDACTED]`) passes.
 *
 * Pure: returns a copy and never mutates its input.
 */

/** What the API puts in place of a credential it removed. Safe to show. */
export const REDACTED = "[REDACTED]";

/** What the UI puts in place of a credential the API failed to remove. */
export const WITHHELD = "[withheld]";

/** `personalos.domain.redaction.SENSITIVE_KEYS`, compared the same way. */
const SENSITIVE_KEYS = new Set([
  "authorization",
  "proxy_authorization",
  "cookie",
  "set_cookie",
  "refresh_token",
  "access_token",
  "id_token",
  "auth_token",
  "session_token",
  "bearer_token",
  "token",
  "client_secret",
  "api_key",
  "apikey",
  "x_api_key",
  "password",
  "passwd",
  "secret",
  "secret_key",
  "private_key",
  "oauth_code",
  "authorization_code",
  "code_verifier",
]);

/**
 * Every value the API already redacted, with the auth scheme in front of it
 * (`Bearer [REDACTED]`), becomes this before the patterns run, and no pattern's
 * value class matches it. So a redacted credential never reads as a live one.
 */
const MARK = "\u0000";
const ALREADY_REDACTED = /(?:\b(?:bearer|basic|digest|token|oauth)\s+)?\[REDACTED\]/gi;

const SECRET_PATTERNS: readonly RegExp[] = [
  /-----BEGIN [A-Z ]*PRIVATE KEY-----/,
  /\b(?:proxy-)?authorization["']?\s*[:=]\s*["']?(?:(?:bearer|basic|digest|token|oauth)\s+)?[^\s"',;}\]\u0000]{4,}/i,
  /\b(?:set-)?cookie["']?\s*[:=]\s*["']?[^\s"'}\u0000]{4,}/i,
  /\bbearer\s+[A-Za-z0-9._~+/=-]{8,}/i,
  /(?<![A-Za-z0-9])(?:(?:refresh|access|id|auth|session|bearer)[_-]?token|client[_-]?secret|(?:x[_-])?api[_-]?key|password|passwd|secret[_-]?key|private[_-]?key|code[_-]?verifier)["']?\s*[:=]\s*["']?[^\s"'&,;}\]\u0000]{4,}/i,
  /\bya29\.[0-9A-Za-z._-]{10,}/,
  /\b1\/\/[0-9A-Za-z_-]{20,}/,
  /\bAIza[0-9A-Za-z_-]{35,}/,
  /\beyJ[A-Za-z0-9_-]{5,}\.eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]+/,
  /\bpos_at_[A-Za-z0-9_-]{8,}/,
  /\bpos_st_[A-Za-z0-9_.-]{8,}/,
  /\bsk-[A-Za-z0-9_-]{20,}/,
  /\bgh[pousr]_[A-Za-z0-9]{36,}/,
  /\bxox[abprs]-[A-Za-z0-9-]{10,}/,
  /\bAKIA[0-9A-Z]{16}\b/,
];

function normalizeKey(key: string): string {
  return key.trim().toLowerCase().replace(/-/g, "_");
}

/** True when `text` holds something shaped like a credential the API did not redact. */
export function looksLikeSecret(text: string): boolean {
  const scrubbed = text.replace(ALREADY_REDACTED, MARK);
  return SECRET_PATTERNS.some((pattern) => pattern.test(scrubbed));
}

export interface Guarded<T> {
  value: T;
  /** Where values were withheld, as `a.b[0].c` paths. Empty when nothing was. */
  withheld: string[];
}

/** `value` with every unredacted credential replaced by `WITHHELD`. */
export function withholdSecrets<T>(value: T): Guarded<T> {
  const withheld: string[] = [];

  const walk = (node: unknown, path: string, sensitive: boolean): unknown => {
    if (typeof node === "string") {
      if (sensitive ? node !== "" && node !== REDACTED : looksLikeSecret(node)) {
        withheld.push(path || "(root)");
        return WITHHELD;
      }
      return node;
    }
    if (sensitive && node !== null && node !== undefined && typeof node !== "boolean") {
      // A number or structure under a credential key: none of it is shown.
      withheld.push(path || "(root)");
      return WITHHELD;
    }
    if (Array.isArray(node)) return node.map((item, i) => walk(item, `${path}[${i}]`, false));
    if (node !== null && typeof node === "object") {
      const out: Record<string, unknown> = {};
      for (const [key, child] of Object.entries(node)) {
        out[key] = walk(child, path ? `${path}.${key}` : key, SENSITIVE_KEYS.has(normalizeKey(key)));
      }
      return out;
    }
    return node;
  };

  const guarded = walk(value, "", false) as T;
  if (withheld.length > 0) {
    // An API bug: say where, never what.
    console.error(
      `PersonalOS UI: the API returned ${withheld.length} unredacted credential-like value(s); not shown`,
      withheld,
    );
  }
  return { value: guarded, withheld };
}
