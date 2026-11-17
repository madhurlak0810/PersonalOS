import { afterEach, describe, expect, it, vi } from "vitest";

import { WITHHELD, looksLikeSecret, withholdSecrets } from "../src/secrets";

afterEach(() => vi.restoreAllMocks());

describe("withholdSecrets", () => {
  it.each([
    "Authorization: Bearer abcdefghijklmnop",
    "bearer eyJhbGciOiJIUzI1NiJ9abc",
    "refresh_token=1//0gabcdefghijklmnopqrstuv",
    '"client_secret": "GOCSPX-abcdef"',
    "ya29.a0AfH6SMBabcdefghij",
    "pos_at_abcdefghijklmnopqrstuvwxyz",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIE...",
    "AKIAABCDEFGHIJKLMNOP",
  ])("recognises %s", (text) => {
    expect(looksLikeSecret(text)).toBe(true);
  });

  it.each([
    "Authorization: Bearer [REDACTED]",
    "Authorization: [REDACTED]",
    "refresh_token=[REDACTED]",
    "Greenhouse timed out after 30s",
    "policy denied this action; asking again gets the same answer",
    "idempotency_key send-1",
    "cred://personalos/google/me@example.com",
  ])("leaves %s alone", (text) => {
    expect(looksLikeSecret(text)).toBe(false);
  });

  it("withholds values under credential keys unless already redacted, and reports where", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const input = {
      receipt: { ok: true, access_token: "live", Refresh_Token: "[REDACTED]", "X-Api-Key": 123 },
      list: [{ password: { nested: "x" } }, "fine"],
    };
    const { value, withheld } = withholdSecrets(input);
    expect(value).toEqual({
      receipt: { ok: true, access_token: WITHHELD, Refresh_Token: "[REDACTED]", "X-Api-Key": WITHHELD },
      list: [{ password: WITHHELD }, "fine"],
    });
    expect(withheld).toEqual(["receipt.access_token", "receipt.X-Api-Key", "list[0].password"]);
    // Never mutates the response it was given.
    expect(input.receipt.access_token).toBe("live");
  });

  it("returns an equal value and says nothing when there is nothing to withhold", () => {
    const error = vi.spyOn(console, "error");
    const input = { a: [1, "two", null], b: { token_count: 3 } };
    expect(withholdSecrets(input)).toEqual({ value: input, withheld: [] });
    expect(error).not.toHaveBeenCalled();
  });
});
