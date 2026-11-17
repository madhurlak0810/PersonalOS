import { useState, type FormEvent } from "react";

import { API_TOKEN_PREFIX, normalizeToken } from "../auth/token";

interface Props {
  apiBaseUrl: string;
  error: string | null;
  checking: boolean;
  onSubmit: (token: string) => void;
}

/** Asks for the local API token. The token is checked against the API before use. */
export function TokenForm({ apiBaseUrl, error, checking, onSubmit }: Props) {
  const [value, setValue] = useState("");

  function submit(event: FormEvent) {
    event.preventDefault();
    const token = normalizeToken(value);
    if (token) onSubmit(token);
  }

  const looksWrong = value.trim() !== "" && !normalizeToken(value).startsWith(API_TOKEN_PREFIX);

  return (
    <form className="token-form" onSubmit={submit}>
      <h2>Connect to the API</h2>
      <p>
        Paste the local API token for <code>{apiBaseUrl}</code>. Print it with:
      </p>
      <pre>python -m personalos.cli auth token</pre>
      <label htmlFor="api-token">API token</label>
      <input
        id="api-token"
        type="password"
        autoComplete="off"
        spellCheck={false}
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder={`${API_TOKEN_PREFIX}…`}
      />
      {looksWrong && (
        <p className="muted">Local tokens start with <code>{API_TOKEN_PREFIX}</code>.</p>
      )}
      {error && (
        <p className="callout error" role="alert">
          {error}
        </p>
      )}
      <button type="submit" disabled={checking || normalizeToken(value) === ""}>
        {checking ? "Checking…" : "Connect"}
      </button>
      <p className="muted">
        The token is kept in this tab's session storage only, and is sent only to the API above.
      </p>
    </form>
  );
}
