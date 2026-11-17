import { useCallback, useEffect, useMemo, useState } from "react";

import { ApiError, NetworkError, createApiClient, type ApiClient } from "./api/client";
import { clearToken, loadToken, saveToken } from "./auth/token";
import { ApplicationBoard } from "./components/ApplicationBoard";
import { ApplicationDetail } from "./components/ApplicationDetail";
import { ApprovalsInbox } from "./components/ApprovalsInbox";
import { TokenForm } from "./components/TokenForm";
import { WorkflowProgress } from "./components/WorkflowProgress";
import { href, useRoute, type Route } from "./router";

interface Props {
  apiBaseUrl: string;
  /** Injected in tests. */
  fetch?: (request: Request) => Promise<Response>;
}

type Session =
  | { kind: "signed-out"; error: string | null }
  | { kind: "checking"; token: string }
  | { kind: "ready"; token: string; actorId: string };

export function App({ apiBaseUrl, fetch }: Props) {
  const route = useRoute();
  const [session, setSession] = useState<Session>(() => {
    const stored = loadToken();
    return stored ? { kind: "checking", token: stored } : { kind: "signed-out", error: null };
  });

  const token = session.kind === "signed-out" ? null : session.token;
  const client: ApiClient | null = useMemo(
    () => (token ? createApiClient({ baseUrl: apiBaseUrl, token, fetch }) : null),
    [apiBaseUrl, token, fetch],
  );

  // Check a token with `GET /v1/auth/whoami` before showing anything with it.
  useEffect(() => {
    if (session.kind !== "checking" || !client) return;
    let cancelled = false;
    client
      .whoami()
      .then((me) => {
        if (cancelled) return;
        saveToken(session.token);
        setSession({ kind: "ready", token: session.token, actorId: me.actor_id });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        clearToken();
        const message =
          error instanceof ApiError && error.isUnauthenticated
            ? "That token was not accepted. Copy it again from `python -m personalos.cli auth token`."
            : error instanceof NetworkError
              ? `Could not reach the API at ${apiBaseUrl}.`
              : error instanceof Error
                ? error.message
                : String(error);
        setSession({ kind: "signed-out", error: message });
      });
    return () => {
      cancelled = true;
    };
  }, [session, client, apiBaseUrl]);

  const signOut = useCallback((error: string | null = null) => {
    clearToken();
    setSession({ kind: "signed-out", error });
  }, []);

  const onUnauthenticated = useCallback(
    () => signOut("The API rejected the token (it may have been rotated). Enter the current one."),
    [signOut],
  );

  return (
    <div className="app">
      <header className="app-header">
        <h1>PersonalOS</h1>
        {session.kind === "ready" && (
          <div className="whoami">
            <span className="muted">signed in as</span> <code>{session.actorId}</code>
            <button type="button" className="link" onClick={() => signOut()}>
              Forget token
            </button>
          </div>
        )}
      </header>
      {session.kind === "ready" && (
        <nav className="app-nav" aria-label="Sections">
          <a href={href.approvals()} aria-current={route.name === "approvals" ? "page" : undefined}>
            Approvals
          </a>
          <a
            href={href.board()}
            aria-current={route.name === "board" || route.name === "application" ? "page" : undefined}
          >
            Applications
          </a>
        </nav>
      )}
      <main>
        {session.kind === "ready" && client ? (
          <Screen route={route} client={client} onUnauthenticated={onUnauthenticated} />
        ) : (
          <TokenForm
            apiBaseUrl={apiBaseUrl}
            error={session.kind === "signed-out" ? session.error : null}
            checking={session.kind === "checking"}
            onSubmit={(value) => setSession({ kind: "checking", token: value })}
          />
        )}
      </main>
    </div>
  );
}

function Screen({
  route,
  client,
  onUnauthenticated,
}: {
  route: Route;
  client: ApiClient;
  onUnauthenticated: () => void;
}) {
  switch (route.name) {
    case "board":
      return <ApplicationBoard client={client} onUnauthenticated={onUnauthenticated} />;
    case "application":
      return (
        <ApplicationDetail
          key={route.id}
          client={client}
          applicationId={route.id}
          onUnauthenticated={onUnauthenticated}
        />
      );
    case "workflow":
      return (
        <section>
          <a href={href.board()} className="small">
            ← Board
          </a>
          <WorkflowProgress
            key={route.id}
            client={client}
            workflowId={route.id}
            onUnauthenticated={onUnauthenticated}
          />
        </section>
      );
    case "approvals":
      return <ApprovalsInbox client={client} onUnauthenticated={onUnauthenticated} />;
  }
}
