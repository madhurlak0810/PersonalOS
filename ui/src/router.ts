/**
 * Hash routing: the page's state lives after `#`, so the static build needs no
 * server-side routes and nothing about the view is ever sent to a server.
 *
 *   #/approvals             the approvals inbox (the default)
 *   #/board                 the application board
 *   #/applications/{id}     one application
 *   #/workflows/{id}        one workflow: live progress, failures, audit trail
 *   #/audit                 the audit trail, searched
 *   #/integrations          integration health
 */
import { useEffect, useState } from "react";

export type Route =
  | { name: "approvals" }
  | { name: "board" }
  | { name: "application"; id: string }
  | { name: "workflow"; id: string }
  | { name: "audit" }
  | { name: "integrations" };

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function parseRoute(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const [head, id] = parts;
  if (head === "board" && parts.length === 1) return { name: "board" };
  if (head === "audit" && parts.length === 1) return { name: "audit" };
  if (head === "integrations" && parts.length === 1) return { name: "integrations" };
  if (head === "applications" && id && UUID.test(id)) return { name: "application", id };
  if (head === "workflows" && id && UUID.test(id)) return { name: "workflow", id };
  return { name: "approvals" };
}

export const href = {
  approvals: () => "#/approvals",
  board: () => "#/board",
  application: (id: string) => `#/applications/${encodeURIComponent(id)}`,
  workflow: (id: string) => `#/workflows/${encodeURIComponent(id)}`,
  audit: () => "#/audit",
  integrations: () => "#/integrations",
};

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseRoute(window.location.hash));
  useEffect(() => {
    const update = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);
  return route;
}
