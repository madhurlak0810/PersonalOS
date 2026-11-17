import { describe, expect, it } from "vitest";

import { SseParser } from "../src/api/sse";
import { parseRoute } from "../src/router";

describe("SseParser", () => {
  it("parses frames split anywhere, keeping the last id", () => {
    const parser = new SseParser();
    const text = "retry: 3000\n\nid: a\nevent: step_started\ndata: {\"x\":1}\n\nid: b\nevent: failure\ndata: line1\ndata: line2\n\n";
    const messages = [];
    for (const ch of text) messages.push(...parser.push(ch));
    expect(messages).toEqual([
      { id: "a", event: "step_started", data: '{"x":1}' },
      { id: "b", event: "failure", data: "line1\nline2" },
    ]);
    expect(parser.lastEventId).toBe("b");
    expect(parser.retryMs).toBe(3000);
  });

  it("ignores comments, handles CRLF, and keeps an id across events", () => {
    const parser = new SseParser();
    const messages = parser.push(": keepalive\r\n\r\nid: z\r\ndata: one\r\n\r\ndata: two\r\n\r\n");
    expect(messages).toEqual([
      { id: "z", event: "message", data: "one" },
      { id: "z", event: "message", data: "two" },
    ]);
  });

  it("does not dispatch an unterminated frame", () => {
    const parser = new SseParser();
    expect(parser.push("id: a\ndata: partial\n")).toEqual([]);
    expect(parser.push("\n")).toEqual([{ id: "a", event: "message", data: "partial" }]);
  });
});

describe("parseRoute", () => {
  const id = "0a000000-0000-4000-8000-000000000001";
  it.each([
    ["", { name: "approvals" }],
    ["#/approvals", { name: "approvals" }],
    ["#/board", { name: "board" }],
    [`#/applications/${id}`, { name: "application", id }],
    [`#/workflows/${id}`, { name: "workflow", id }],
    ["#/applications/not-a-uuid", { name: "approvals" }],
    ["#/workflows/../../x", { name: "approvals" }],
  ])("%s", (hash, route) => {
    expect(parseRoute(hash)).toEqual(route);
  });
});
