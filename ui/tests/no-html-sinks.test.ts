/**
 * The UI renders untrusted text (model output, recruiter mail) only as React
 * text nodes. This guards that structurally: no source file may reach for an
 * API that parses a string as HTML.
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const SINKS = [
  "dangerouslySetInnerHTML",
  "innerHTML",
  "outerHTML",
  "insertAdjacentHTML",
  "document.write",
  "createContextualFragment",
  "DOMParser",
  "srcdoc",
  "eval(",
  "new Function",
];

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return sources(path);
    return /\.(ts|tsx)$/.test(name) && !name.endsWith(".d.ts") ? [path] : [];
  });
}

describe("source", () => {
  const files = sources(join(__dirname, "..", "src"));

  it("is found", () => {
    expect(files.length).toBeGreaterThan(5);
  });

  it.each(SINKS)("never uses %s", (sink) => {
    const offenders = files.filter((file) => readFileSync(file, "utf8").includes(sink));
    expect(offenders).toEqual([]);
  });
});
