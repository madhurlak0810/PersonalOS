import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ACTION_STATUSES } from "../src/api/client";
import { StatusBadge } from "../src/components/StatusBadge";

describe("StatusBadge", () => {
  it.each(ACTION_STATUSES)("labels %s with the enum value itself", (status) => {
    render(<StatusBadge status={status} />);
    expect(screen.getByTestId("action-status")).toHaveTextContent(status);
  });

  it.each(ACTION_STATUSES.filter((status) => status !== "EXECUTED"))(
    "never describes %s as completed",
    (status) => {
      render(<StatusBadge status={status} />);
      const text = screen.getByTestId("action-status").textContent ?? "";
      expect(text).not.toMatch(/(?<!was )\bdone\b|\bcompleted?\b|\bexecuted\b/i);
      expect(text).toMatch(/\bnot\b|\bnothing\b/i);
    },
  );

  it("styles only EXECUTED as done", () => {
    render(<StatusBadge status="RECOMMENDED" />);
    expect(screen.getByTestId("action-status")).not.toHaveClass("status-executed");
    expect(screen.getByTestId("action-status")).toHaveClass("status-recommended");
  });
});
