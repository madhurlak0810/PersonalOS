/** Display helpers. Pure, so the components stay about layout. */
import type { ActionStatus, PendingApproval } from "./api/client";

/**
 * Parse an API timestamp. The API writes UTC times without an offset
 * (`datetime.utcnow()`), which `Date` would otherwise read as local time.
 */
export function parseApiDate(value: string): Date {
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(value);
  return new Date(hasZone ? value : `${value}Z`);
}

export function formatDateTime(value: string): string {
  const date = parseApiDate(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

/** "in 3 hours", "5 minutes ago". */
export function formatRelative(value: string, now: Date): string {
  const date = parseApiDate(value);
  const seconds = Math.round((date.getTime() - now.getTime()) / 1000);
  const units: [Intl.RelativeTimeFormatUnit, number][] = [
    ["day", 86_400],
    ["hour", 3_600],
    ["minute", 60],
  ];
  const format = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  for (const [unit, size] of units) {
    if (Math.abs(seconds) >= size) return format.format(Math.round(seconds / size), unit);
  }
  return format.format(seconds, "second");
}

/** Expired by the API's say-so, or because its window has closed since it was listed. */
export function isExpired(approval: PendingApproval, now: Date): boolean {
  return approval.status === "expired" || parseApiDate(approval.expires_at) <= now;
}

const ACTION_TYPE_LABELS: Record<string, string> = {
  submit_application: "Submit application",
  send_recruiter_message: "Send recruiter message",
  send_email: "Send email",
  overwrite_document: "Overwrite document",
  create_calendar_event: "Create calendar event",
  update_calendar_event: "Move calendar event",
  cancel_calendar_event: "Cancel calendar event",
};

export function actionTypeLabel(actionType: string): string {
  return ACTION_TYPE_LABELS[actionType] ?? actionType;
}

export const EMAIL_ACTION_TYPES = new Set(["send_email", "send_recruiter_message"]);
export const CALENDAR_ACTION_TYPES = new Set([
  "create_calendar_event",
  "update_calendar_event",
  "cancel_calendar_event",
]);

/**
 * What each status means, worded as `personalos/cli_views.py` words it.
 * Only `EXECUTED` says anything happened; the rest say plainly that it did not.
 */
export const STATUS_DESCRIPTIONS: Record<ActionStatus, string> = {
  RECOMMENDED: "Suggested — not acted on",
  AWAITING_APPROVAL: "Waiting for your approval — nothing has been sent",
  EXECUTED: "Done — executed, with a receipt",
  DENIED: "Denied — nothing was done",
  FAILED: "Failed — did not happen",
};
