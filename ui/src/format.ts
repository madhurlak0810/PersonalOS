/** Display helpers. Pure, so the components stay about layout. */
import type {
  ActionStatus,
  ApplicationStatus,
  FailureClass,
  FailureStatus,
  PendingApproval,
} from "./api/client";

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
  apply_file_plan: "Apply file plan",
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

const APPLICATION_STATUS_LABELS: Record<ApplicationStatus, string> = {
  discovered: "Discovered",
  saved: "Saved",
  preparing: "Preparing",
  ready_to_apply: "Ready to apply",
  applied: "Applied",
  response: "Response",
  interviewing: "Interviewing",
  offer: "Offer",
  accepted: "Accepted",
  declined: "Declined",
  rejected: "Rejected",
  withdrawn: "Withdrawn",
  skipped: "Skipped",
  follow_up_pending: "Follow-up pending",
  stalled: "Stalled",
};

/** "Ready to apply" for `ready_to_apply`; the raw value for anything unknown. */
export function applicationStatusLabel(status: string): string {
  return APPLICATION_STATUS_LABELS[status as ApplicationStatus] ?? status;
}

/** A 0..1 score as a whole percentage. */
export function formatScore(score: number): string {
  return `${Math.round(score * 100)}%`;
}

/** A `<input type="date">` value as the API's naive-UTC timestamp at that day's start or end. */
export function dateInputToApi(value: string, edge: "start" | "end"): string | undefined {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return undefined;
  return edge === "start" ? `${value}T00:00:00` : `${value}T23:59:59.999999`;
}

const FAILURE_CLASS_LABELS: Record<FailureClass, string> = {
  model_transient: "Model unavailable (transient)",
  model_semantic: "Model output did not meet its contract",
  provider_transient: "Provider unavailable (transient)",
  provider_permanent: "Provider refused permanently",
  concurrency: "Conflict with another writer",
  partial_side_effect: "Failed after a side effect",
  policy_denial: "Denied by policy",
  human_timeout: "No answer in time",
  unclassified: "Unclassified",
};

/** "Provider refused permanently" for `provider_permanent`; the raw value for anything unknown. */
export function failureClassLabel(value: string): string {
  return FAILURE_CLASS_LABELS[value as FailureClass] ?? value;
}

/**
 * A failure's status as shown. A policy denial is `denied`, never "failed":
 * it is an answer, and nothing about it can be retried.
 */
export const FAILURE_STATUS_TEXT: Record<FailureStatus, { label: string; detail: string }> = {
  failed: { label: "FAILED", detail: "The step did not complete" },
  denied: { label: "DENIED", detail: "Policy refused this action — nothing was done" },
  retry_queued: { label: "RETRY QUEUED", detail: "Waiting for the worker to re-run the step" },
  dismissed: { label: "DISMISSED", detail: "Closed without a retry" },
};

export const AUDIT_STEP_LABELS: Record<string, string> = {
  intent: "Intent",
  policy_decision: "Policy decision",
  approval: "Approval",
  execution: "Execution",
  receipt: "Receipt",
  event: "Event",
};

/** A policy verdict as the policy engine names it: `REQUIRE_APPROVAL`. */
export function policyDecisionLabel(value: string): string {
  return value.toUpperCase();
}
