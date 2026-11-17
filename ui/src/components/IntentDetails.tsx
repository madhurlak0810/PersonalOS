/**
 * The exact action a reviewer is approving, shown before they decide.
 *
 * Everything here is model- or email-derived and so untrusted: a recruiter's
 * message body, a drafted reply, an event title. It is rendered only as React
 * text nodes -- never parsed as HTML, never turned into a link --
 * so markup in it shows up as the characters it is made of and does nothing.
 */
import type { ActionPreview, AttachmentPreview, PendingApproval } from "../api/client";
import { CALENDAR_ACTION_TYPES, EMAIL_ACTION_TYPES, formatDateTime } from "../format";

export function IntentDetails({ approval }: { approval: PendingApproval }) {
  const { action_type: type, preview, payload } = approval;
  if (EMAIL_ACTION_TYPES.has(type)) {
    return (
      <>
        {preview ? <EmailIntent preview={preview} /> : <MissingPreview />}
        <PayloadView payload={payload} collapsed />
      </>
    );
  }
  if (CALENDAR_ACTION_TYPES.has(type)) {
    return (
      <>
        <CalendarIntent actionType={type} payload={payload} />
        <PayloadView payload={payload} collapsed />
      </>
    );
  }
  return (
    <>
      {preview && <DocumentIntent preview={preview} />}
      <PayloadView payload={payload} collapsed={false} />
    </>
  );
}

// --- Email ----------------------------------------------------------------------

function EmailIntent({ preview }: { preview: ActionPreview }) {
  return (
    <section className="intent" aria-label="Email to be sent">
      <dl className="fields">
        <dt>Recipient</dt>
        <dd data-testid="email-recipient">{preview.recipient}</dd>
        <dt>Subject</dt>
        <dd data-testid="email-subject">{preview.subject ?? <em>(no subject)</em>}</dd>
        <dt>Thread</dt>
        <dd data-testid="email-thread">
          {preview.in_reply_to ? (
            <>
              Reply in existing thread <code>{preview.in_reply_to}</code>
            </>
          ) : (
            "New thread"
          )}
        </dd>
      </dl>
      <BodyDiff diff={preview.body_diff ?? ""} />
      <Attachments attachments={preview.attachments ?? []} expectsList />
    </section>
  );
}

// --- Documents and applications -------------------------------------------------

function DocumentIntent({ preview }: { preview: ActionPreview }) {
  return (
    <section className="intent" aria-label="What this action sends">
      <dl className="fields">
        <dt>Recipient</dt>
        <dd>{preview.recipient}</dd>
        {preview.subject && (
          <>
            <dt>Subject</dt>
            <dd>{preview.subject}</dd>
          </>
        )}
      </dl>
      {preview.body_diff ? <BodyDiff diff={preview.body_diff} /> : null}
      <Attachments attachments={preview.attachments ?? []} expectsList={false} />
    </section>
  );
}

// --- Calendar -------------------------------------------------------------------

/** `ActionIntent.payload` keys of the calendar kinds (`personalos/domain/interview_scheduling.py`). */
const CALENDAR_FIELDS: [key: string, label: string, kind: "text" | "time"][] = [
  ["title", "Title", "text"],
  ["starts_at", "Starts", "time"],
  ["ends_at", "Ends", "time"],
  ["time_zone", "Time zone", "text"],
  ["role", "Role", "text"],
  ["interview_starts_at", "Interview starts", "time"],
  ["event_id", "Event id", "text"],
];

function CalendarIntent({
  actionType,
  payload,
}: {
  actionType: string;
  payload: PendingApproval["payload"];
}) {
  if (!payload) {
    return <MissingPayload what="the event's details" />;
  }
  const rows = CALENDAR_FIELDS.filter(([key]) => payload[key] !== undefined && payload[key] !== null);
  return (
    <section className="intent" aria-label="Calendar event">
      {actionType === "cancel_calendar_event" && (
        <p className="callout">This deletes the event below from your calendar.</p>
      )}
      <dl className="fields" data-testid="calendar-event">
        {rows.map(([key, label, kind]) => {
          const raw = stringify(payload[key]);
          return (
            <div className="field-row" key={key}>
              <dt>{label}</dt>
              <dd>
                {kind === "time" ? (
                  <>
                    {formatDateTime(raw)} <code className="muted">{raw}</code>
                  </>
                ) : (
                  raw
                )}
              </dd>
            </div>
          );
        })}
      </dl>
    </section>
  );
}

// --- Shared pieces --------------------------------------------------------------

function lineClass(line: string): string {
  if (line.startsWith("+++") || line.startsWith("---")) return "diff-file";
  if (line.startsWith("@@")) return "diff-hunk";
  if (line.startsWith("+")) return "diff-add";
  if (line.startsWith("-")) return "diff-del";
  return "diff-ctx";
}

/** A unified diff (`personalos.domain.job_search.body_diff`), one text line per row. */
export function BodyDiff({ diff }: { diff: string }) {
  if (!diff) {
    return <p className="muted">Body: unchanged (no differences).</p>;
  }
  return (
    <figure className="diff" aria-label="Body diff">
      <figcaption>Body diff (− current, + proposed)</figcaption>
      <pre data-testid="body-diff">
        {diff.split("\n").map((line, index) => (
          <span className={`diff-line ${lineClass(line)}`} key={index}>
            {line}
            {"\n"}
          </span>
        ))}
      </pre>
    </figure>
  );
}

function Attachments({
  attachments,
  expectsList,
}: {
  attachments: AttachmentPreview[];
  expectsList: boolean;
}) {
  if (attachments.length === 0) {
    return expectsList ? <p data-testid="attachments">Attachments: none.</p> : null;
  }
  return (
    <div data-testid="attachments">
      <h4>Attachments ({attachments.length})</h4>
      <ul className="attachments">
        {attachments.map((item, index) => (
          <li key={`${item.name}-${index}`}>
            <span className="attachment-name">{item.name}</span>
            {item.artifact_type && <span className="muted"> · {item.artifact_type}</span>}
            {item.sha256 && (
              <code className="muted" title={item.sha256}>
                {" "}
                sha256 {item.sha256.slice(0, 12)}…
              </code>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** The action's typed payload, verbatim, as the hash covers it. */
function PayloadView({
  payload,
  collapsed,
}: {
  payload: PendingApproval["payload"];
  collapsed: boolean;
}) {
  if (!payload) {
    return collapsed ? null : <MissingPayload what="the action's payload" />;
  }
  const body = <pre data-testid="action-payload">{JSON.stringify(payload, null, 2)}</pre>;
  if (collapsed) {
    return (
      <details className="payload">
        <summary>Full action payload</summary>
        {body}
      </details>
    );
  }
  return (
    <section className="payload" aria-label="Action payload">
      <h4>Action payload</h4>
      {body}
    </section>
  );
}

function MissingPayload({ what }: { what: string }) {
  return (
    <p className="callout warning" role="note">
      The API did not return {what}, or it could not be verified against this approval's
      hash. Review it with <code>python -m personalos.cli approvals</code> before approving.
    </p>
  );
}

function MissingPreview() {
  return (
    <p className="callout warning" role="note">
      No preview of this email was recorded with the approval.
    </p>
  );
}

function stringify(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}
