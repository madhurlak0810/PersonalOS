/**
 * Moving an application: the choice an effectful move needs, the request, and
 * what the API said back.
 *
 * - A move with no possible external effect is sent as it is.
 * - A move that *could* have one (to `applied`; to `withdrawn`, `declined` or
 *   `accepted`) is never sent until the user picks, explicitly, between having
 *   PersonalOS do it -- which goes to the approvals inbox and changes nothing
 *   until approved -- and only recording something that already happened.
 * - A refused move (422) shows the API's message and the states it said are
 *   allowed, and the card stays where it was.
 */
import { useState } from "react";

import {
  ApiError,
  NetworkError,
  effectFor,
  type ApiClient,
  type ApplicationStatus,
  type TransitionEffect,
  type TransitionRequest,
  type TransitionResponse,
} from "../api/client";
import { applicationStatusLabel } from "../format";
import { href } from "../router";

export type MoveFeedback =
  | { kind: "applied"; to: ApplicationStatus }
  | { kind: "queued"; to: ApplicationStatus; effect: TransitionEffect | null; workflowId: string | null }
  | { kind: "invalid"; message: string; allowed: ApplicationStatus[] }
  | { kind: "error"; message: string };

export const EFFECT_LABELS: Record<TransitionEffect, string> = {
  submit_application: "Submit the application",
  notify_recruiter: "Tell the recruiter",
};

/** Thrown out of `sendMove` so the caller can sign out. */
export class Unauthenticated extends Error {}

/** Send a move and turn whatever comes back into something to show. */
export async function sendMove(
  client: ApiClient,
  applicationId: string,
  request: TransitionRequest,
): Promise<{ response: TransitionResponse | null; feedback: MoveFeedback }> {
  try {
    const response = await client.transition(applicationId, request);
    const feedback: MoveFeedback =
      response.outcome === "queued"
        ? {
            kind: "queued",
            to: response.to_status,
            effect: response.effect ?? null,
            workflowId: response.workflow_id ?? null,
          }
        : { kind: "applied", to: response.to_status };
    return { response, feedback };
  } catch (error) {
    if (error instanceof ApiError && error.isUnauthenticated) throw new Unauthenticated();
    if (error instanceof ApiError && error.allowedNextStates) {
      return {
        response: null,
        feedback: { kind: "invalid", message: error.message, allowed: error.allowedNextStates },
      };
    }
    const message =
      error instanceof NetworkError
        ? "Could not reach the PersonalOS API; the application was not moved."
        : error instanceof ApiError && error.isConflict
          ? `${error.message} The application was not moved.`
          : error instanceof Error
            ? error.message
            : String(error);
    return { response: null, feedback: { kind: "error", message } };
  }
}

interface PromptProps {
  label: string;
  from: ApplicationStatus;
  to: ApplicationStatus;
  busy?: boolean;
  onCancel: () => void;
  onConfirm: (request: TransitionRequest) => void;
}

/**
 * The choice a move to an effect-bearing state needs. Nothing is pre-selected:
 * the move is not sent until one option is picked.
 */
export function EffectPrompt({ label, from, to, busy, onCancel, onConfirm }: PromptProps) {
  const effect = effectFor(to);
  const [choice, setChoice] = useState<"effect" | "record" | null>(null);
  const [reason, setReason] = useState("");
  const [recipient, setRecipient] = useState("");
  const [message, setMessage] = useState("");
  if (!effect) return null;

  const needsRecipient = effect === "notify_recruiter";
  const complete =
    choice === "record" ||
    (choice === "effect" && (!needsRecipient || (recipient.trim() && message.trim())));

  const confirm = () => {
    if (!complete) return;
    const body: TransitionRequest = { to, reason: reason.trim() || null };
    if (choice === "effect") {
      body.effect = effect;
      if (needsRecipient) body.recipient = recipient.trim();
      if (message.trim()) body.message = message.trim();
    }
    onConfirm(body);
  };

  const effectText =
    effect === "submit_application"
      ? "Submit it through PersonalOS. It goes to the approvals inbox; nothing is submitted until you approve it there."
      : "Tell the recruiter through PersonalOS. The message goes to the approvals inbox; nothing is sent until you approve it there.";
  const recordText =
    effect === "submit_application"
      ? "I already applied elsewhere — only record it. Nothing is sent."
      : "Only record it. Nothing is sent.";

  return (
    <form
      className="effect-prompt"
      data-testid="effect-prompt"
      aria-label={`Move ${label} to ${applicationStatusLabel(to)}`}
      onSubmit={(event) => {
        event.preventDefault();
        confirm();
      }}
    >
      <p>
        Move <strong>{label}</strong> from {applicationStatusLabel(from)} to{" "}
        <strong>{applicationStatusLabel(to)}</strong>. This move can have an effect outside
        PersonalOS ({EFFECT_LABELS[effect].toLowerCase()}). Choose one:
      </p>
      <fieldset>
        <legend className="visually-hidden">What should happen</legend>
        <label>
          <input
            type="radio"
            name="effect-choice"
            checked={choice === "effect"}
            onChange={() => setChoice("effect")}
          />{" "}
          {effectText}
        </label>
        <label>
          <input
            type="radio"
            name="effect-choice"
            checked={choice === "record"}
            onChange={() => setChoice("record")}
          />{" "}
          {recordText}
        </label>
      </fieldset>
      {choice === "effect" && needsRecipient && (
        <label className="field">
          Recruiter's email
          <input
            type="email"
            value={recipient}
            onChange={(event) => setRecipient(event.target.value)}
            required
          />
        </label>
      )}
      {choice === "effect" && (
        <label className="field">
          {needsRecipient ? "Message" : "Cover note (optional; the latest cover letter otherwise)"}
          <textarea
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            rows={3}
            maxLength={10_000}
            required={needsRecipient}
          />
        </label>
      )}
      <label className="field">
        Reason (optional)
        <input value={reason} onChange={(event) => setReason(event.target.value)} maxLength={2000} />
      </label>
      <div className="actions">
        <button type="submit" className="approve" disabled={!complete || busy}>
          {choice === "effect" ? "Queue for approval" : "Move"}
        </button>
        <button type="button" onClick={onCancel} disabled={busy}>
          Cancel
        </button>
      </div>
    </form>
  );
}

interface FeedbackProps {
  feedback: MoveFeedback;
  /** Offered for each allowed state after a refused move. */
  onPick?: (to: ApplicationStatus) => void;
  onDismiss?: () => void;
}

export function MoveFeedbackView({ feedback, onPick, onDismiss }: FeedbackProps) {
  const dismiss = onDismiss && (
    <button type="button" className="link" onClick={onDismiss} aria-label="Dismiss">
      ×
    </button>
  );
  switch (feedback.kind) {
    case "applied":
      return (
        <p className="callout success" role="status" data-testid="move-feedback">
          Moved to {applicationStatusLabel(feedback.to)}. {dismiss}
        </p>
      );
    case "queued":
      return (
        <div className="callout warning" role="status" data-testid="move-feedback">
          <p>
            Queued for approval
            {feedback.effect ? `: ${EFFECT_LABELS[feedback.effect].toLowerCase()}` : ""}. It stays
            where it is until you approve it and it has run; then it moves to{" "}
            {applicationStatusLabel(feedback.to)}.
          </p>
          <a href={href.approvals()}>Open the approvals inbox</a>
          {feedback.workflowId && (
            <>
              {" · "}
              <a href={href.workflow(feedback.workflowId)}>Follow its progress</a>
            </>
          )}{" "}
          {dismiss}
        </div>
      );
    case "invalid":
      return (
        <div className="callout error" role="alert" data-testid="move-feedback">
          <p>{feedback.message}</p>
          {feedback.allowed.length === 0 ? (
            <p>No moves are possible from here.</p>
          ) : (
            <p data-testid="allowed-states">
              Allowed next states:{" "}
              {feedback.allowed.map((state, index) => (
                <span key={state}>
                  {index > 0 && ", "}
                  {onPick ? (
                    <button type="button" className="link" onClick={() => onPick(state)}>
                      {applicationStatusLabel(state)}
                    </button>
                  ) : (
                    applicationStatusLabel(state)
                  )}
                </span>
              ))}
            </p>
          )}
          {dismiss}
        </div>
      );
    case "error":
      return (
        <p className="callout error" role="alert" data-testid="move-feedback">
          {feedback.message} {dismiss}
        </p>
      );
  }
}
