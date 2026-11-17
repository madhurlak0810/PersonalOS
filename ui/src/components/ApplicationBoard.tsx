/**
 * The application board: `GET /v1/applications` as a kanban, one column per
 * lifecycle state, with filters and pages.
 *
 * Cards move by drag and drop or with each card's "Move to" menu, both through
 * `POST /v1/applications/{id}/transition`. An applied move updates the board
 * from the card the API returned, at once; a refused one leaves the card where
 * it was and shows why; a queued one (an effect awaiting approval) leaves it
 * where it was and links to the approvals inbox.
 */
import { useCallback, useEffect, useState, type DragEvent, type ReactNode } from "react";

import {
  APPLICATION_STATUSES,
  ApiError,
  NetworkError,
  effectFor,
  type ApiClient,
  type ApplicationCard,
  type ApplicationStatus,
  type BoardQuery,
  type BoardResponse,
  type TransitionRequest,
} from "../api/client";
import { applicationStatusLabel, dateInputToApi, formatDateTime } from "../format";
import { href } from "../router";
import { EffectPrompt, MoveFeedbackView, Unauthenticated, sendMove, type MoveFeedback } from "./Move";

interface Props {
  client: ApiClient;
  onUnauthenticated: () => void;
  pageSize?: number;
}

type Load =
  | { kind: "loading"; previous: BoardResponse | null }
  | { kind: "loaded"; board: BoardResponse }
  | { kind: "error"; message: string };

interface Filters {
  states: ApplicationStatus[];
  company: string;
  from: string;
  to: string;
}

const NO_FILTERS: Filters = { states: [], company: "", from: "", to: "" };
const DRAG_TYPE = "application/x-personalos-application";

function toQuery(filters: Filters, offset: number, limit: number): BoardQuery {
  return {
    states: filters.states,
    company: filters.company.trim() || undefined,
    createdFrom: dateInputToApi(filters.from, "start"),
    createdTo: dateInputToApi(filters.to, "end"),
    offset,
    limit,
  };
}

/** The board with `card` taken out of whichever column held it and put at the top of its own. */
export function placeCard(board: BoardResponse, card: ApplicationCard): BoardResponse {
  let removed = false;
  let added = false;
  const columns = board.columns.map((column) => {
    const applications = column.applications ?? [];
    const without = applications.filter((item) => item.application_id !== card.application_id);
    let total = column.total;
    if (without.length !== applications.length) {
      removed = true;
      total -= 1;
    }
    if (column.status === card.status) {
      added = true;
      return { ...column, total: total + 1, applications: [card, ...without] };
    }
    return { ...column, total, applications: without };
  });
  const total = board.total + (added ? 1 : 0) - (removed ? 1 : 0);
  return { ...board, columns, total };
}

export function ApplicationBoard({ client, onUnauthenticated, pageSize = 50 }: Props) {
  const [draft, setDraft] = useState<Filters>(NO_FILTERS);
  const [filters, setFilters] = useState<Filters>(NO_FILTERS);
  const [offset, setOffset] = useState(0);
  const [load, setLoad] = useState<Load>({ kind: "loading", previous: null });
  const [prompt, setPrompt] = useState<{ card: ApplicationCard; to: ApplicationStatus } | null>(
    null,
  );
  const [busy, setBusy] = useState<ReadonlySet<string>>(new Set());
  const [feedback, setFeedback] = useState<ReadonlyMap<string, MoveFeedback>>(new Map());
  const [dropTarget, setDropTarget] = useState<ApplicationStatus | null>(null);

  const fail = useCallback(
    (error: unknown) => {
      if (error instanceof ApiError && error.isUnauthenticated) {
        onUnauthenticated();
        return;
      }
      setLoad({
        kind: "error",
        message:
          error instanceof NetworkError
            ? "Could not reach the PersonalOS API. Is `python -m personalos.cli api` running?"
            : error instanceof Error
              ? error.message
              : String(error),
      });
    },
    [onUnauthenticated],
  );

  const refresh = useCallback(async () => {
    setLoad((prev) => ({
      kind: "loading",
      previous: prev.kind === "loaded" ? prev.board : prev.kind === "loading" ? prev.previous : null,
    }));
    try {
      const board = await client.listApplications(toQuery(filters, offset, pageSize));
      setLoad({ kind: "loaded", board });
    } catch (error) {
      fail(error);
    }
  }, [client, filters, offset, pageSize, fail]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const board = load.kind === "loaded" ? load.board : load.kind === "loading" ? load.previous : null;

  const setCardFeedback = (id: string, value: MoveFeedback | null) =>
    setFeedback((prev) => {
      const next = new Map(prev);
      if (value) next.set(id, value);
      else next.delete(id);
      return next;
    });

  const send = async (card: ApplicationCard, request: TransitionRequest) => {
    const id = card.application_id;
    setBusy((prev) => new Set(prev).add(id));
    setCardFeedback(id, null);
    try {
      const { response, feedback: result } = await sendMove(client, id, request);
      if (response && response.outcome === "applied") {
        setLoad((prev) =>
          prev.kind === "loaded" ? { kind: "loaded", board: placeCard(prev.board, response.application) } : prev,
        );
      }
      setCardFeedback(id, result);
      setPrompt((open) => (open?.card.application_id === id ? null : open));
    } catch (error) {
      if (error instanceof Unauthenticated) onUnauthenticated();
      else setCardFeedback(id, { kind: "error", message: String(error) });
    } finally {
      setBusy((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  };

  const requestMove = (card: ApplicationCard, to: ApplicationStatus) => {
    if (to === card.status || busy.has(card.application_id)) return;
    setCardFeedback(card.application_id, null);
    if (effectFor(to)) {
      setPrompt({ card, to });
      return;
    }
    setPrompt((open) => (open?.card.application_id === card.application_id ? null : open));
    void send(card, { to });
  };

  const findCard = (id: string): ApplicationCard | undefined =>
    board?.columns.flatMap((column) => column.applications ?? []).find((c) => c.application_id === id);

  const onDrop = (status: ApplicationStatus) => (event: DragEvent) => {
    event.preventDefault();
    setDropTarget(null);
    const id = event.dataTransfer.getData(DRAG_TYPE) || event.dataTransfer.getData("text/plain");
    const card = id ? findCard(id) : undefined;
    if (card) requestMove(card, status);
  };

  const toggleState = (state: ApplicationStatus) =>
    setDraft((prev) => ({
      ...prev,
      states: prev.states.includes(state)
        ? prev.states.filter((s) => s !== state)
        : APPLICATION_STATUSES.filter((s) => s === state || prev.states.includes(s)),
    }));

  const first = board ? board.offset + 1 : 0;
  const last = board ? board.offset + board.columns.reduce((n, c) => n + (c.applications?.length ?? 0), 0) : 0;

  return (
    <section className="board" aria-labelledby="board-title">
      <div className="inbox-header">
        <h2 id="board-title">Applications</h2>
        <button type="button" onClick={() => void refresh()} disabled={load.kind === "loading"}>
          {load.kind === "loading" ? "Loading…" : "Refresh"}
        </button>
      </div>

      <form
        className="board-filters"
        aria-label="Filter applications"
        onSubmit={(event) => {
          event.preventDefault();
          setOffset(0);
          setFilters(draft);
        }}
      >
        <fieldset className="state-filter">
          <legend>States</legend>
          {APPLICATION_STATUSES.map((state) => (
            <label key={state}>
              <input
                type="checkbox"
                checked={draft.states.includes(state)}
                onChange={() => toggleState(state)}
              />{" "}
              {applicationStatusLabel(state)}
            </label>
          ))}
        </fieldset>
        <label className="field">
          Company
          <input
            value={draft.company}
            maxLength={255}
            onChange={(event) => setDraft({ ...draft, company: event.target.value })}
          />
        </label>
        <label className="field">
          Created from
          <input
            type="date"
            value={draft.from}
            onChange={(event) => setDraft({ ...draft, from: event.target.value })}
          />
        </label>
        <label className="field">
          Created to
          <input
            type="date"
            value={draft.to}
            onChange={(event) => setDraft({ ...draft, to: event.target.value })}
          />
        </label>
        <div className="actions">
          <button type="submit">Apply filters</button>
          <button
            type="button"
            onClick={() => {
              setDraft(NO_FILTERS);
              setFilters(NO_FILTERS);
              setOffset(0);
            }}
          >
            Clear
          </button>
        </div>
      </form>

      {load.kind === "error" && (
        <p className="callout error" role="alert">
          {load.message}
        </p>
      )}

      {board && (
        <>
          <div className="pager">
            <span className="muted" data-testid="board-range">
              {board.total === 0 ? "No applications match." : `Showing ${first}–${last} of ${board.total}`}
            </span>
            <div className="actions">
              <button
                type="button"
                disabled={board.offset === 0 || load.kind === "loading"}
                onClick={() => setOffset(Math.max(0, board.offset - board.limit))}
              >
                Previous page
              </button>
              <button
                type="button"
                disabled={board.next_offset == null || load.kind === "loading"}
                onClick={() => board.next_offset != null && setOffset(board.next_offset)}
              >
                Next page
              </button>
            </div>
          </div>

          <div className="columns">
            {board.columns.map((column) => (
              <section
                key={column.status}
                className={`column${dropTarget === column.status ? " drop-target" : ""}`}
                data-testid="board-column"
                data-status={column.status}
                aria-label={`${applicationStatusLabel(column.status)} (${column.total})`}
                onDragOver={(event) => {
                  event.preventDefault();
                  event.dataTransfer.dropEffect = "move";
                  setDropTarget(column.status);
                }}
                onDragLeave={() => setDropTarget((current) => (current === column.status ? null : current))}
                onDrop={onDrop(column.status)}
              >
                <h3 className="column-title">
                  {applicationStatusLabel(column.status)}{" "}
                  <span className="muted" data-testid="column-total">
                    {column.total}
                  </span>
                </h3>
                {(column.applications ?? []).map((card) => (
                  <BoardCard
                    key={card.application_id}
                    card={card}
                    busy={busy.has(card.application_id)}
                    feedback={feedback.get(card.application_id) ?? null}
                    onMove={(to) => requestMove(card, to)}
                    onDismiss={() => setCardFeedback(card.application_id, null)}
                  >
                    {prompt?.card.application_id === card.application_id && (
                      <EffectPrompt
                        label={`${card.company} — ${card.title}`}
                        from={card.status}
                        to={prompt.to}
                        busy={busy.has(card.application_id)}
                        onCancel={() => setPrompt(null)}
                        onConfirm={(request) => void send(card, request)}
                      />
                    )}
                  </BoardCard>
                ))}
              </section>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

interface CardProps {
  card: ApplicationCard;
  busy: boolean;
  feedback: MoveFeedback | null;
  onMove: (to: ApplicationStatus) => void;
  onDismiss: () => void;
  children?: ReactNode;
}

function BoardCard({ card, busy, feedback, onMove, onDismiss, children }: CardProps) {
  const label = `${card.company} — ${card.title}`;
  return (
    <article
      className={`board-card${busy ? " busy" : ""}`}
      data-testid="board-card"
      data-application-id={card.application_id}
      draggable={!busy}
      aria-busy={busy}
      onDragStart={(event) => {
        event.dataTransfer.setData(DRAG_TYPE, card.application_id);
        event.dataTransfer.setData("text/plain", card.application_id);
        event.dataTransfer.effectAllowed = "move";
      }}
    >
      <a href={href.application(card.application_id)} className="card-title">
        <strong>{card.company}</strong>
        <span>{card.title}</span>
      </a>
      {card.location && <span className="muted small">{card.location}</span>}
      <span className="muted small">
        {card.last_activity_at
          ? `Active ${formatDateTime(card.last_activity_at)}`
          : `Added ${formatDateTime(card.created_at)}`}
      </span>
      <label className="move-select small">
        <span className="visually-hidden">Move {label} to</span>
        <select
          aria-label={`Move ${label} to`}
          value=""
          disabled={busy}
          onChange={(event) => {
            const to = event.target.value as ApplicationStatus;
            if (to) onMove(to);
          }}
        >
          <option value="">Move to…</option>
          {APPLICATION_STATUSES.filter((state) => state !== card.status).map((state) => (
            <option key={state} value={state}>
              {applicationStatusLabel(state)}
            </option>
          ))}
        </select>
      </label>
      {children}
      {feedback && <MoveFeedbackView feedback={feedback} onPick={onMove} onDismiss={onDismiss} />}
    </article>
  );
}
