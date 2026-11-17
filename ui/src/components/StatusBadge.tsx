import type { ActionStatus } from "../api/client";
import { STATUS_DESCRIPTIONS } from "../format";

interface Props {
  status: ActionStatus;
  /** Overrides the standard description, e.g. "Approved; not carried out yet". */
  detail?: string;
}

/**
 * The action-status label every item carries. The enum value is shown verbatim,
 * so what the UI says is exactly what the API said; only `EXECUTED` is styled
 * as done.
 */
export function StatusBadge({ status, detail }: Props) {
  return (
    <span className={`status status-${status.toLowerCase()}`} data-testid="action-status">
      <span className="status-enum">{status}</span>
      <span className="status-detail">{detail ?? STATUS_DESCRIPTIONS[status]}</span>
    </span>
  );
}
