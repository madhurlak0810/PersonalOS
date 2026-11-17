/**
 * A Server-Sent Events parser, per the WHATWG event-stream format.
 *
 * Fed text as it arrives, in chunks that may split a line or a field anywhere;
 * returns the events completed so far. `id:` and `retry:` persist across
 * events as the spec says, so `lastEventId` is always the cursor a reconnect
 * should resume from. Comment lines (`: keepalive`) are ignored.
 */

export interface SseMessage {
  /** The `id:` in effect when this event was dispatched; `""` if none was ever set. */
  id: string;
  /** The `event:` field; `"message"` when the frame had none. */
  event: string;
  data: string;
}

export class SseParser {
  private buffer = "";
  private data: string[] = [];
  private eventType = "";
  /** The last `id:` seen, carried across events. */
  lastEventId = "";
  /** The reconnection delay the server asked for (`retry:`), in ms. */
  retryMs: number | null = null;

  /** Feed a chunk; get back the events it completed. */
  push(chunk: string): SseMessage[] {
    this.buffer += chunk;
    const messages: SseMessage[] = [];
    // A trailing "\r" might be the first half of "\r\n", so leave it in the buffer.
    const pattern = /\r\n|\n|\r(?!$)/;
    for (;;) {
      const match = pattern.exec(this.buffer);
      if (!match) break;
      const line = this.buffer.slice(0, match.index);
      this.buffer = this.buffer.slice(match.index + match[0].length);
      const message = this.line(line);
      if (message) messages.push(message);
    }
    return messages;
  }

  private line(line: string): SseMessage | null {
    if (line === "") return this.dispatch();
    if (line.startsWith(":")) return null;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    switch (field) {
      case "data":
        this.data.push(value);
        break;
      case "event":
        this.eventType = value;
        break;
      case "id":
        if (!value.includes("\0")) this.lastEventId = value;
        break;
      case "retry":
        if (/^\d+$/.test(value)) this.retryMs = Number(value);
        break;
      default:
        break;
    }
    return null;
  }

  private dispatch(): SseMessage | null {
    const data = this.data;
    const event = this.eventType || "message";
    this.data = [];
    this.eventType = "";
    if (data.length === 0) return null;
    return { id: this.lastEventId, event, data: data.join("\n") };
  }
}
