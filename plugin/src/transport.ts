// SPDX-License-Identifier: AGPL-3.0-or-later
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

export function localAddress(value: string): string {
  const url = new URL(value);
  if (url.protocol !== "http:" || !["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)
      || url.username || url.password || url.search || url.hash || !["", "/"].includes(url.pathname)) {
    throw new Error("LoopLab requires a loopback HTTP address, for example http://127.0.0.1:7341");
  }
  return url.origin;
}

export async function localToken(configured: string): Promise<string> {
  if (configured.trim()) return configured.trim();
  const root = process.env.XDG_STATE_HOME || join(homedir(), ".local", "state");
  try {
    return (await readFile(join(root, "looplab", "token"), "utf8")).trim();
  } catch {
    throw new Error("Start the LoopLab companion with 'python -m looplab serve', or enter its local access token in plugin settings.");
  }
}

export async function call(base: string, token: string, path: string, signal: AbortSignal, body?: unknown): Promise<Response> {
  if (!path.startsWith("/") || path.startsWith("//")) throw new Error("Invalid API path");
  const response = await fetch(localAddress(base) + path, {
    method: body === undefined ? "GET" : "POST", redirect: "error", signal,
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    ...(body === undefined ? {} : { body: JSON.stringify(body) })
  });
  if (!response.ok) {
    const text = await response.text();
    let message = text;
    try { message = JSON.parse(text).error || text; } catch { /* plain HTTP error */ }
    throw new Error(`LoopLab ${response.status}: ${message.slice(0, 1500)}`);
  }
  return response;
}

export async function* events(body: ReadableStream<Uint8Array>): AsyncGenerator<any> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data = block.split("\n").filter(line => line.startsWith("data:")).map(line => line.slice(5).trimStart()).join("\n");
        if (!data) continue;
        if (data === "[DONE]") return;
        const event = JSON.parse(data);
        if (event.error) throw new Error(typeof event.error === "string" ? event.error : JSON.stringify(event.error));
        yield event;
      }
      if (done) {
        if (buffer.trim()) throw new Error("Truncated event stream");
        return;
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
