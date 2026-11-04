// SPDX-License-Identifier: AGPL-3.0-or-later
import { Chat, GeneratorController } from "@lmstudio/sdk";
import { configSchematics, globalConfigSchematics } from "./config";
import { call, events, localToken } from "./transport";
import { setTimeout as delay } from "node:timers/promises";

export function messages(history: Chat): any[] {
  if (history.hasFiles()) throw new Error("LoopLab currently supports text and tool messages. Remove attachments for this model.");
  return history.getMessagesArray().flatMap((message): any[] => {
    const role = message.getRole();
    if (role === "tool") return message.getToolCallResults().map(result => ({
      role: "tool", tool_call_id: result.toolCallId, content: result.content
    }));
    const calls = message.getToolCallRequests();
    return [{ role, content: message.getText() || (calls.length ? null : ""),
      ...(calls.length ? { tool_calls: calls.map((c, index) => ({
        id: c.id || `call_${index}`, type: "function", function: { name: c.name, arguments: JSON.stringify(c.arguments || {}) }
      })) } : {}) }];
  });
}

function summary(report: any): string {
  if (!report || report.status === "untested") return "This model has not been tested yet.";
  if (report.suite?.startsWith("looplab-explore-")) {
    const rows = (report.profiles || []).map((p: any) =>
      `| ${p.passes} | ${p.anchor_identity?.passed ? "Passed" : "Failed"} | ${p.loop_smoke?.passed ? "Passed" : p.status} |`);
    return `**${report.model?.name}: ${report.status}**\n\n` +
      `Adapter: ${report.model?.tier}. Original-model check: ${report.baseline_check?.passed ? "passed" : "not passed or not run"}.\n\n` +
      (rows.length ? `| Passes | Anchor identity | Loop smoke test |\n|---|---|---|\n${rows.join("\n")}\n\n` : "") +
      (report.reasons?.length ? report.reasons.join("; ") + "\n\n" : "") +
      (report.diagnostic ? `Diagnostic:\n\n\`\`\`text\n${report.diagnostic}\n\`\`\`\n\n` : "") +
      (report.repair ? "A repair handoff has been saved for a later coding pass.\n\n" : "") +
      "Successful profiles can be selected manually. These are experimental compatibility checks; quality improvement has not been tested.";
  }
  const baseline = report.baseline;
  const rows = (report.profiles || []).map((p: any) =>
    `| ${p.passes} | ${p.anchor_identity?.passed ? "Passed" : "Failed"} | ${p.selection_passed}/${p.selection_total} | ${p.holdout_total ? `${p.holdout_passed}/${p.holdout_total}` : "Not selected"} |`);
  return `**${report.model?.name || "Model"}: ${report.status}**\n\n` +
    (baseline ? `Original: selection ${baseline.selection.passed}/${baseline.selection.total}, holdout ${baseline.holdout.passed}/${baseline.holdout.total}.\n\n` : "") +
    (rows.length ? `| Passes | Anchor identity | Selection tasks | Held-out tasks |\n|---|---|---|---|\n${rows.join("\n")}\n\n` : "") +
    `Recommended: **${report.recommended_profile || "baseline"}**. This is a local diagnostic, not a reproduction of the paper's benchmark.`;
}

export async function generate(ctl: GeneratorController, history: Chat): Promise<void> {
  const config = ctl.getPluginConfig(configSchematics);
  const global = ctl.getGlobalPluginConfig(globalConfigSchematics);
  const base = global.get("baseUrl");
  const token = await localToken(global.get("accessToken"));
  const signal = ctl.abortSignal;
  const json = async (path: string, body?: unknown) => (await call(base, token, path, signal, body)).json();
  const model = config.get("model").trim();
  const action = config.get("action");
  const inventory = async () => {
    const data = await json("/looplab/refresh", {});
    ctl.fragmentGenerated("Select a model by entering its name or ID in plugin settings.\n\n" +
      data.models.map((m: any) => `- **${m.name}**: ${m.eligible ? `${m.tier || "validated"} adapter, eligible for testing` : m.reasons.join("; ")}\n  ID: \`${m.id}\``).join("\n"));
  };
  if (action === "scan" || (!model && !["test-all", "explore-all", "jobs", "cancel"].includes(action))) {
    await inventory(); return;
  }
  if (action === "jobs" || action === "cancel") {
    const status = await json("/looplab/status");
    if (action === "cancel") {
      for (const job of status.jobs.filter((j: any) => ["queued", "running"].includes(j.status))) {
        await json(`/looplab/jobs/${job.id}/cancel`, {});
      }
      ctl.fragmentGenerated("Cancellation requested. The current inference request will finish before the worker stops.");
    } else ctl.fragmentGenerated(status.jobs.map((j: any) => `- ${j.id}: ${j.status} — ${j.progress}`).join("\n") || "No test jobs yet.");
    return;
  }
  const getReport = () => json(`/looplab/reports/${encodeURIComponent(model)}`);
  if (action === "report") { ctl.fragmentGenerated(summary(await getReport())); return; }
  let report = model ? await getReport() : null;
  const shouldTest = ["test-model", "test-all", "explore-all", "explore-model"].includes(action) ||
    (action === "chat" && config.get("profile") !== "baseline" && config.get("autoTest") && report?.status === "untested");
  if (shouldTest) {
    const job = await json("/looplab/jobs", { model: ["test-all", "explore-all"].includes(action) ? "all" : model,
      quick: config.get("quick"), explore: action.startsWith("explore-") });
    ctl.fragmentGenerated(`Auto-test started (${job.id}). You can cancel it from the plugin's Cancel active tests action.\n\n`);
    let last = "";
    try {
      while (true) {
        signal.throwIfAborted();
        const current = await json(`/looplab/jobs/${job.id}`);
        if (!["queued", "running"].includes(current.status)) {
          for (const item of current.reports) ctl.fragmentGenerated(summary(item) + "\n\n");
          if (current.status !== "complete") throw new Error(`Auto-test ${current.status}: ${current.progress}`);
          break;
        }
        // Report model/profile transitions; avoid printing every task into the chat.
        if (current.progress !== last && !current.progress.startsWith("Task ")) {
          ctl.fragmentGenerated(current.progress + "\n"); last = current.progress;
        }
        await delay(2000, undefined, { signal });
      }
    } catch (error) {
      if (signal.aborted) {
        await call(base, token, `/looplab/jobs/${job.id}/cancel`, AbortSignal.timeout(3000), {}).catch(() => {});
      }
      throw error;
    }
    if (action !== "chat") return;
    report = await getReport();
  }
  const tools = ctl.getToolDefinitions();
  const response = await call(base, token, "/v1/chat/completions", signal, {
    model, looplab_profile: config.get("profile"), messages: messages(history), stream: true,
    max_tokens: config.get("maxTokens"), ...(tools.length ? { tools } : {})
  });
  if (!response.body) throw new Error("No response stream");
  const activePasses = Number(response.headers.get("X-LoopLab-Passes"));
  if (activePasses >= 1) {
    ctl.fragmentGenerated(activePasses === 1 ? "**LoopLab · original model**\n\n" :
      `**LoopLab · ${activePasses} passes (experimental)**\n\n`);
  }
  const calls = new Map<number, {id?: string; name: string; arguments: string}>();
  for await (const event of events(response.body)) {
    signal.throwIfAborted();
    const delta = event.choices?.[0]?.delta;
    if (!delta) continue;
    if (delta.reasoning_content) ctl.fragmentGenerated(delta.reasoning_content, { reasoningType: "reasoning" });
    if (delta.content) ctl.fragmentGenerated(delta.content);
    for (const change of delta.tool_calls || []) {
      const entry = calls.get(change.index) || { name: "", arguments: "" };
      entry.id ||= change.id;
      entry.name += change.function?.name || "";
      entry.arguments += change.function?.arguments || "";
      calls.set(change.index, entry);
    }
  }
  for (const [, entry] of [...calls].sort((a,b) => a[0]-b[0])) {
    ctl.toolCallGenerationStarted({ toolCallId: entry.id });
    try {
      const args = JSON.parse(entry.arguments || "{}");
      if (!entry.name || !args || typeof args !== "object" || Array.isArray(args)) throw new Error("Invalid tool call");
      ctl.toolCallGenerationNameReceived(entry.name);
      ctl.toolCallGenerationArgumentFragmentGenerated(entry.arguments);
      ctl.toolCallGenerationEnded({ type: "function", id: entry.id, name: entry.name, arguments: args });
    } catch (error) { ctl.toolCallGenerationFailed(error instanceof Error ? error : new Error(String(error))); }
  }
}
