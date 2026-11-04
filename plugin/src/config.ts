// SPDX-License-Identifier: AGPL-3.0-or-later
import { createConfigSchematics } from "@lmstudio/sdk";

export const configSchematics = createConfigSchematics()
  .field("action", "select", { displayName: "Action", options: [
    { value: "chat", displayName: "Chat" },
    { value: "scan", displayName: "Find models and check architecture support" },
    { value: "test-model", displayName: "Auto-test selected model" },
    { value: "test-all", displayName: "Auto-test all local models" },
    { value: "report", displayName: "Show selected model's report" },
    { value: "jobs", displayName: "Show test progress" },
    { value: "cancel", displayName: "Cancel active tests" }
  ] }, "chat")
  .field("model", "string", { displayName: "Model", subtitle: "Model name or ID from Find models. Leave empty to list models." }, "")
  .field("profile", "select", { displayName: "Loop profile", options: [
    { value: "recommended", displayName: "Recommended by held-out tests (baseline unless demonstrated better)" },
    { value: "baseline", displayName: "Original model" },
    { value: "loop2", displayName: "2 passes, anchored damping (experimental)" },
    { value: "loop3", displayName: "3 passes, anchored damping (experimental)" },
    { value: "loop4", displayName: "4 passes, anchored damping (experimental)" }
  ] }, "recommended")
  .field("autoTest", "boolean", { displayName: "Test before first looped chat", subtitle: "Runs baseline, numerical identity probes and separate selection/holdout tasks. Can take several minutes." }, true)
  .field("quick", "boolean", { displayName: "Quick compatibility screen", subtitle: "Fewer tasks; never recommends a loop profile from this screen." }, false)
  .field("maxTokens", "numeric", { displayName: "Maximum generated tokens", min: 1, max: 2048, int: true }, 512)
  .build();

export const globalConfigSchematics = createConfigSchematics()
  .field("baseUrl", "string", { displayName: "LoopLab local address" }, "http://127.0.0.1:7341")
  .field("accessToken", "string", { displayName: "Local access token (optional)", isProtected: true,
    subtitle: "Leave blank to read this computer's LoopLab token from its standard state directory." }, "")
  .build();
