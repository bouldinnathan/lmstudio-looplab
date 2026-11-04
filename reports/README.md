# Validation records

The latest [all-local compatibility survey](all-local-survey.md) covers the expanded runtime and its generated experimental adapters. Its [JSON record](all-local-survey.json) includes loader errors, per-profile identity and repeatability checks, resource outcomes, and repair handoffs. It is a compatibility survey, not a quality evaluation. Profiles can pass anchor identity yet fail repeat-request stability; those exploration profiles remain unavailable for chat.

`native-auto-cpu.json` and `native-auto-vulkan.json` cover the expanded runtime with 24 independent-reference checks each. The CPU run explicitly disables device, operation and KV-cache offloading and automatic device fitting. Only four architectures have independent mathematical references; generated architecture candidates retain the experimental tier.

The expanded generator also passed a live 2-pass chat with Gemma 4 12B through the real companion and Vulkan worker; see [`integration-auto.json`](integration-auto.json). The controller was a test harness, so this does not independently verify LM Studio's graphical plugin host. The final application checks passed 20 Python tests and 9 Node tests; [GitHub CI](https://github.com/bouldinnathan/lmstudio-looplab/actions/runs/41776316816) is green.

`native-cpu.json` and `native-vulkan.json` contain 24 checks each against an independent NumPy implementation. They compare full next-token distributions through prefill and four autoregressive steps, covering four architectures, damping, anchoring, and both cache policies. The maximum allowed log-probability error is 0.005; the small tolerance accommodates F16 KV storage and different floating-point execution orders.

These checks establish numerical behavior on synthetic tiny F32 networks. Real quantized checkpoints additionally need their own anchor-identity check. A compatibility pass does not establish improved answer quality.

The named model files are allowlisted exports from the automatic diagnostic suite. They contain architecture and checkpoint-header identity, runtime settings, aggregate scores, truncation counts, timing, cache probes and the recommendation. They contain no model weights, local file paths, access tokens, or complete model outputs.

The 12 selection and 12 held-out tasks are small, public diagnostics. Their scores should not be compared to the paper's benchmarks. Only one candidate reaches the held-out split; quick runs never recommend a loop profile. Prefix caching stays disabled even when one cache probe happens to agree.

The LM Studio plugin was installed successfully on the development machine. Generator-controller integration tests cover discovery, first-use evaluation, streaming, tool calls and cancellation. Live model discovery through the companion also passed. Live streamed chat also passed through the real companion and native Vulkan worker in both baseline and 2-pass modes; see `integration.json`. Its generator controller was supplied by a test harness. An external SDK client was denied `plugins.use` permission by LM Studio, so graphical-host invocation remains unverified.

## Initial local model results

These earlier quality results use the original runtime fingerprint. The broader survey above reports the newer runtime separately; it does not replace missing quality evidence with smoke-test success.

Full diagnostic run on 2026-11-04 UTC. All five models passed the 2-, 3- and 4-pass anchor-identity probes with a maximum observed log-probability difference of zero. The selected candidate in each case used two passes.

| Model | Original held-out score | Selected passes | Looped held-out score | Recommended |
|---|---:|---:|---:|---|
| [deepcoder-14b](deepcoder-14b.json) | 8/12 | 2 | 8/12 | Original |
| [deepseek-r1-32b](deepseek-r1-32b.json) | 10/12 | 2 | 10/12 | Original |
| [devstral-24b](devstral-24b.json) | 8/12 | 2 | 9/12 | Original |
| [qwen3-coder-30b](qwen3-coder-30b.json) | 10/12 | 2 | 10/12 | Original |
| [reader-lm-1.5b](reader-lm-1.5b.json) | 0/12 | 2 | 0/12 | Original |

Scores are out of 12 held-out diagnostics. Devstral's one additional correct answer did not satisfy the paired evidence threshold. No candidate was promoted automatically. DeepSeek-R1 32B here is a dense Qwen2-architecture checkpoint; this does not establish support for DeepSeek/Kimi MLA models.
