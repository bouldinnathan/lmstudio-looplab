# Validation records

`native-cpu.json` and `native-vulkan.json` contain 24 checks each against an independent NumPy implementation. They compare full next-token distributions through prefill and four autoregressive steps, covering four architectures, damping, anchoring, and both cache policies. The maximum allowed log-probability error is 0.005; the small tolerance accommodates F16 KV storage and different floating-point execution orders.

These checks establish numerical behavior on synthetic tiny F32 networks. Real quantized checkpoints additionally need their own anchor-identity check. A compatibility pass does not establish improved answer quality.

The named model files are allowlisted exports from the automatic diagnostic suite. They contain architecture and checkpoint-header identity, runtime settings, aggregate scores, truncation counts, timing, cache probes and the recommendation. They contain no model weights, local file paths, access tokens, or complete model outputs.

The 12 selection and 12 held-out tasks are small, public diagnostics. Their scores should not be compared to the paper's benchmarks. Only one candidate reaches the held-out split; quick runs never recommend a loop profile. Prefix caching stays disabled even when one cache probe happens to agree.

The LM Studio plugin was installed successfully on the development machine. Generator-controller integration tests cover discovery, first-use evaluation, streaming, tool calls and cancellation. Live model discovery through the companion also passed. An external SDK client was denied `plugins.use` permission by LM Studio, so those tests do not establish successful end-to-end invocation from the graphical LM Studio host.
