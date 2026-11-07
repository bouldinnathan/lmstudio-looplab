# Validation records

## Current results

**[Performance charts](performance.md)** show correct answers and measured time cost for every retained quality comparison, including the earlier Devstral test. **[Public research data](research-data/README.md)** provides 1,920 retained generated answers, original main/pilot API responses, per-question grades/timings, all-model coverage and offline verification. Upstream inputs are attributed separately; the older Qwen diagnostic's missing per-case outputs are explicitly marked.

The completed [public-task quality comparison](qwen-public-benchmark.md) contains **1,600 responses: 100 fixed math/logic questions for each of four profiles on four checkpoints**. It has explicit conclusions for [Qwen3-Coder 30B](qwen-public-benchmark.md#qwen3-coder-30b), [DeepSeek-R1 32B](qwen-public-benchmark.md#deepseek-r1-32b), [DeepCoder 14B](qwen-public-benchmark.md#deepcoder-14b) and [Reader-LM 1.5B](qwen-public-benchmark.md#reader-lm-15b), alongside every original/2/3/4-pass score, timings, truncations and paired statistics. The [JSON record](qwen-public-benchmark.json) includes the fixed protocol, environment and per-question correctness without dataset text.

Qwen3-Coder's best observed result was three loops (**84 → 88/100**); DeepSeek-R1's was two loops (**72 → 78/100**). DeepCoder scored lower with every loop count (**54 → 50/52/50**). Reader-LM scored **3 → 3/5/4**, but its intended HTML-conversion task was not tested. No gain or decrease was statistically significant after correction for all 12 comparisons. Every looped batch took longer, and the 768-token limit truncated many reasoning answers. The best observed settings are experimental candidates; automatic recommendations remain baseline.

| Test record | Coverage and result | What it establishes |
|---|---|---|
| [Public-task quality comparison](qwen-public-benchmark.md) | Four checkpoints, 16 profiles, 1,600 scored responses | Observed accuracy and speed on this math/logic sample; no statistically established quality change |
| [All-local compatibility survey](all-local-survey.md) and [recovery](qwen-other-recovery.md) | 54 unique GGUF files inventoried; 17 checkpoints with 51 passing loop profiles | Which models load and pass loop identity/effect/repeatability checks; every inventoried file has a result |
| [Native CPU reference](native-cpu-repeated.json) and [Vulkan F32 reference](native-vulkan-f32.json) | 24 checks per backend across four reference architectures | Agreement with the independent numerical reference on synthetic networks |
| [LM Studio recovery/API checks](qwen-other-recovery.md#lm-studio-api-checks) and [installation audit](lmstudio-native-installation.json) | 51 indexed loop profiles; live Qwen, Granite and earlier GPT-OSS API checks | Model-picker/API integration, streaming and switching on the checked cases |
| [Application CI](https://github.com/bouldinnathan/lmstudio-looplab/actions/runs/41944570865) | 38 Python tests, 9 plugin tests, plus a fresh CPU native build/reference run | Software and numerical validation; not a model-quality benchmark |

The compatibility table covers all 54 inventoried files, including unsupported and failed cases. Quality evidence is narrower: four checkpoints have the 100-question comparison above; five have the [earlier small held-out diagnostic](#initial-local-model-results) below. **A compatibility pass alone leaves quality improvement unknown.** In particular, GPT-OSS 120B abliterated passed looping and live API checks, but has no comparative quality result here. The Qwen result should not be transferred to that checkpoint or to arbitrary abliterated models.

## Earlier integration and diagnostic records

The earlier [ordinary-model integration](lmstudio-native-integration.md) verified 37 loop profiles in LM Studio's normal model index and HTTP API, before [recovery](qwen-other-recovery.md) expanded the set to 51. GPT-OSS 120B passed 2-, 3- and 4-pass chat, streaming, a follow-up and automatic profile switching through LM Studio's actual server. The current [installation audit](lmstudio-native-installation.json) records exact API keys, unchanged source headers and descriptor storage; the earlier [API results](lmstudio-api-check.json) contain the arithmetic smoke checks.

The latest [all-local compatibility survey](all-local-survey.md) covers the expanded runtime and its generated experimental adapters. Its [JSON record](all-local-survey.json) includes loader errors, per-profile identity and repeatability checks, resource outcomes, and repair handoffs. It is a compatibility survey, not a quality evaluation. Profiles can pass anchor identity yet fail repeat-request stability; those exploration profiles remain unavailable for chat.

`native-auto-cpu.json` and `native-auto-vulkan.json` cover the expanded runtime with 24 independent-reference checks each. The CPU run explicitly disables device, operation and KV-cache offloading and automatic device fitting. Only four architectures have independent mathematical references; generated architecture candidates retain the experimental tier.

The expanded generator also passed a live 2-pass chat with Gemma 4 12B through the real companion and Vulkan worker; see [`integration-auto.json`](integration-auto.json). The controller was a test harness, so this does not independently verify LM Studio's graphical plugin host. At that stage, application checks passed 20 Python tests and 9 Node tests; see the [earlier GitHub CI run](https://github.com/bouldinnathan/lmstudio-looplab/actions/runs/41776316816). The current validation counts are listed above.

A [fresh GPT-OSS 120B abliterated check](gpt-oss-120b-abliterated.md) passed all 2-, 3- and 4-loop compatibility checks and live chat through the generator/companion, including a follow-up message. This checkpoint uses strict CPU mode. Its detailed report includes response timing, source-file verification and the graphical-host limitation.

`native-cpu.json` and `native-vulkan.json` contain 24 checks each against an independent NumPy implementation. They compare full next-token distributions through prefill and four autoregressive steps, covering four architectures, damping, anchoring, and both cache policies. The maximum allowed log-probability error is 0.005; the small tolerance accommodates F16 KV storage and different floating-point execution orders.

These checks establish numerical behavior on synthetic tiny F32 networks. Real quantized checkpoints additionally need their own anchor-identity check. A compatibility pass does not establish improved answer quality.

The named model summary files are allowlisted exports from the automatic diagnostic suite. They contain architecture and checkpoint-header identity, runtime settings, aggregate scores, truncation counts, timing, cache probes and the recommendation. Complete retained answer records are now published separately under [research-data](research-data/README.md), with provenance and checksums. Model weights, credentials and private user chats are excluded.

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
