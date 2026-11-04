# All local model survey

Completed 2026-11-04 UTC. Inventoried **54 unique GGUF files**, deduplicating LM Studio links and Ollama blobs. Discovery errors: 0.

**14 models have at least one passing loop profile**, including **10 models using newly generated experimental adapters**. These are compatibility checks, not quality benchmarks. No profile was automatically recommended.

For generation models that loaded and had adapters, the sweep checked K=2/3/4 with beta=1 for baseline identity, then beta=0.5 for finite output, a measurable loop effect and repeat-request stability. Every loading or loop failure has an explicit outcome. Embeddings were checked for finite vectors; projectors and legacy static-repeat variants were inventoried separately.

| Outcome | Files |
|---|---:|
| baseline failed | 20 |
| baseline only | 6 |
| compatible experimental | 10 |
| conformance failed | 3 |
| embedding passed | 2 |
| not applicable | 3 |
| partially compatible | 4 |
| preflight skipped | 5 |
| resource skipped | 1 |

| Model | Architecture | Baseline check | Passing loop counts | Outcome |
|---|---|---|---|---|
| gemma4-12b | clip | not run | — | not applicable |
| embeddinggemma-latest | gemma3 | failed | — | baseline failed |
| qwen3.5-122b-uncensored-q4_k_p | clip | not run | — | not applicable |
| reader-lm-1.5b | qwen2 | passed | 2, 4 | partially compatible |
| bge-m3-latest | bert | embedding passed | — | embedding passed |
| muse-glimmer-30b | clip | not run | — | not applicable |
| granite4.1-3b | granite | passed | 2 | partially compatible |
| glm-ocr-latest | glmocr | failed | — | baseline failed |
| qwen3-embedding-4b | qwen3 | embedding passed | — | embedding passed |
| bespoke-minicheck-latest | internlm2 | passed | 2, 3, 4 | compatible experimental |
| granite4.1-guardian-8b-q4_K_M | granite | passed | 2, 4 | partially compatible |
| ornith-9b | qwen35 | passed | — | baseline only |
| qwen3.5-9b | qwen35 | failed | — | baseline failed |
| deepseek-ocr-3b | deepseekocr | failed | — | baseline failed |
| gemma4-12b | gemma4 | passed | 2, 3, 4 | compatible experimental |
| translategemma-12b | gemma3 | failed | — | baseline failed |
| deepcoder-14b | qwen2 | passed | 2, 3, 4 | compatible experimental |
| ministral-3-14b | mistral3 | failed | — | baseline failed |
| deepcoder-14b-looped-2x-mid4 | qwen2 | not run | — | preflight skipped |
| phi4-reasoning-plus | phi3 | passed | — | conformance failed |
| gpt-oss-20b | gptoss | failed | — | baseline failed |
| devstral-24b | llama | passed | 2, 3, 4 | compatible experimental |
| lfm2-24b | lfm2moe | passed | — | baseline only |
| devstral-small-2-24b | mistral3 | failed | — | baseline failed |
| devstral-24b-looped-2x-mid4 | llama | not run | — | preflight skipped |
| muse-glimmer-30b | muse-glimmer | passed | 2, 3, 4 | compatible experimental |
| granite4.1-30b | granite | passed | — | conformance failed |
| gemma4-26b | gemma4 | failed | — | baseline failed |
| qwen3-coder-30b | qwen3moe | passed | — | conformance failed |
| north-mini-code-1.0-q4_K_M | cohere2moe | passed | 2, 3, 4 | compatible experimental |
| command-r-35b | command-r | passed | 2, 3, 4 | compatible experimental |
| glm-4.7-flash-q4_K_M | glm4moelite | failed | — | baseline failed |
| olmo-3.1-32b-think | olmo3 | failed | — | baseline failed |
| qwen3-vl-30b-a3b-instruct-q4_K_M | qwen3vlmoe | failed | — | baseline failed |
| deepseek-r1-32b | qwen2 | passed | 2, 3, 4 | compatible experimental |
| gemma4-31b | gemma4 | failed | — | baseline failed |
| qwen3-coder-30b-looped-2x-mid4 | qwen3moe | not run | — | preflight skipped |
| laguna-xs-2.1-q4_K_M | laguna | passed | 2, 3 | partially compatible |
| ornith-35b | qwen35moe | passed | — | baseline only |
| qwen3.5-35b | qwen35moe | failed | — | baseline failed |
| qwen3.6-35b-a3b-q4_K_M | qwen35moe | failed | — | baseline failed |
| nemotron-3.5-lightning-30b | nemotron_h_moe | passed | — | baseline only |
| nemotron3-33b | nemotron_h_omni | failed | — | baseline failed |
| glm-4.7-flash-q8_0 | deepseek2 | passed | — | baseline only |
| qwen2.5vl-72b | qwen25vl | failed | — | baseline failed |
| qwen3-next-80b-a3b-thinking | qwen3next | passed | — | baseline only |
| qwen3-coder-next-q4_K_M | qwen3next | failed | — | baseline failed |
| command-r-plus-latest | command-r | passed | 2, 3, 4 | compatible experimental |
| gpt-oss-120b-abliterated-latest | gpt-oss | passed | 2, 3, 4 | compatible experimental |
| gpt-oss-120b | gptoss | failed | — | baseline failed |
| llama4-scout | llama4 | failed | — | baseline failed |
| qwen3.5-122b-uncensored-q4_k_p | qwen35moe | not run | — | resource skipped |
| nemotron-3-super-120b | nemotron_h_moe | not run | — | preflight skipped |
| Kimi-K2.7-Code | deepseek2 | not run | — | preflight skipped |

The [complete JSON report](all-local-survey.json) includes per-profile identity/repeatability measurements, native-loader errors, adapter-source hints and repair handoffs. Local repair files are also in `~/.local/state/looplab/repairs/`. A generated adapter is weaker evidence than the four architectures covered by the independent NumPy reference.

Storage: additional runtime source/build files use **551.8 MiB**; all LoopLab state uses **5.2 MiB**. **No model-weight copies were written.** All 54 source files retained their header hash, size and modification time. Free space after testing: **484.9 GiB**.

The initial Vulkan fallback failures left **201.4 MiB** in 3 root-owned system crash dumps. Those are included in the storage audit and remain outside the repository. CPU fallback now disables device, operation and KV offloading explicitly.

Re-run with `python -m looplab explore all`. Completed results resume automatically; use `--no-resume` for fresh inference. The plugin provides **Explore all models, including experimental adapters**. Select passing loop counts manually; the recommended profile remains baseline.
