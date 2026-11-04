# Qwen and related-model recovery

Verified on 2026-11-04 with LoopLab 0.3.1, LM Studio 0.4.25+1 and the existing LoopLab runtime pack `looplab-linux-x86_64@2.69.0`. No native-runtime rebuild or model conversion was needed.

**51 passing profiles across 17 checkpoints** are now indexed in both LM Studio's ordinary model picker and `/v1/models`, up from 37 profiles across 14 checkpoints. All seven affected checkpoints now pass K=2/3/4:

| Model | Previously passing loops | Now passing | Execution |
|---|---|---|---|
| reader-lm-1.5b | 2, 4 | 2, 3, 4 | CPU |
| granite4.1-3b | 2 | 2, 3, 4 | GPU, F32 compute |
| granite4.1-guardian-8b-q4_K_M | 2, 4 | 2, 3, 4 | CPU |
| phi4-reasoning-plus | none | 2, 3, 4 | GPU, F32 compute |
| granite4.1-30b | none | 2, 3, 4 | CPU |
| qwen3-coder-30b | none | 2, 3, 4 | GPU, F32 compute |
| laguna-xs-2.1-q4_K_M | 2, 3 | 2, 3, 4 | GPU, existing settings |

For Qwen3-Coder, the unfixed Vulkan run repeated baseline exactly but changed loop probabilities between the first and subsequent requests: maximum errors were 0.013094, 0.773222 and 0.084794 at K=2/3/4. Disabling Vulkan descriptor reuse, graph reuse or submission overlap did not resolve the K=3 discrepancy. Disabling F16 compute did; the resulting GPU setting passed all three anchor-identity and repeat-request checks. The quantized source weights and F16 KV format remain unchanged.

Exploration now tries F32 Vulkan compute and then strict CPU execution when numerical checks fail. It keeps the original tolerance and only adopts a setting that adds passing profiles without losing previously passing ones. Settings are bound to checkpoint identity and the native-build fingerprint, shared by the companion and LM Studio, and included in conformance validation. Failed or cancelled trials do not replace active settings. Ordinary Auto-test runs this stage before its quality benchmark.

Laguna's fourth profile exposed a separate checker bug: its top-k token set changed enough that the full-distribution comparator could not align it. The checker now recognizes an effect when the probability of a shared token measurably changes. Identity and repeatability still require the original strict full comparison. A fresh sweep passed all three profiles with Laguna's existing GPU settings.

## LM Studio API checks

The exact Qwen keys are `qwen3-coder-30b-looped-2x`, `qwen3-coder-30b-looped-3x` and `qwen3-coder-30b-looped-4x`. These are the new numerical recurrence profiles; the older `qwen3-coder-30b-looped-2x-mid4` entry is a separate static-repeat conversion.

Through LM Studio's normal `/v1/chat/completions` endpoint, both Qwen3-Coder and Granite 30B returned **391** for 17 × 23 at all three loop counts, and **300** for a two-pass follow-up subtracting 91. JSON and SSE responses passed, and just-in-time model switching worked. Qwen's active native worker also had the expected four-pass and F32-compute settings.

A later attempt to load another profile while a manually loaded LoopLab profile held the worker was blocked by the existing single-worker guard. Unload a manually loaded loop profile before starting Auto-test or loading another one. Ordinary models use the official-engine passthrough and do not share this worker lease. Network and authentication settings remain unchanged.

Records: [Qwen API](qwen-lmstudio-api.json), [Granite API](granite-lmstudio-api.json), [all indexed entries](lmstudio-native-installation.json), [recovery details](qwen-other-recovery.json).

## Quality testing

The quick Qwen diagnostic tied baseline: **3/4 selection answers correct for baseline and every loop count**. The selected two-pass setting also tied baseline at **3/4 held-out answers**, with zero paired wins and losses. This is not evidence of an improvement; the recommendation remains baseline. [Quick quality report](qwen-recovery-quality.json).

For the full built-in diagnostic, select **Auto-test selected model** and turn **Quick compatibility screen** off, or run:

```sh
.venv/bin/python -m looplab test qwen3-coder-30b
```

That uses 12 selection tasks and 12 separate held-out tasks. A loop recommendation requires the selected candidate to meet the paired significance threshold on holdout. The suite is a small public diagnostic, not a broad capability benchmark or reproduction of the paper's scores.

## Validation and storage

- 33 Python tests and 9 plugin tests passed.
- 24 independent NumPy/native checks passed on CPU and 24 on Vulkan with F32 compute, including repeated requests. [CPU](native-cpu-repeated.json), [Vulkan F32](native-vulkan-f32.json).
- [CI passed](https://github.com/bouldinnathan/lmstudio-looplab/actions/runs/41874166540), including a fresh CPU runtime build.
- All 54 original GGUF headers, sizes and recorded modification times match. This is not a full hash of every tensor byte.
- Descriptor files total **374,700 bytes (365.9 KiB)**, an increase of **90,008 bytes (87.9 KiB)**. No model weights were copied or converted. Existing runtime binaries were reused.
- The earlier **201.4 MiB of crash dumps remain marked deletable**. They are unrelated to inference and need administrator access for removal.

Qwen 3.5/3.6, Qwen Next and other recurrent/SSM families remain excluded from looping. Some local GGUF variants also have native-loader metadata/tensor incompatibilities. Those failures remain visible in the [complete local survey](all-local-survey.md); this recovery does not claim support for arbitrary architectures.
