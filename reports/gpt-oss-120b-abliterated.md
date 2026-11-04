# GPT-OSS 120B abliterated: fresh verification

Completed 2026-11-04 UTC for `gpt-oss-120b-abliterated-latest`. **All 2-, 3- and 4-loop compatibility and live-chat checks passed.**

The later [ordinary-model integration check](lmstudio-native-integration.md) also passed all three profiles through LM Studio's real HTTP server, including streaming and automatic model switching. With that integration installed, choose **GPT-OSS 120B Abliterated Latest Looped 2x**, **3x**, or **4x** in the normal model dropdown, or use API keys `gpt-oss-120b-abliterated-latest-looped-2x` / `3x` / `4x`. The generator-based checks below describe the earlier, separate path.

The original 59.0 GiB checkpoint ran in strict CPU mode with a 2,048-token context. The loop configuration uses the middle four layers (indices 16–19), layer mode, beta=0.5, first-pass KV restoration and pinned MoE routing. No checkpoint conversion or weight copy was needed.

| Loop count | Baseline-identity error | Repeat-request error | Live chat |
|---|---:|---:|---|
| 2 | 0.0 | 0.0 | Passed: `391` |
| 3 | 0.0 | 0.0 | Passed: `391` |
| 4 | 0.0 | 0.0 | Passed: `391` |

The original model and every tested loop profile answered `391` for `17 × 23`. The 2-loop profile then answered `300` when the full conversation asked it to subtract 91 from its previous answer. Streaming and the active-loop banner were checked through the real SDK generator and local companion. The controller was a test harness; the graphical LM Studio host was not independently verified.

| Chat check | Answer | Seconds including loading |
|---|---|---:|
| baseline: arithmetic | `391` | 37.1 |
| loop2: arithmetic | `391` | 37.2 |
| loop2: follow_up | `300` | 5.0 |
| loop3: arithmetic | `391` | 41.3 |
| loop4: arithmetic | `391` | 35.8 |

The initial response after each profile change includes loading the model. The follow-up reused the loaded worker; cross-request prompt-prefix caching remained disabled. These timings describe this run, not a performance guarantee.

The source GGUF retained its header hash, size and modification time after both the compatibility and chat checks. No new crash dumps were observed. The previously marked 201.4 MiB of disposable crash dumps remain marked safe to delete.

In LM Studio, open **Chat → model picker**, search **looplab**, and choose **bouldinnathan/looplab** under **Your Generators**. Enter **gpt-oss-120b-abliterated-latest** as the model in LoopLab's settings, and select **2**, **3** or **4 passes, anchored damping (experimental)**. LM Studio's Developer/server picker does not show generators, and this method creates no additional GPT-OSS model-file entry. The recommended profile still defaults to baseline; these checks establish compatibility and simple chat operation, not improved answer quality.

Machine-readable results: [fresh compatibility checks](gpt-oss-120b-abliterated-recheck.json), [live chat checks](gpt-oss-120b-abliterated-chat.json).
