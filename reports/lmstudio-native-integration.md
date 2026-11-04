# Ordinary model entries and LM Studio API verification

Verified on 2026-11-04 with LM Studio **0.4.25+1**, LoopLab bridge **0.3.0**, and the separately named **looplab-linux-x86_64@2.69.0** runtime. This test used the real LM Studio HTTP server, model index and just-in-time loader.

**37 tested profiles across 14 checkpoints** are installed as ordinary model entries. All 37 were found in both `lms ls` and LM Studio's `/v1/models`. They can be selected in the ordinary model dropdown and requested by API model key. The generator plugin remains available separately for discovery, testing and reports.

| Requested model / check | Transport | Answer | Seconds, including any loading |
|---|---|---|---:|
| `gpt-oss-120b-abliterated-latest-looped-2x`: 17 × 23 | JSON | 391 | 102.703 |
| Same model: subtract 91 from the previous answer | SSE | 300 | 4.399 |
| `gpt-oss-120b-abliterated-latest-looped-3x`: 17 × 23 | SSE | 391 | 47.037 |
| `gpt-oss-120b-abliterated-latest-looped-4x`: 17 × 23 | SSE | 391 | 46.621 |

Requests went to LM Studio's normal `http://127.0.0.1:1234/v1/chat/completions` endpoint. The script switched model IDs without manual unloads; LM Studio's enabled “Only Keep Last JIT Loaded Model” setting unloaded the previous profile. Runtime logs identified the active pass count, strict CPU execution, 2,048-token context and disabled prefix caching. The first request includes a cold load; timings are observations, not guarantees.

A separate two-pass Reader-LM check also returned text through LM Studio using the Vulkan worker. The original, unmodified Reader-LM entry then loaded and returned text through the same selected runtime's official-engine passthrough. These are integration smoke tests; they do not establish an answer-quality improvement.

The installed model-entry files total **284,692 bytes (278.0 KiB)**. No weights were copied. Official runtime libraries and executables share their existing file inodes; only the custom launcher and metadata are new. All 54 original checkpoint headers and sizes matched the earlier complete survey. The three old crash dumps, **211,216,673 bytes (201.4 MiB)**, remain marked deletable; no new crash dumps were created.

The entries are small descriptors with no usable model tensors. A direct stock-runtime load of a descriptor failed during vocabulary loading, confirming that selecting a stock engine cannot silently run a baseline model under a looped name. The custom launcher verifies descriptor ownership, source identity, the native build fingerprint and current per-profile conformance before loading the real source file.

LM Studio normalizes some API keys by removing quantization tags. Use the actual keys returned by `lms ls` or `/v1/models`; the installation record below contains those exact keys. Rollback matches indexed file paths rather than potentially normalized or ambiguous names.

The integration preserves LM Studio's existing network and authentication settings. This machine's verification used loopback; access from a second computer was not tested.

Records: [API chat checks](lmstudio-api-check.json), [installed entries and storage audit](lmstudio-native-installation.json). Reproduce chat checks with:

```sh
.venv/bin/python scripts/check_lmstudio_api.py --output reports/local/lmstudio-api.json
```

Application tests cover descriptor storage, conformance invalidation, automatic removal of failed profiles, preservation of user-edited settings, official-runtime hard links, rollback with normalized names, transport authentication, SSE forwarding and mandatory prefix-cache disabling. The native numerical operator was unchanged by this integration; its independent reference results remain recorded separately.
