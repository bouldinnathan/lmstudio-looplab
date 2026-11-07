# Public research data

**These are LoopLab's generated experimental results**, published by [bouldinnathan](https://github.com/bouldinnathan), from the local runs on November 4–6, 2026 UTC. Model answers, returned reasoning text, correctness labels, request timings, token usage and compatibility measurements are available here. The benchmark questions and reference answers come from the upstream datasets linked below; they are not presented as new LoopLab data.

Start with the [correct-answer and time-cost charts](../performance.md), [main study](../qwen-public-benchmark.md), [all-model coverage CSV](model-coverage.csv), or [per-question measurements CSV](question-metrics.csv). Download everything through the repository's **Code → Download ZIP**, [this ZIP link](https://github.com/bouldinnathan/lmstudio-looplab/archive/refs/heads/main.zip), or `git clone`. No model weights are needed to inspect or verify the results.

## What is available

| Experiment | Retained generated data | Location |
|---|---|---|
| Main Qwen-family comparison | **1,600 complete API responses**, 100 questions × 4 profiles × 4 checkpoints; full payloads, final answers, returned reasoning, timings, token counts, finish reasons, all harness filters and grades | [qwen-lm-eval-v1](qwen-lm-eval-v1), [protocol](qwen-lm-eval-v1/protocol.json) |
| Earlier integration pilot | **8 complete API responses**, 4 questions × original/2-loop Qwen3-Coder; kept separate from the main study | [qwen-lm-eval-pilot](qwen-lm-eval-pilot) |
| Small diagnostics | **312 saved case answers**: 72 each for DeepCoder, DeepSeek-R1, Devstral and Reader-LM, plus 24 from the later Qwen quick run; correctness, timings, usage and available baseline probes | [diagnostics](diagnostics), [case definitions](diagnostic-cases.json) |
| Initial Qwen3-Coder 12+12 diagnostic | Scores and timings for all evaluated profiles; **original per-case outputs unavailable** because the later quick run replaced that saved record | [original summary](../qwen3-coder-30b.json) |
| Compatibility exploration | 54 file records, including failures, skipped/unsupported cases, per-profile conformance and backend recovery | [exploration](exploration), [coverage CSV](model-coverage.csv), [human-readable survey](../all-local-survey.md) |
| Repair/recovery diagnostics | 33 retained repair handoffs, including loader errors, tensor metadata and reproduction commands; seven-model precision-recovery record | [repairs](repairs), [precision recovery](additional/precision-recovery.json) |
| Native numerical and LM Studio API tests | Existing numerical-reference, generator, chat, streaming and switching records | [report index](../README.md) |

The five checkpoints with quality measurements are **Qwen3-Coder 30B, DeepCoder 14B, DeepSeek-R1 32B, Reader-LM 1.5B and Devstral 24B**. Devstral only has the smaller diagnostic. The coverage CSV identifies all 54 inventoried model files and whether each has quality evidence. GPT-OSS 120B abliterated, for example, has compatibility/API results but no comparative quality score. A compatibility pass is not a correct-answer benchmark.

There are **1,920 retained generated quality answers** across the main study, pilot and small diagnostics. They are not 1,920 independent questions: profiles reuse the same questions, and the studies have different protocols. Keep their results separate. The original Qwen diagnostic's missing 72 case outputs are not included in that count.

## Read one generated response

Each `requests.jsonl` line has `task`, `doc_id`, `request_sha256` and `record`:

| Field | Origin and meaning |
|---|---|
| `record.payload.messages` | Exact chat messages submitted to the runtime, using upstream benchmark text |
| Other `record.payload` fields | Actual request controls: model ID, temperature, seed, output limit, stop strings and template/cache settings |
| `record.response.choices[0].message.content` | **Generated final-answer text**, including incomplete text when generation was truncated |
| `record.response.choices[0].message.reasoning_content` | **Generated reasoning text, when returned by the runtime**; its absence does not imply no internal reasoning |
| Other `record.response` fields | Original API response: finish reason, usage, native timings, IDs and creation time |
| `record.seconds` | Measured client-side request latency, including prompt evaluation and generation |
| `record.rendered_prompt_tokens` | Length from the runtime's preflight template/tokenization check; exact rendered token IDs were not saved |
| `task`, `doc_id` | Task name and original zero-based dataset row index; join to the pinned dataset revision in `protocol.json` |
| `request_sha256` | SHA256 of the original payload serialized with sorted keys and `ensure_ascii=False`; retains the original request-file identity |

For example, from the repository root:

```python
import json
from pathlib import Path

path = Path("reports/research-data/qwen-lm-eval-v1/qwen3-coder-30b-0fd305b1/loop3/requests.jsonl")
row = json.loads(path.read_text().splitlines()[0])
print(row["task"], row["doc_id"])
print(row["record"]["response"]["choices"][0]["message"])
print("Request seconds:", row["record"]["seconds"])
```

The neighboring `harness.json` contains all scored samples, both strict and flexible extraction results, upstream documents/targets, prompt/question/target hashes, task configurations and original harness scores. It may contain multiple filter rows for the same question. The **full original generated text is in `requests.jsonl`**; the scorer deliberately replaced truncated final answers with an empty string before extracting an answer. Do not mistake that scorer input for the actual generation.

`manifest.json` beside each profile records its exact loop settings, runtime/patch fingerprints, source GGUF size and header hash. [`qwen-benchmark-environment.json`](../qwen-benchmark-environment.json) records the hardware, Python/packages, native binary hash and checkpoint quantizations. These identify the local files as recorded; full model-weight hashes were not measured, and no weights are redistributed.

## Tables and grading

- [`question-metrics.csv`](question-metrics.csv): all 1,608 main/pilot requests with stable joins, reported correctness, decimal-normalized correctness, original official correctness, timing, output length and finish reason. In the main study, `correct` uses decimal normalization; in the older pilot it preserves the official scorer used at the time. The pilot remains **3/4 versus 3/4** in its original report and chart; retrospective decimal normalization gives 4/4 for both.
- [`performance.csv`](performance.csv): all **54 measured study/model/split/profile rows** used by the five charts. `request_seconds` is the sum of requests for that row, not model load time, energy use or a fixed-length throughput measurement. Missing holdout profiles are absent rather than zero. `raw_answers=summary_only` identifies the older Qwen diagnostic.
- [`model-coverage.csv`](model-coverage.csv): all 54 inventoried files, compatibility outcome, presence of each quality study and a link to the retained compatibility record.

The main study uses official flexible extraction, plus exact-decimal normalization for GSM8K. Truncations count as wrong. The official scores and filtered answers are retained alongside the normalized scores. See [the grading/paired-statistics code](../../looplab/quality_metrics.py), [original runner](../../scripts/benchmark_qwen.py) and [protocol limitations](../qwen-public-benchmark.md#protocol-and-limitations). All generated answer text is inert data; the evaluation did not execute it.

## Verify and reproduce

From a checkout or extracted ZIP, Python 3.10+ is sufficient:

```sh
python scripts/verify_research_data.py
```

The verifier checks the SHA256 manifest, original request hashes, exact question/prompt joins, recorded output versus scorer input, sample coverage, all published main scores and timings, paired statistics, all 312 retained diagnostic answers, chart-source rows and all-model coverage. It needs no model, server, downloaded dataset or third-party Python package. **It reuses the harness's saved extraction labels**, then recomputes normalization and statistics; it does not independently rerun the harness's regex filters. Researchers can apply their own graders to the unchanged generated outputs.

To regenerate the figures, install the optional plotting dependency in an environment of your choice:

```sh
python -m pip install -r requirements-plots.txt
python scripts/plot_results.py
```

The figures are available as PNG and editable vector SVG under [charts](../charts), with a [chart manifest](../charts/manifest.json) tying them to the CSV and plotting script. To repeat inference with the original models, follow the [benchmark instructions](../../README.md#optional-public-task-qwen-comparison); the recorded protocol and environment are the reference. To publish another retained local snapshot, use [`export_research_data.py`](../../scripts/export_research_data.py), review the export and rerun the verifier. Exporting is a separate action; ordinary evaluation does not publish anything.

## Provenance, licensing and known gaps

[`provenance.json`](provenance.json) separates project-generated outputs from third-party inputs. Exact upstream dataset versions are:

- [GSM8K, revision `740312add88f781978c0658806c59bc2815b9866`](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866), from [OpenAI's grade-school-math project](https://github.com/openai/grade-school-math).
- [BBH, revision `b5306be6f827cfafbb545ff5a51f96916029b0fd`](https://huggingface.co/datasets/SaylorTwift/bbh/tree/b5306be6f827cfafbb545ff5a51f96916029b0fd), a mirror of [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard).
- Task definitions and filters: [EleutherAI lm-evaluation-harness 0.4.13](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.13).

The full upstream datasets and evaluation software are linked, not bundled. The selected questions, targets and task configurations are retained inside the recorded requests/harness results to make this experiment inspectable. Their MIT notices are preserved in [upstream-licenses](upstream-licenses). Project code, original measurements, annotations and charts use the repository's [AGPL-3.0-or-later license](../../LICENSE), to the extent copyright applies. This does not change the rights in upstream benchmark text, model weights or any third-party material reproduced by a model.

The [data manifest](manifest.json) gives file sizes, checksums and every redaction. Only 54 absolute installation-directory prefixes in harness `metadata.config_source` were removed; package-relative task paths remain. Prompts, model output text, recorded timing and scoring were not redacted. Original per-request objects were grouped into JSONL and associated with question IDs; their JSON values and payload fingerprints are unchanged.

Some data never survived the original runs: the older Qwen diagnostic's per-case outputs; full HTTP responses/separate reasoning from small diagnostics; warm-up responses; exact rendered chat-template token IDs; and most raw compatibility-probe distributions. The available original diagnostic baseline probes are included. Rotating private worker/service logs are excluded because they can mix experiment traffic with interactive use. Credentials, user chats, caches, model files and crash dumps are not research artifacts in this publication. These gaps are explicit in the provenance record; no missing outputs have been reconstructed or replaced with a new run.

For a stable research reference, cite this repository **at the exact commit you downloaded**, the study name (`qwen-lm-eval-v1` for the main comparison), and `manifest.json`. Dataset revisions and native runtime fingerprints are fixed within the data; the repository's `main` branch can change.
