# Qwen original vs looped: public-task comparison

No loop profile showed a statistically significant accuracy gain after correcting for all 12 comparisons. This study does not establish that looping improves these models; the baseline recommendation remains unchanged.

Completed 100 questions for each of four profiles on four local Qwen-family checkpoints: **1,600 responses**. These are fixed subsets of public math and logic tasks, not official full-benchmark scores.

| Local checkpoint | Original | 2 loops | 3 loops | 4 loops |
|---|---:|---:|---:|---:|
| qwen3-coder-30b | 84/100 | 82/100 | 88/100 | 86/100 |
| reader-lm-1.5b | 3/100 | 3/100 | 5/100 | 4/100 |
| deepcoder-14b | 54/100 | 50/100 | 52/100 | 50/100 |
| deepseek-r1-32b | 72/100 | 78/100 | 75/100 | 73/100 |

## Results for each model

Across the 12 loop-versus-original comparisons, seven scores increased, one tied and four decreased. These comparisons share questions and checkpoints; they are not 12 independent demonstrations of a general benefit. Every looped batch took longer than its original-model batch. Four loops never achieved the highest score for any of these checkpoints. Neither an accuracy increase nor a decrease was statistically significant after correction for all 12 comparisons.

The interpretations below describe this fixed 100-question sample and its 768-token output limit. An experimental choice based on the best observed score is a judgment from these results; it is not an automatic promotion or a validated improvement on unseen tasks. All automatic recommendations remain on the original model.

### Qwen3-Coder 30B

**Observed result: a small gain at three loops.** Three loops scored **88/100 versus 84/100** for the original, correcting five previously wrong answers and spoiling one previously correct answer. The batch took **15.3% longer**. Both profiles had six truncated answers. Two loops scored 82 and four loops scored 86, so the direction depended on the loop count.

**Practical interpretation:** three loops is the most promising experimental choice for this checkpoint when quality matters more than compute cost. The gain is not statistically established, and this math/logic test did not measure coding quality. The two-sided paired p-value was 0.21875 before correction and 1.0 after correction.

### DeepSeek-R1 32B

**Observed result: the largest score gain was at two loops.** Two loops scored **78/100 versus 72/100**, correcting eight answers and spoiling two. The batch took **6.5% longer**. Three loops scored 75 and four loops scored 73. This is the local dense **DeepSeek-R1-Distill-Qwen-32B** checkpoint, not a result for every DeepSeek model.

**Practical interpretation:** two loops is the most promising experimental choice for this checkpoint. Truncations fell from **22 to 16**, so completion within the fixed token budget matters to the result. Math accuracy stayed at 44/50 for every profile; the net gains came from the logic tasks. The two-sided paired p-value was 0.109375 before correction and 1.0 after correction. The gain is not statistically established.

### DeepCoder 14B

**Observed result: every loop count scored lower overall.** The original scored **54/100**; two, three and four loops scored **50, 52 and 50**. Looped batches took **10.5%–24.6% longer**. Three loops improved the math subtotal from 41/50 to 43/50, but the combined logic score fell enough to reduce the total.

**Practical interpretation:** retain the original for this tested workload and budget. **39–44 answers per profile were truncated**, which substantially limits conclusions about longer reasoning or coding tasks. The observed decreases were not statistically established either; this is an unfavorable sample result, not proof that looping always harms this checkpoint.

### Reader-LM 1.5B

**Observed result: scores stayed very low, with no useful conclusion for its intended task.** The original scored **3/100**, and two, three and four loops scored **3, 5 and 4**. Every profile scored zero on both logic tasks, and 67–83 responses per profile had an empty final answer. Looped batches took **37.5%–75.2% longer**.

**Practical interpretation:** the 3-to-5 change is not evidence that HTML conversion improved. [Reader-LM is trained for HTML-to-Markdown conversion](https://huggingface.co/jinaai/reader-lm-1.5b), which this benchmark did not test. Its three-loop score is the best observed score on these outside-domain questions, not a validated choice for its actual workload.

## Paired changes

Wins and losses compare answers to the same questions. Positive changes favor looping. Intervals are 95% stratified paired bootstrap intervals in percentage points; p-values are exact two-sided paired tests, with Holm correction across all 12 comparisons.

| Model | Loops | Change | Wins / losses | 95% interval | Adjusted p |
|---|---:|---:|---:|---:|---:|
| qwen3-coder-30b | 2 | -2 | 2 / 4 | [-7, 3] | 1 |
| qwen3-coder-30b | 3 | +4 | 5 / 1 | [0, 9] | 1 |
| qwen3-coder-30b | 4 | +2 | 5 / 3 | [-3, 8] | 1 |
| reader-lm-1.5b | 2 | +0 | 1 / 1 | [-3, 3] | 1 |
| reader-lm-1.5b | 3 | +2 | 3 / 1 | [-2, 6] | 1 |
| reader-lm-1.5b | 4 | +1 | 2 / 1 | [-2, 4] | 1 |
| deepcoder-14b | 2 | -4 | 4 / 8 | [-11, 3] | 1 |
| deepcoder-14b | 3 | -2 | 5 / 7 | [-9, 4] | 1 |
| deepcoder-14b | 4 | -4 | 5 / 9 | [-11, 3] | 1 |
| deepseek-r1-32b | 2 | +6 | 8 / 2 | [0, 12] | 1 |
| deepseek-r1-32b | 3 | +3 | 7 / 4 | [-3, 9] | 1 |
| deepseek-r1-32b | 4 | +1 | 3 / 2 | [-3, 5] | 1 |

## Task scores and output limits

| Model | Passes | Math /50 | Boolean /25 | Ordering /25 | Truncated | Empty final | Standard scorer total |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3-coder-30b | 1 | 42 | 24 | 18 | 6 | 0 | 81/100 |
| qwen3-coder-30b | 2 | 42 | 24 | 16 | 10 | 0 | 79/100 |
| qwen3-coder-30b | 3 | 43 | 24 | 21 | 6 | 0 | 85/100 |
| qwen3-coder-30b | 4 | 42 | 24 | 20 | 4 | 0 | 83/100 |
| reader-lm-1.5b | 1 | 3 | 0 | 0 | 4 | 83 | 3/100 |
| reader-lm-1.5b | 2 | 3 | 0 | 0 | 2 | 70 | 3/100 |
| reader-lm-1.5b | 3 | 5 | 0 | 0 | 3 | 68 | 5/100 |
| reader-lm-1.5b | 4 | 4 | 0 | 0 | 3 | 67 | 4/100 |
| deepcoder-14b | 1 | 41 | 12 | 1 | 39 | 31 | 52/100 |
| deepcoder-14b | 2 | 40 | 8 | 2 | 44 | 36 | 48/100 |
| deepcoder-14b | 3 | 43 | 7 | 2 | 43 | 34 | 50/100 |
| deepcoder-14b | 4 | 40 | 8 | 2 | 44 | 33 | 48/100 |
| deepseek-r1-32b | 1 | 44 | 21 | 7 | 22 | 15 | 68/100 |
| deepseek-r1-32b | 2 | 44 | 23 | 11 | 16 | 11 | 74/100 |
| deepseek-r1-32b | 3 | 44 | 22 | 9 | 18 | 10 | 71/100 |
| deepseek-r1-32b | 4 | 44 | 22 | 7 | 21 | 13 | 69/100 |

## Observed speed

Loading and warm-up are excluded. Request latency includes prompt evaluation and generation; answers can have different lengths. Native decode rates exclude prompt evaluation. This is one sequential run per profile, with randomized profile order, not a controlled fixed-output-length speed test. Inference ran on a Ryzen AI 9 HX 370 with Radeon 890M shared-memory graphics and 93.6 GiB RAM.

| Model | Passes | Mean request (s) | Median (s) | Output tokens | Decode tokens/s | Total request time vs original |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-coder-30b | 1 | 10.44 | 7.51 | 27581 | 30.22 | 1.00× |
| qwen3-coder-30b | 2 | 11.43 | 7.97 | 28475 | 28.24 | 1.09× |
| qwen3-coder-30b | 3 | 12.04 | 8.50 | 27152 | 25.71 | 1.15× |
| qwen3-coder-30b | 4 | 12.93 | 8.81 | 28150 | 24.45 | 1.24× |
| reader-lm-1.5b | 1 | 1.36 | 0.38 | 5831 | 60.64 | 1.00× |
| reader-lm-1.5b | 2 | 1.87 | 0.80 | 7688 | 54.43 | 1.37× |
| reader-lm-1.5b | 3 | 2.29 | 0.93 | 8688 | 49.12 | 1.68× |
| reader-lm-1.5b | 4 | 2.38 | 1.03 | 8161 | 45.13 | 1.75× |
| deepcoder-14b | 1 | 71.62 | 72.02 | 58637 | 8.47 | 1.00× |
| deepcoder-14b | 2 | 79.15 | 85.72 | 60346 | 7.87 | 1.11× |
| deepcoder-14b | 3 | 84.21 | 91.84 | 60194 | 7.37 | 1.18× |
| deepcoder-14b | 4 | 89.25 | 98.11 | 60027 | 6.92 | 1.25× |
| deepseek-r1-32b | 1 | 133.26 | 121.17 | 51399 | 3.97 | 1.00× |
| deepseek-r1-32b | 2 | 141.91 | 134.76 | 51624 | 3.74 | 1.06× |
| deepseek-r1-32b | 3 | 148.28 | 140.69 | 51163 | 3.55 | 1.11× |
| deepseek-r1-32b | 4 | 157.14 | 148.97 | 51644 | 3.38 | 1.18× |

## Protocol and limitations

- EleutherAI lm-evaluation-harness 0.4.13; GSM8K and BBH zero-shot chain-of-thought tasks. Dataset revisions and exact sample indices were fixed before the run and are included in the JSON report.
- The same questions, original GGUF, native runtime, precision, chat template, 2,048-token context, greedy decoding, seed 1729 and 768-token output budget were used for all four profiles of each checkpoint. The checkpoint non-thinking template was requested where supported; CoT remained in the benchmark prompt.
- Loops use the four middle layers, beta 0.5 and first-pass cache restoration. Qwen3-Coder uses layer recurrence and Vulkan F32 compute; DeepCoder and DeepSeek use block recurrence and their existing Vulkan settings; Reader-LM uses block recurrence on CPU. This is the runtime used by the LM Studio loop descriptors, accessed directly for the comparison. Frontend overhead and the stock LM Studio engine were not measured.
- The official flexible extraction filters were used. GSM8K results additionally normalize exact decimal values so "$7.00." matches "7". Original harness scores are retained above and in JSON. Truncated answers are wrong under both scoring variants. No generated code or external LLM judge was used.
- The aggregate weights math at 50% and each logic task at 25%. It does not measure coding, long-context use, general chat, HTML conversion or all capabilities. Reader-LM is specialized for [HTML-to-Markdown conversion](https://huggingface.co/jinaai/reader-lm-1.5b); these are outside-domain probes for it.
- A 768-token budget can disadvantage models that generate long reasoning traces. Empty and truncated answers are reported explicitly. No result here establishes the paper's best window, an optimal loop count or general superiority.
- Model recommendations and LM Studio profile entries were not changed by this evaluation.

## Reproduce and inspect

See the [benchmark instructions](../README.md#optional-public-task-qwen-comparison) and [runner](../scripts/benchmark_qwen.py). The [JSON report](qwen-public-benchmark.json) contains all statistics, paired per-question correctness without benchmark text, exclusions, protocol and environment. Private prompts and raw responses remain under `~/.local/state/looplab/benchmarks/qwen-lm-eval-v1/`.

The excluded Qwen 3.5/3.6/Next and incompatible VL checkpoints still lack validated looping profiles. Legacy static repeated-layer conversions were not used as numerical-loop candidates.

## Validation and storage

- 38 Python tests and 9 plugin tests passed. [CI run](https://github.com/bouldinnathan/lmstudio-looplab/actions/runs/41944570865).
- Audited 1600 saved responses against the declared questions and settings. All source GGUF sizes, modification times and header hashes match their pre-run manifests; full tensor hashes were not recomputed. No weights were copied or downloaded, and observed prompt-cache reuse was zero.
- Allocated space reported by `du`: evaluation environment **736.3 MiB**, dataset cache **7.4 MiB**, and saved run records **15.6 MiB**. These figures exclude any pre-existing pip cache and are not a measurement of network transfer. The evaluation environment and dataset cache can be removed after testing without affecting model inference; they would need reinstalling to rerun this benchmark.
- The earlier **201.4 MiB of crash dumps remain marked deletable**; this evaluation did not remove them.
