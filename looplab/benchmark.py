"""Small reproducible probes, with separate selection and held-out tasks.

This is a local diagnostic suite, not a reproduction of the paper's benchmark.
Generated text is parsed as JSON and is never executed.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
from dataclasses import asdict
import hashlib
import json
import math
import re
import time

from .models import Profile, atomic_json, storage_snapshot

SUITE_VERSION = "looplab-diagnostic-v2"
EVALUATION_POLICY = {"temperature": 0, "seed": 1729, "max_tokens": 192,
                     "cache_prompt": False, "reasoning_effort": "none", "reasoning_budget_tokens": 0,
                     "add_generation_prompt": False, "continue_final_message": "content",
                     "response_format": {"type": "json_object"},
                     "chat_template_kwargs": {"enable_thinking": False}}
ANSWER_PREFIX = '{"answer":'
CASES = [
    ("sort", "Sort [8, -2, 4, 4, 0] numerically in ascending order.", [-2, 0, 4, 4, 8]),
    ("dedupe", "Remove duplicate integers from [3,1,3,2,1,4], preserving first occurrence order.", [3, 1, 2, 4]),
    ("trace", "Python: a=[1,2]; b=a; b.append(3). What is a?", [1, 2, 3]),
    ("overlap", "Merge overlapping closed intervals [[1,4],[2,6],[9,10],[10,12]].", [[1, 6], [9, 12]]),
    ("modulo", "What is the integer value of Python's (-17) % 5?", 3),
    ("count", "How many substrings equal to 'aa' occur in 'aaaaa', allowing overlaps?", 4),
    ("fibonacci", "Given F(0)=0,F(1)=1,F(n)=F(n-1)+F(n-2), what is F(10)?", 55),
    ("path", "Undirected edges: A-B, B-C, A-D, D-E, E-C. Minimum number of edges from A to C?", 2),
    ("bisect", "Return the zero-based insertion index for 4 in [1,3,4,4,8], before existing equal values.", 2),
    ("unicode", "Reverse the string 'ab12CD' character by character.", "DC21ba"),
    ("inclusion", "How many integers from 1 through 30 inclusive are divisible by 3 or 5?", 14),
    ("rle", "Run-length encode 'aaabbcaaaa' as a list of [character,count] pairs.", [["a",3],["b",2],["c",1],["a",4]]),
    ("hold-sort", "Sort [12,-5,0,12,3,-1] numerically in descending order.", [12,12,3,0,-1,-5]),
    ("hold-dedupe", "Remove duplicate letters from 'cabacbd', preserving first occurrence order. Return a string.", "cabd"),
    ("hold-trace", "Python: a=[1,2]; b=a[:]; b.append(3). What is a?", [1,2]),
    ("hold-overlap", "Merge overlapping closed intervals [[5,7],[1,2],[2,4],[8,9]], and sort by start.", [[1,4],[5,7],[8,9]]),
    ("hold-division", "What is the integer value of Python's (-19) // 4?", -5),
    ("hold-count", "How many substrings equal to 'aba' occur in 'abababa', allowing overlaps?", 3),
    ("hold-fibonacci", "Given F(0)=0,F(1)=1,F(n)=F(n-1)+F(n-2), what is F(12)?", 144),
    ("hold-path", "Directed edges: A->B, B->D, A->C, C->E, E->D. Minimum edges from A to D?", 2),
    ("hold-bisect", "Return the zero-based insertion index for 6 in [2,6,6,9], after all existing equal values.", 3),
    ("hold-reverse", "Reverse the string 'z9X20' character by character.", "02X9z"),
    ("hold-inclusion", "How many integers from 1 through 40 inclusive are divisible by 4 or 6?", 13),
    ("hold-rle", "Run-length encode 'bbbaacbbb' as a list of [character,count] pairs.", [["b",3],["a",2],["c",1],["b",3]]),
]
SUITE_SHA256 = hashlib.sha256(json.dumps({"cases": CASES, "policy": EVALUATION_POLICY,
                                       "assistant_prefill": ANSWER_PREFIX}, sort_keys=True).encode()).hexdigest()


class Cancelled(Exception):
    pass


def check_cancel(event):
    if event and event.is_set():
        raise Cancelled("Evaluation cancelled")


def parse_answer(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"answer"}:
        raise ValueError("Expected exactly one JSON key: answer")
    return value["answer"]


def equal_json(actual, expected):
    # Python considers True == 1; benchmark answers must retain their JSON types.
    return json.dumps(actual, sort_keys=True, ensure_ascii=False) == json.dumps(expected, sort_keys=True, ensure_ascii=False)


def paired_pvalue(baseline, candidate):
    wins = sum(not a and b for a, b in zip(baseline, candidate))
    losses = sum(a and not b for a, b in zip(baseline, candidate))
    n = wins + losses
    p = sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n if n else 1.0
    return {"wins": wins, "losses": losses, "one_sided_sign_test_p": p}


def choose_recommendation(baseline, candidate, conformance):
    test = paired_pvalue([r["passed"] for r in baseline], [r["passed"] for r in candidate])
    # Selection is performed on a separate split. Only this single winner reaches holdout.
    recommend = bool(conformance and test["wins"] > test["losses"] and test["one_sided_sign_test_p"] <= 0.05)
    return {**test, "recommend_loop": recommend,
            "verdict": "local_holdout_improvement" if recommend else "no_demonstrated_improvement"}


def probe(worker, prompt="The capital of France is", cache=False):
    return worker.json("/completion", {"prompt": prompt, "n_predict": 12, "temperature": 0,
        "seed": 1729, "cache_prompt": cache, "return_tokens": True, "n_probs": 5, "ignore_eos": False})


def distribution_difference(a, b):
    left, right = a.get("completion_probabilities", []), b.get("completion_probabilities", [])
    if not left or len(left) != len(right):
        return None
    differences = []
    for x, y in zip(left, right):
        xp, yp = x.get("top_logprobs", []), y.get("top_logprobs", [])
        if not xp or len(xp) != len(yp):
            return None
        xmap = {p.get("id", p.get("token")): p for p in xp}
        ymap = {p.get("id", p.get("token")): p for p in yp}
        shared = xmap.keys() & ymap.keys()
        if len(shared) < max(1, len(xp)-1) or x.get("id") != y.get("id"):
            return None
        for token in shared:
            p, q = xmap[token], ymap[token]
            pv, qv = p.get("logprob"), q.get("logprob")
            if not isinstance(pv, (int,float)) or not isinstance(qv, (int,float)) or not math.isfinite(pv) or not math.isfinite(qv):
                return None
            differences.append(abs(pv-qv))
    return max(differences, default=None)


def evaluate_cases(worker, cases, cancel=None, progress=None):
    rows = []
    for index, (case_id, question, expected) in enumerate(cases):
        check_cancel(cancel)
        if progress:
            progress(f"Task {index+1}/{len(cases)}: {case_id}")
        started = time.monotonic()
        response = worker.json("/v1/chat/completions", {
            **EVALUATION_POLICY,
            "messages": [{"role": "system", "content": "Answer precisely. Return only a JSON object with one key, answer. No explanation or reasoning text."},
                         {"role": "user", "content": question},
                         {"role": "assistant", "content": ANSWER_PREFIX}],
        })
        choice = response["choices"][0]
        text = choice["message"].get("content") or ""
        row = {"id": case_id, "passed": False, "output": text, "expected": expected,
               "seconds": time.monotonic()-started, "finish_reason": choice.get("finish_reason"),
               "usage": response.get("usage", {})}
        try:
            answer = parse_answer(text)
            row["passed"] = equal_json(answer, expected) and choice.get("finish_reason") != "length"
        except (ValueError, TypeError):
            row["format_error"] = True
        rows.append(row)
    return rows


def test_model(worker, model, state, passes=(2,3,4), quick=False, cancel=None, progress=None):
    # Refuse a competing run before it can overwrite another run's report.
    if model.eligible:
        worker.claim()
    report = {"schema": 1, "model": model.public(), "started": time.time(), "status": "running",
              "suite": SUITE_VERSION, "suite_sha256": SUITE_SHA256, "quick": quick,
              "evaluation_policy": {**EVALUATION_POLICY, "assistant_prefill": ANSWER_PREFIX},
              "paper_reproduction": False, "recommended_profile": "baseline", "profiles": [],
              "limitations": ["Quantized checkpoints and a small local diagnostic suite differ from the paper's BF16 benchmarks.",
                              "Held-out tasks are public diagnostics, not a general capability guarantee.",
                              "Prompt-prefix reuse is disabled by default; a cache probe is reported separately."],
              "storage_before": storage_snapshot(state)}
    target = state / "reports" / f"{model.id}.json"
    def save():
        atomic_json(target, report)
    if not model.eligible:
        report.update(status="unsupported", reasons=model.reasons)
        save()
        return report
    selection = CASES[:4] if quick else CASES[:12]
    holdout = CASES[12:16] if quick else CASES[12:]
    try:
        report["runtime"] = worker.build_info()
        if progress:
            progress(f"{model.name}: baseline")
        with worker.use(model, Profile.for_model(model)):
            base_probe = probe(worker)
            base_selection = evaluate_cases(worker, selection, cancel, progress)
            base_holdout = evaluate_cases(worker, holdout, cancel, progress)
        report["baseline"] = {"probe": base_probe, "selection": base_selection, "holdout": base_holdout}
        save()
        for count in passes:
            check_cancel(cancel)
            if progress:
                progress(f"{model.name}: {count} passes, anchor identity check")
            identity_profile = Profile.for_model(model, count, anchor=1.0)
            with worker.use(model, identity_profile):
                identity = probe(worker)
            delta = distribution_difference(base_probe, identity)
            identity_ok = (base_probe.get("tokens") == identity.get("tokens") and bool(base_probe.get("tokens"))
                           and delta is not None and delta < 0.005)
            entry = {"profile": asdict(Profile.for_model(model, count)), "passes": count,
                     "anchor_identity": {"passed": identity_ok, "max_logprob_error": delta},
                     "selection": [], "status": "conformance_failed"}
            if identity_ok:
                profile = Profile.for_model(model, count)
                with worker.use(model, profile):
                    uncached = probe(worker, cache=False)
                    cached = probe(worker, cache=True)
                    entry["cache_probe"] = {"same_tokens": uncached.get("tokens") == cached.get("tokens"),
                        "tokens_cached": cached.get("tokens_cached", 0),
                        "max_logprob_error": distribution_difference(uncached, cached),
                        "enabled_by_default": False}
                    entry["selection"] = evaluate_cases(worker, selection, cancel, progress)
                    entry["status"] = "tested"
            report["profiles"].append(entry)
            save()
        valid = [p for p in report["profiles"] if p["status"] == "tested"]
        if valid:
            winner = max(valid, key=lambda p: (sum(r["passed"] for r in p["selection"]), -p["passes"]))
            with worker.use(model, Profile.for_model(model, winner["passes"])):
                winner["holdout"] = evaluate_cases(worker, holdout, cancel, progress)
            recommendation = choose_recommendation(base_holdout, winner["holdout"], winner["anchor_identity"]["passed"])
            # Quick mode can screen compatibility but never promotes a profile.
            if quick:
                recommendation.update(recommend_loop=False, verdict="quick_screen_only")
            report["recommendation"] = recommendation
            if recommendation["recommend_loop"]:
                report["recommended_profile"] = f"loop{winner['passes']}"
        report["status"] = "tested" if valid else "conformance_failed"
    except Cancelled:
        report["status"] = "cancelled"
    except KeyboardInterrupt:
        report["status"] = "cancelled"
        raise
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=f"{type(exc).__name__}: {exc}")
    finally:
        worker.close()
        report["finished"] = time.time()
        report["storage_after"] = storage_snapshot(state)
        save()
    return report


def public_summary(report):
    """Explicit allowlist: never export local paths, tokens, logs, or conversations."""
    result = {k: report[k] for k in ("schema", "suite", "suite_sha256", "quick", "status", "error_type", "evaluation_policy",
              "paper_reproduction", "recommended_profile", "recommendation", "limitations") if k in report}
    if "model" in report:
        result["model"] = {k: report["model"][k] for k in ("id", "name", "architecture", "layers", "size_bytes",
                           "header_sha256", "eligible", "reasons", "mode") if k in report["model"]}
    if "runtime" in report:
        result["runtime"] = {k: report["runtime"][k] for k in ("llama_cpp_revision", "patch_sha256", "method",
                             "architectures", "cache", "moe_routing", "version", "backend", "execution_policy")
                             if k in report["runtime"]}
    def metrics(rows):
        return {"passed": sum(r["passed"] for r in rows), "total": len(rows),
                "truncated": sum(r.get("finish_reason") == "length" for r in rows),
                "format_errors": sum(bool(r.get("format_error")) for r in rows),
                "seconds": sum(r.get("seconds", 0) for r in rows),
                "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in rows)}
    result["profiles"] = [{"passes": p["passes"], "status": p["status"],
                            "anchor_identity": p["anchor_identity"], "cache_probe": p.get("cache_probe"),
                            "selection_passed": sum(r["passed"] for r in p.get("selection", [])),
                            "selection_total": len(p.get("selection", [])),
                            "holdout_passed": sum(r["passed"] for r in p.get("holdout", [])),
                            "holdout_total": len(p.get("holdout", [])),
                            "selection_metrics": metrics(p.get("selection", [])),
                            "holdout_metrics": metrics(p.get("holdout", []))} for p in report.get("profiles", [])]
    if "baseline" in report:
        result["baseline"] = {split: metrics(report["baseline"][split]) for split in ("selection", "holdout")}
    return result
