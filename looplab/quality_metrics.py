"""Paired summaries for an independent, fixed-question model comparison."""
# SPDX-License-Identifier: AGPL-3.0-or-later
import math
from decimal import Decimal
import random
import re
import statistics


def numeric_answer(value):
    """Normalize one extracted decimal, without searching surrounding prose."""
    value = str(value).strip()
    if not re.fullmatch(r'\$?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\.?', value):
        return None
    return Decimal(value.removeprefix('$').replace(',', '').removesuffix('.'))


def sample_scores(result):
    scores = {}
    for task, rows in result['samples'].items():
        for row in rows:
            if row['filter'] != 'flexible-extract':
                continue
            key = (task, row['doc_id'])
            if key in scores:
                raise ValueError('Duplicate benchmark question')
            value = row['exact_match']
            if value not in (0, 1, False, True):
                raise ValueError('Expected binary exact-match scores')
            official = bool(value)
            if task == 'gsm8k_cot_zeroshot':
                # The harness removes decimal points rather than normalizing
                # numbers: "$7.00." is incorrectly unequal to "7". Keep its
                # extraction and report its score separately, but compare the
                # extracted number with the dataset's marked final answer.
                extracted = row['filtered_resps']
                if len(extracted) != 1 or '####' not in row['target']:
                    raise ValueError('Unexpected GSM8K answer format')
                predicted = numeric_answer(extracted[0])
                expected = numeric_answer(row['target'].rsplit('####', 1)[1])
                if expected is None:
                    raise ValueError('Invalid GSM8K numeric target')
                value = predicted is not None and predicted == expected
            scores[key] = {'correct': bool(value), 'doc_hash': row['doc_hash'],
                           'official_correct': official, 'target_hash': row['target_hash'],
                           'prompt_hash': row.get('prompt_hash')}
    if not scores:
        raise ValueError('No scored benchmark questions')
    return scores


def paired_comparison(baseline, candidate, seed=1729, bootstrap=4000):
    if baseline.keys() != candidate.keys():
        raise ValueError('Paired comparisons require identical question IDs')
    groups = {}
    wins = losses = 0
    for key, left in baseline.items():
        right = candidate[key]
        if any(left.get(k) != right.get(k) for k in ('doc_hash', 'target_hash', 'prompt_hash')):
            raise ValueError('Paired question or target changed')
        difference = int(right['correct']) - int(left['correct'])
        groups.setdefault(key[0], []).append(difference)
        wins += difference == 1
        losses += difference == -1
    discordant = wins + losses
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant)
    rng = random.Random(seed)
    n = len(baseline)
    changes = sorted(100 * sum(sum(rng.choices(group, k=len(group))) for group in groups.values()) / n
                     for _ in range(bootstrap))
    return {'questions': n, 'wins': wins, 'losses': losses,
            'accuracy_change_points': 100 * (wins - losses) / n,
            'paired_bootstrap_95_percent_interval': [changes[int(.025 * bootstrap)], changes[int(.975 * bootstrap)]],
            'two_sided_p': p}


def holm_adjust(pvalues):
    """Holm family-wise correction across all declared loop/model comparisons."""
    adjusted = [None] * len(pvalues)
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        previous = max(previous, min(1.0, pvalues[index] * (len(pvalues) - rank)))
        adjusted[index] = previous
    return adjusted


def profile_metrics(result, records):
    scores = sample_scores(result)
    by_task = {}
    for (task, _), row in scores.items():
        counter = by_task.setdefault(task, {'correct': 0, 'questions': 0})
        counter['questions'] += 1
        counter['correct'] += row['correct']
    if len(records) != len(scores):
        raise ValueError('Response records and scored questions differ')
    times = [r['seconds'] for r in records]
    completion = sum(r['response'].get('usage', {}).get('completion_tokens', 0) for r in records)
    native = [r['response'].get('timings', {}) for r in records]
    decode_ms = sum(t.get('predicted_ms', 0) for t in native)
    return {'questions': len(scores), 'correct': sum(r['correct'] for r in scores.values()),
            'accuracy': sum(r['correct'] for r in scores.values()) / len(scores), 'tasks': by_task,
            'official_correct': sum(r['official_correct'] for r in scores.values()),
            'numeric_normalization_changes': sum(r['official_correct'] != r['correct'] for r in scores.values()),
            'request_seconds': sum(times), 'mean_request_seconds': statistics.mean(times),
            'median_request_seconds': statistics.median(times), 'completion_tokens': completion,
            'end_to_end_output_tokens_per_second': completion / sum(times) if sum(times) else None,
            'native_decode_tokens_per_second': (1000 * sum(t.get('predicted_n', 0) for t in native) / decode_ms
                                                if decode_ms and all('predicted_n' in t and 'predicted_ms' in t
                                                                     for t in native) else None),
            'cached_prompt_tokens': sum(r['response'].get('usage', {}).get('prompt_tokens_details', {})
                                       .get('cached_tokens', 0) for r in records),
            'truncated': sum(r['response']['choices'][0].get('finish_reason') == 'length' for r in records),
            'empty_final_answer': sum(not r['response']['choices'][0]['message'].get('content') for r in records)}
