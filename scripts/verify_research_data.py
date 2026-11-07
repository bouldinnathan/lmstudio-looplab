#!/usr/bin/env python3
"""Offline integrity and score/timing audit of the published research records.

Only the Python standard library and this repository are needed; no inference,
datasets, model weights, evaluation packages or network connection are required.
Harness extraction labels are retained, not independently re-extracted here.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looplab.benchmark import equal_json, parse_answer, public_summary
from looplab.quality_metrics import holm_adjust, paired_comparison, profile_metrics, sample_scores
from scripts.export_research_data import RUNS, digest, performance_rows, read


def same(left, right, label):
    if isinstance(left, float) or isinstance(right, float):
        valid = isinstance(left, (float, int)) and isinstance(right, (float, int)) and math.isclose(left, right, abs_tol=1e-8, rel_tol=1e-10)
    else:
        valid = left == right
    if not valid:
        raise ValueError(f'Mismatch: {label}')


def verify(reports):
    data = reports / 'research-data'
    manifest = read(data / 'manifest.json')
    for name, expected in manifest['files'].items():
        path = (data / name).resolve()
        if not path.is_relative_to(data.resolve()):
            raise ValueError('Manifest path escapes research directory')
        content = path.read_bytes()
        same(len(content), expected['bytes'], name + ' length')
        same(hashlib.sha256(content).hexdigest(), expected['sha256'], name + ' SHA256')

    with (data / 'question-metrics.csv').open(newline='') as f:
        question_rows = list(csv.DictReader(f))
    question_index = {(r['study'], r['model_id'], r['profile'], r['task'], int(r['doc_id'])): r for r in question_rows}
    same(len(question_index), len(question_rows), 'unique question CSV keys')
    n_requests = 0
    main_outcomes = {}
    for run in RUNS:
        comparison = read(data / run / 'comparison.json')
        protocol = read(data / run / 'protocol.json')
        same(comparison['protocol'], protocol, run + ' protocol')
        comparisons = []
        for model in comparison['models']:
            ident = model['model']['id']
            scores = {}
            for profile in model['profiles']:
                passes = profile['passes']
                profile_name = 'baseline' if passes == 1 else 'loop' + str(passes)
                folder = data / run / ident / profile_name
                stamp = read(folder / 'manifest.json')
                same(stamp['protocol_sha256'], digest(protocol), 'protocol fingerprint')
                same(stamp['profile'], profile['profile'], 'loop settings')
                same(stamp['model_header_sha256'], model['model']['header_sha256'], 'checkpoint header fingerprint')
                harness = read(folder / 'harness.json')
                entries = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
                records = [r['record'] for r in entries]
                scores[passes] = sample_scores(harness)
                same(len(records), protocol['questions_per_profile'], 'request count')
                same({(e['task'], e['doc_id']) for e in entries}, set(scores[passes]), 'request/sample pairing')
                for task, ids in protocol['sample_ids'].items():
                    same({e['doc_id'] for e in entries if e['task'] == task}, set(ids), 'fixed sample indices')
                by_question = {}
                for task, samples in harness['samples'].items():
                    for sample in samples:
                        by_question.setdefault((task, sample['doc_id']), []).append(sample)
                for entry in entries:
                    key = entry['task'], entry['doc_id']
                    record = entry['record']
                    payload = record['payload']
                    same(digest(payload), entry['request_sha256'], 'request payload fingerprint')
                    same(payload['max_tokens'], protocol['max_tokens'], 'token budget')
                    same(payload['cache_prompt'], False, 'cache disabled')
                    choice = record['response']['choices'][0]
                    # The original scorer blanks truncated answers; their complete
                    # partial output is still present in record.response.
                    graded_text = '' if choice['finish_reason'] == 'length' else choice['message'].get('content') or ''
                    for sample in by_question[key]:
                        same(sample['resps'], [[graded_text]], 'scorer input versus raw response')
                        same(json.loads(sample['arguments'][0][0][0]), payload['messages'], 'saved prompt')
                    score = scores[passes][key]
                    row = question_index[run, ident, profile_name, *key]
                    same(int(row['correct']), int(score['official_correct'] if run == RUNS[1] else score['correct']), 'reported per-question score')
                    same(int(row['decimal_normalized_correct']), int(score['correct']), 'normalized per-question score')
                    same(int(row['official_correct']), int(score['official_correct']), 'official per-question score')
                    same(float(row['seconds']), record['seconds'], 'per-question timing')
                    same(row['request_sha256'], entry['request_sha256'], 'CSV request link')
                    if run == RUNS[0]:
                        main_outcomes[ident, *key, passes] = score
                measured = profile_metrics(harness, records)
                summary = read(folder / 'summary.json')
                for key, value in measured.items():
                    if run == RUNS[1] and key in ('correct', 'accuracy', 'tasks', 'numeric_normalization_changes'):
                        continue  # The pilot used the unnormalized official scorer.
                    if key in summary:
                        same(value, summary[key], run + ' ' + key)
                    if key in profile:
                        same(value, profile[key], run + ' comparison ' + key)
                if run == RUNS[1]:
                    same(profile['correct'], measured['official_correct'], 'historical pilot score')
                n_requests += len(records)
            if run == RUNS[0]:
                baseline = next(p for p in model['profiles'] if p['passes'] == 1)
                for profile in model['profiles']:
                    if profile['passes'] == 1:
                        continue
                    paired = paired_comparison(scores[1], scores[profile['passes']])
                    for key, value in paired.items():
                        same(value, profile['comparison'][key], 'paired statistic ' + key)
                    same(profile['request_seconds'] / baseline['request_seconds'],
                         profile['comparison']['request_time_ratio'], 'relative time cost')
                    comparisons.append(profile['comparison'])
        if comparisons:
            corrected = holm_adjust([c['two_sided_p'] for c in comparisons])
            for row, p in zip(comparisons, corrected):
                same(row['holm_adjusted_p'], p, 'Holm adjusted p')

    public = read(reports / 'qwen-public-benchmark.json')
    same(public['models'], read(data / RUNS[0] / 'comparison.json')['models'], 'public comparison')
    for row in public['question_outcomes']:
        for passes, correct in row['correct_by_passes'].items():
            score = main_outcomes[row['model_id'], row['task'], row['dataset_index'], int(passes)]
            same(score['correct'], correct, 'published question outcome')
            same(score['doc_hash'], row['question_sha256'], 'published question hash')

    case_data = read(data / 'diagnostic-cases.json')
    cases = {case[0]: case for case in case_data['cases']}
    diagnostic_answers = 0
    for path in sorted((data / 'diagnostics').glob('*.json')):
        report = read(path)
        if report.get('status') != 'tested':
            continue
        same(report['suite_sha256'], case_data['suite_sha256'], 'diagnostic cases/policy')
        groups = [report['baseline'][s] for s in ('selection', 'holdout')]
        groups += [p.get(s, []) for p in report['profiles'] for s in ('selection', 'holdout')]
        for group in groups:
            for row in group:
                same(row['expected'], cases[row['id']][2], 'diagnostic target')
                try:
                    passed = equal_json(parse_answer(row['output']), row['expected']) and row['finish_reason'] != 'length'
                except (ValueError, TypeError):
                    passed = False
                same(passed, row['passed'], 'diagnostic regrade')
                diagnostic_answers += 1
        summary = public_summary(report)
        name = 'qwen-recovery-quality' if report['quick'] else report['model']['name']
        saved = read(reports / (name + '.json'))
        same(summary['baseline'], saved['baseline'], 'diagnostic baseline summary')
        for left, right in zip(summary['profiles'], saved['profiles']):
            for split in ('selection', 'holdout'):
                same(left[split + '_metrics'], right[split + '_metrics'], 'diagnostic profile summary')
    same(diagnostic_answers, manifest['counts']['diagnostic_answers'], 'diagnostic answer count')
    same(n_requests, len(question_rows), 'question CSV coverage')
    with (data / 'performance.csv').open(newline='') as f:
        actual = list(csv.DictReader(f))
    expected = [{k: str(v) for k, v in row.items()} for row in performance_rows(reports)]
    same(actual, expected, 'chart source data')
    with (data / 'model-coverage.csv').open(newline='') as f:
        coverage = list(csv.DictReader(f))
    same(len(coverage), 54, 'inventoried model coverage')
    for row in coverage:
        saved = read(data / row['compatibility_record'])
        same(saved['model']['id'], row['model_id'], 'compatibility identity')
        same(saved['status'], row['compatibility_status'], 'compatibility status')
    result = {'verified_files': len(manifest['files']), 'main_responses': 1600,
              'pilot_responses': 8, 'diagnostic_answers_regraded': diagnostic_answers,
              'model_files_covered': len(coverage), 'performance_rows': len(actual)}
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=ROOT / 'reports')
    verify(parser.parse_args().reports)
