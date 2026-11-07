#!/usr/bin/env python3
"""Export retained experiment records, never the whole private state directory.

This is an explicit publishing step, separate from model evaluation. It makes no
network requests, performs no inference and copies no model weights.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from looplab.benchmark import ANSWER_PREFIX, CASES, EVALUATION_POLICY, SUITE_SHA256
from looplab.quality_metrics import sample_scores

RUNS = ('qwen-lm-eval-v1', 'qwen-lm-eval-pilot')
DIAGNOSTICS = ('deepcoder-14b', 'deepseek-r1-32b', 'devstral-24b',
               'qwen3-coder-30b', 'reader-lm-1.5b', 'qwen-recovery-quality')
SECRET_KEY = re.compile(r'^(api[_-]?key|authorization|password|secret|cookie|access[_-]?token)$', re.I)


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def joined_requests(folder, harness):
    """Join on the actual messages, then retain the original payload hash too."""
    by_prompt = {}
    for task, samples in harness['samples'].items():
        for sample in samples:
            if sample['filter'] != 'flexible-extract':
                continue
            messages = json.loads(sample['arguments'][0][0][0])
            key = digest(messages)
            if key in by_prompt:
                raise ValueError('Ambiguous prompt-to-question join')
            by_prompt[key] = (task, sample['doc_id'])
    joined = []
    seen = set()
    for path in sorted((folder / 'requests').glob('*.json')):
        record = read(path)
        if digest(record['payload']) != path.stem:
            raise ValueError(f'Request hash mismatch: {path.name}')
        task, doc_id = by_prompt[digest(record['payload']['messages'])]
        if (task, doc_id) in seen:
            raise ValueError('Duplicate request for a scored question')
        seen.add((task, doc_id))
        joined.append({'request_sha256': path.stem, 'task': task, 'doc_id': doc_id,
                       'record': record})
    if len(seen) != len(by_prompt):
        raise ValueError('Missing request for a scored question')
    return sorted(joined, key=lambda r: (r['task'], r['doc_id']))


class Export:
    def __init__(self, destination, state):
        self.destination = destination
        self.state = state
        self.files = set()
        self.redactions = []
        token_file = state / 'token'
        self.token = token_file.read_text().strip() if token_file.exists() else ''

    def clean(self, value, filename, pointer=''):
        # Fail closed on unexpected secrets/paths rather than publish them or
        # silently edit generated answers. The sole known private field is the
        # installed harness task's absolute config_source path.
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if SECRET_KEY.fullmatch(key):
                    raise ValueError(f'Unexpected credential field in {filename}')
                result[key] = self.clean(item, filename, pointer + '/' + key)
            return result
        if isinstance(value, list):
            return [self.clean(v, filename, pointer + '/' + str(i)) for i, v in enumerate(value)]
        if isinstance(value, str):
            if self.token and self.token in value:
                raise ValueError(f'Unexpected private token in {filename}')
            if pointer.endswith('/metadata/config_source') and '/site-packages/lm_eval/' in value:
                value = 'lm_eval/' + value.split('/site-packages/lm_eval/', 1)[1]
                self.redactions.append({'file': filename, 'json_pointer': pointer,
                                        'change': 'Removed local installation directory; kept package-relative task path.'})
            if re.search(r'/(?:home|Users|mnt|media|tmp|run/user)/', value):
                raise ValueError(f'Unexpected local path in {filename} at {pointer}')
        return value

    def json(self, name, value):
        value = self.clean(value, name)
        path = self.destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        self.files.add(name)

    def jsonl(self, name, rows):
        path = self.destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(self.clean(row, name, '/' + str(i)), ensure_ascii=False, allow_nan=False)
                                + '\n' for i, row in enumerate(rows)))
        self.files.add(name)

    def csv(self, name, rows):
        if not rows:
            raise ValueError(f'No rows for {name}')
        with (self.destination / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        self.files.add(name)


def performance_rows(reports):
    rows = []
    for run in RUNS:
        comparison = read(reports / 'research-data' / run / 'comparison.json')
        for model in comparison['models']:
            for profile in model['profiles']:
                rows.append({'study': run, 'model_id': model['model']['id'],
                    'model': model['model']['name'], 'split': 'test', 'passes': profile['passes'],
                    'correct': profile['correct'], 'questions': profile['questions'],
                    'request_seconds': profile['request_seconds'], 'truncated': profile['truncated'],
                    'raw_answers': 'available',
                    'source': f"{run}/{model['model']['id']}/{'baseline' if profile['passes'] == 1 else 'loop' + str(profile['passes'])}/summary.json"})
    for name in DIAGNOSTICS:
        data = read(reports / f'{name}.json')
        for split in ('selection', 'holdout'):
            candidates = [(1, data['baseline'][split])] + [(p['passes'], p[split + '_metrics']) for p in data['profiles']]
            for passes, metrics in candidates:
                if not metrics['total']:
                    continue  # Untested holdout profiles are missing, never zero scores.
                rows.append({'study': data['suite'] + ('-quick' if data['quick'] else ''),
                    'model_id': data['model']['id'], 'model': data['model']['name'],
                    'split': split, 'passes': passes, 'correct': metrics['passed'],
                    'questions': metrics['total'], 'request_seconds': metrics['seconds'],
                    'truncated': metrics['truncated'],
                    'raw_answers': 'summary_only' if name == 'qwen3-coder-30b' else 'available',
                    'source': '../' + name + '.json'})
    return rows


def export(state, reports):
    destination = reports / 'research-data'
    destination.mkdir(parents=True, exist_ok=True)
    out = Export(destination, state)
    counts = {}
    question_rows = []
    for run in RUNS:
        source = state / 'benchmarks' / run
        for path in sorted(source.glob('*.json')):
            out.json(f'{run}/{path.name}', read(path))
        count = 0
        for folder in sorted(p.parent for p in source.glob('*/*/harness.json')):
            rel = f'{run}/{folder.parent.name}/{folder.name}'
            harness = read(folder / 'harness.json')
            records = joined_requests(folder, harness)
            for path in sorted(folder.glob('*.json')):
                out.json(f'{rel}/{path.name}', read(path))
            out.jsonl(f'{rel}/requests.jsonl', records)
            scores = sample_scores(harness)
            for item in records:
                record = item['record']
                score = scores[item['task'], item['doc_id']]
                choice = record['response']['choices'][0]
                question_rows.append({'study': run, 'model_id': folder.parent.name,
                    'profile': folder.name, 'task': item['task'], 'doc_id': item['doc_id'],
                    'request_sha256': item['request_sha256'],
                    'correct': int(score['official_correct'] if run == RUNS[1] else score['correct']),
                    'decimal_normalized_correct': int(score['correct']),
                    'official_correct': int(score['official_correct']),
                    'seconds': record['seconds'], 'finish_reason': choice.get('finish_reason'),
                    'completion_tokens': record['response'].get('usage', {}).get('completion_tokens'),
                    'rendered_prompt_tokens': record['rendered_prompt_tokens'],
                    'record_file': f'{rel}/requests.jsonl'})
            count += len(records)
        counts[run] = count
    # Pilot predates decimal normalization: its comparison.json remains 3/4.
    # The CSV retains each run's reported score and labels the normalized score separately.
    out.csv('question-metrics.csv', question_rows)

    diagnostic_answers = 0
    for source_name, target_name in [('reports', 'diagnostics'), ('exploration', 'exploration'), ('repairs', 'repairs')]:
        paths = sorted((state / source_name).glob('*.json'))
        for path in paths:
            data = read(path)
            out.json(f'{target_name}/{path.name}', data)
            if source_name == 'reports' and data.get('status') == 'tested':
                if data['suite_sha256'] != SUITE_SHA256:
                    raise ValueError('Retained diagnostic uses different case definitions')
                diagnostic_answers += sum(len(data['baseline'][s]) for s in ('selection', 'holdout'))
                diagnostic_answers += sum(len(p.get(s, [])) for p in data['profiles'] for s in ('selection', 'holdout'))
        counts[target_name + '_files'] = len(paths)
    counts['diagnostic_answers'] = diagnostic_answers
    out.json('diagnostic-cases.json', {'suite_sha256': SUITE_SHA256, 'cases': CASES,
        'evaluation_policy': EVALUATION_POLICY, 'assistant_prefill': ANSWER_PREFIX,
        'system_message': 'Answer precisely. Return only a JSON object with one key, answer. No explanation or reasoning text.',
        'provenance': 'Case definitions exported from source; suite hash matches all five retained tested diagnostic records. These are not saved HTTP request bodies.'})
    out.json('additional/precision-recovery.json', read(reports / 'local/precision-recovery.json'))
    out.csv('performance.csv', performance_rows(reports))

    quality_ids = {read(reports / (name + '.json'))['model']['id'] for name in DIAGNOSTICS}
    main_ids = set(read(destination / RUNS[0] / 'protocol.json')['model_ids'])
    coverage = []
    for model in read(reports / 'all-local-survey.json')['models']:
        metadata = model['model']
        ident = metadata['id']
        coverage.append({'model_id': ident, 'model': metadata['name'], 'architecture': metadata['architecture'],
            'compatibility_status': model['status'], 'main_100_question_comparison': ident in main_ids,
            'small_quality_diagnostic': ident in quality_ids,
            'quality_raw_status': ('main_and_quick_available_initial_v2_summary_only' if ident.startswith('qwen3-coder-30b-0')
                                   else 'available' if ident in quality_ids else 'not_quality_tested'),
            'compatibility_record': f'exploration/{ident}.json'})
    out.csv('model-coverage.csv', coverage)
    out.json('provenance.json', {
        'schema': 1, 'source_repository': 'https://github.com/bouldinnathan/lmstudio-looplab',
        'repository_head_before_export': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'generated_by_this_project': ['Model completions and reasoning_content when returned',
            'Request latency, native timings, token usage, finish reasons',
            'Correctness labels, paired comparisons, compatibility and numerical checks'],
        'third_party_inputs': {
            'openai/gsm8k': {'revision': '740312add88f781978c0658806c59bc2815b9866',
                'url': 'https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866',
                'upstream': 'https://github.com/openai/grade-school-math', 'license': 'MIT',
                'license_source': 'https://raw.githubusercontent.com/openai/grade-school-math/master/LICENSE'},
            'SaylorTwift/bbh': {'revision': 'b5306be6f827cfafbb545ff5a51f96916029b0fd',
                'url': 'https://huggingface.co/datasets/SaylorTwift/bbh/tree/b5306be6f827cfafbb545ff5a51f96916029b0fd',
                'upstream': 'https://github.com/suzgunmirac/BIG-Bench-Hard', 'license': 'MIT',
                'license_source': 'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/LICENSE'},
            'lm-evaluation-harness': {'version': '0.4.13',
                'url': 'https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.13', 'license': 'MIT',
                'license_source': 'https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/v0.4.13/LICENSE.md'}},
        'input_output_distinction': 'record.payload is benchmark input; record.response is generated output. harness.samples.doc and target are upstream benchmark data; filtered_resps and exact_match are derived grading.',
        'storage_format': 'Original request JSON objects are losslessly grouped into requests.jsonl and joined to task/doc_id. Other retained JSON values are unchanged except the documented path redactions.',
        'missing_data': [
            'Initial Qwen3-Coder diagnostic-v2 per-case outputs were overwritten by a diagnostic-v3 quick run. Its original scores and timings survive in ../qwen3-coder-30b.json; they are not reconstructed responses.',
            'Small diagnostics retained final-answer text, timing and usage, not full HTTP responses or separate reasoning_content.',
            'Warm-up responses, exact rendered chat-template token IDs, and most compatibility-probe token distributions were not retained. Available original diagnostic baseline probes are included.',
            'Rotating worker/service logs are excluded because they can mix benchmark traffic with interactive use. No claim of complete historical logs.',
            'Model identities use recorded size and GGUF-header hash, not a full weight hash. Full weight hashes were not recorded.'],
        'excluded_nonresearch_material': ['Credentials', 'User chats', 'Model weights', 'Model caches',
            'Private worker logs', 'Core dumps', 'Installed executables'],
        'counts': counts})
    for name in ('gsm8k-MIT.txt', 'bbh-MIT.txt', 'lm-evaluation-harness-MIT.txt'):
        path = destination / 'upstream-licenses' / name
        if not path.is_file():
            raise ValueError('Missing upstream license: ' + name)
        out.files.add('upstream-licenses/' + name)
    manifest = {'schema': 1, 'counts': counts, 'redactions': out.redactions,
                'files': {name: {'sha256': hashlib.sha256((destination / name).read_bytes()).hexdigest(),
                                 'bytes': (destination / name).stat().st_size} for name in sorted(out.files)}}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'counts': counts, 'files': len(out.files),
                      'bytes': sum(f['bytes'] for f in manifest['files'].values()),
                      'path_redactions': len(out.redactions)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=Path.home() / '.local/state/looplab')
    parser.add_argument('--reports', type=Path, default=ROOT / 'reports')
    args = parser.parse_args()
    export(args.state, args.reports)
