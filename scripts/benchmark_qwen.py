#!/usr/bin/env python3
"""Compare existing Qwen checkpoints with loops using lm-eval's public tasks.

Optional dependency: lm-eval[api]==0.4.13, installed in a separate environment.
No generated code is executed. Task data and exact responses stay in local state.
This is a bounded zero-shot comparison, not a reproduction of published scores.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HOME', str(ROOT / '.runtime/eval-cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

from looplab.models import atomic_json, state_directory, storage_snapshot
from looplab.quality_metrics import holm_adjust, paired_comparison, profile_metrics, sample_scores
from looplab.server import App

VERSION = 'looplab-lm-eval-comparison-v1'
HARNESS_VERSION = '0.4.13'
DATASETS = {'openai/gsm8k': '740312add88f781978c0658806c59bc2815b9866',
            'SaylorTwift/bbh': 'b5306be6f827cfafbb545ff5a51f96916029b0fd'}
TASKS = ['gsm8k_cot_zeroshot', 'bbh_cot_zeroshot_boolean_expressions',
         'bbh_cot_zeroshot_logical_deduction_five_objects']
SEED = 1729


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def json_value(value):
    # Harness output contains JsonChatStr, NumPy scalars and task functions.
    if hasattr(value, 'prompt'):
        return value.prompt
    if hasattr(value, 'item'):
        return value.item()
    if callable(value):
        return f'{value.__module__}.{value.__qualname__}'
    raise TypeError(type(value).__name__)


def harness_tasks():
    from lm_eval.tasks import TaskManager
    manager = TaskManager()
    # In this pinned harness release, inline dictionaries are complete task
    # definitions, not overrides of a registered name. Use its factory to keep
    # the official task definition while pinning the data before downloading.
    return {name: manager._factory.build(manager.task_index[name], registry=manager.task_index,
        overrides={'dataset_kwargs': {'revision': DATASETS[
            'openai/gsm8k' if name.startswith('gsm8k') else 'SaylorTwift/bbh']}}) for name in TASKS}


def backend(worker, model, directory, expected, max_tokens):
    from lm_eval.models.openai_completions import LocalChatCompletion

    class RecordedChat(LocalChatCompletion):
        def __init__(self):
            super().__init__(model=model.id, base_url=worker.base + '/v1/chat/completions',
                             tokenizer_backend=None, tokenized_requests=False,
                             max_gen_toks=max_tokens, max_length=worker.context,
                             num_concurrent=1, max_retries=1, seed=SEED)
            self.records = []

        def model_call(self, messages, *, generate=True, gen_kwargs=None, **kwargs):
            if not generate:
                raise ValueError('Only generation benchmarks are supported')
            payload = self._create_payload(self.create_message(messages), generate=True,
                                          gen_kwargs=copy.deepcopy(gen_kwargs), seed=SEED, eos=self.eos_string)
            payload.update(cache_prompt=False, stream=False)
            # Request the checkpoint's non-thinking template when it supports it.
            # CoT is still requested by the benchmark; no assistant answer is prefilled.
            payload['chat_template_kwargs'] = {'enable_thinking': False}
            key = digest(payload)
            path = directory / 'requests' / f'{key}.json'
            reused = path.exists()
            if reused:
                record = json.loads(path.read_text())
                if record['payload'] != payload:
                    raise ValueError('Cached request mismatch')
            else:
                rendered = worker.json('/apply-template', {'messages': payload['messages'],
                    'chat_template_kwargs': payload['chat_template_kwargs']})
                tokens = worker.json('/tokenize', {'content': rendered['prompt'], 'add_special': False})['tokens']
                if len(tokens) + payload['max_tokens'] > worker.context:
                    raise ValueError('Benchmark question exceeds the fixed context budget')
                start = time.monotonic()
                response = worker.json('/v1/chat/completions', payload)
                elapsed = time.monotonic() - start
                if len(response.get('choices', [])) != 1:
                    raise ValueError('Expected one generated answer')
                record = {'payload': payload, 'response': response, 'seconds': elapsed,
                          'rendered_prompt_tokens': len(tokens)}
                atomic_json(path, record)
            self.records.append(record)
            print(f'{model.name} {directory.name}: {len(self.records)}/{expected} requests'
                  f' ({"resumed" if reused else str(round(record["seconds"], 1)) + "s"})', flush=True)
            answer = copy.deepcopy(record['response'])
            # Never award a lucky number extracted from an unfinished derivation.
            if answer['choices'][0].get('finish_reason') == 'length':
                answer['choices'][0]['message']['content'] = ''
            return answer

    return RecordedChat()


def compare(app, models, destination, tasks, protocol, passes):
    from lm_eval import simple_evaluate
    protocol_sha = digest(protocol)
    outcomes = []
    for model in models:
        info = app.worker.build_info(model)
        model_result = {'model': model.public(), 'execution_mode': app.worker.execution_mode(model),
                        'runtime': {k: info[k] for k in ('llama_cpp_revision', 'patch_sha256', 'execution_policy')},
                        'profiles': []}
        raw_results = {}
        # Deterministic randomized profile order, declared in the plan.
        order = protocol['profile_order'][model.id]
        for count in order:
            if count not in passes:
                continue
            profile = app.profile(model, 'baseline' if count == 1 else f'loop{count}')
            folder = destination / model.id / ('baseline' if count == 1 else f'loop{count}')
            manifest = {'protocol_sha256': protocol_sha, 'model_header_sha256': model.header_sha256,
                        'size_bytes': model.size_bytes, 'mtime_ns': model.mtime_ns,
                        'runtime': model_result['runtime'], 'profile': asdict(profile)}
            stamp = folder / 'manifest.json'
            if stamp.exists() and json.loads(stamp.read_text()) != manifest:
                raise ValueError('Existing run has different inputs; choose another output directory')
            atomic_json(stamp, manifest)
            result_path = folder / 'harness.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                records = [json.loads(p.read_text()) for p in sorted((folder / 'requests').glob('*.json'))]
                print(f'{model.name} {folder.name}: reused complete result', flush=True)
            else:
                with app.worker.use(model, profile):
                    # Exclude model loading and a short warm-up from all timings.
                    app.worker.json('/completion', {'prompt': 'The capital of France is', 'n_predict': 8,
                        'temperature': 0, 'seed': SEED, 'cache_prompt': False})
                    adapter = backend(app.worker, model, folder, protocol['questions_per_profile'], protocol['max_tokens'])
                    result = simple_evaluate(model=adapter, tasks=list(tasks.values()),
                        samples=protocol['sample_ids'], num_fewshot=0, apply_chat_template=True,
                        gen_kwargs={'max_gen_toks': protocol['max_tokens'], 'temperature': 0, 'do_sample': False},
                        bootstrap_iters=1000, random_seed=SEED, numpy_random_seed=SEED,
                        torch_random_seed=None, fewshot_random_seed=SEED, log_samples=True)
                    if result is None:
                        raise ValueError('Evaluation returned no result')
                    result = json.loads(json.dumps(result, default=json_value))
                    records = adapter.records
                    # The serialization must succeed before marking this profile complete.
                    atomic_json(result_path, result)
            metrics = profile_metrics(result, records)
            raw_results[count] = result
            model_result['profiles'].append({'passes': count, 'profile': asdict(profile), **metrics,
                                            'official_metrics': result['results']})
            atomic_json(folder / 'summary.json', model_result['profiles'][-1])
            print(json.dumps({'model': model.name, 'passes': count, 'correct': metrics['correct'],
                              'questions': metrics['questions'], 'truncated': metrics['truncated']}), flush=True)
        if 1 in raw_results:
            original = sample_scores(raw_results[1])
            baseline = next(p for p in model_result['profiles'] if p['passes'] == 1)
            for entry in model_result['profiles']:
                if entry['passes'] != 1:
                    entry['comparison'] = paired_comparison(original, sample_scores(raw_results[entry['passes']]))
                    entry['comparison']['request_time_ratio'] = entry['request_seconds'] / baseline['request_seconds']
        model_result['profiles'].sort(key=lambda p: p['passes'])
        outcomes.append(model_result)
        write_comparison(destination, protocol, outcomes)
    return write_comparison(destination, protocol, outcomes)


def write_comparison(destination, protocol, outcomes):
    comparisons = [p['comparison'] for m in outcomes for p in m['profiles'] if 'comparison' in p]
    for row, corrected in zip(comparisons, holm_adjust([p['two_sided_p'] for p in comparisons])):
        row['holm_adjusted_p'] = corrected
    report = {'suite': VERSION, 'protocol': protocol, 'models': outcomes,
              'complete': len(outcomes) == len(protocol['model_ids']),
              'quality_recommendation_changed': False}
    atomic_json(destination / 'comparison.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', help='Local exact name or ID; default: all supported Qwen-family originals')
    parser.add_argument('--state', type=Path, default=state_directory())
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gsm8k', type=int, default=50)
    parser.add_argument('--bbh', type=int, default=25, help='Questions per BBH task')
    parser.add_argument('--max-tokens', type=int, default=768)
    parser.add_argument('--passes', type=int, nargs='+', choices=(1, 2, 3, 4), default=[1, 2, 3, 4])
    args = parser.parse_args()
    if min(args.gsm8k, args.bbh, args.max_tokens) < 1:
        parser.error('Counts must be positive')
    if importlib.metadata.version('lm_eval') != HARNESS_VERSION:
        parser.error(f'This runner requires lm-eval[api]=={HARNESS_VERSION}')
    app = App(args.state)
    destination = args.output or args.state / 'benchmarks/qwen-lm-eval-v1'
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    models, excluded = [], []
    candidates = [app.model(name) for name in args.model] if args.model else list(app.models.values())
    for model in candidates:
        if not model.architecture.startswith('qwen') or model.role != 'generation':
            continue
        try:
            if not model.eligible:
                raise ValueError('; '.join(model.reasons))
            for count in args.passes:
                app.profile(model, 'baseline' if count == 1 else f'loop{count}')
        except ValueError as exc:
            excluded.append({'id': model.id, 'name': model.name, 'reason': str(exc), 'architecture': model.architecture})
            continue
        models.append(model)
    if not models:
        raise ValueError('No Qwen checkpoints with current passing loop profiles')
    models.sort(key=lambda m: (m.name != 'qwen3-coder-30b', m.size_bytes))
    tasks = harness_tasks()
    samples = {}
    for name, task in tasks.items():
        count = args.gsm8k if name.startswith('gsm8k') else args.bbh
        samples[name] = sorted(random.Random(f'{SEED}:{name}').sample(range(len(task.test_docs())), count))
    orders = {}
    for model in models:
        order = sorted(set(args.passes))
        random.Random(f'{SEED}:{model.id}').shuffle(order)
        orders[model.id] = order
    protocol = {'version': VERSION, 'harness': 'EleutherAI/lm-evaluation-harness', 'harness_version': HARNESS_VERSION,
        'dataset_revisions': DATASETS, 'sample_ids': samples, 'questions_per_profile': sum(map(len, samples.values())),
        'max_tokens': args.max_tokens, 'context': 2048, 'temperature': 0, 'seed': SEED,
        'num_fewshot': 0, 'chat_template_kwargs': {'enable_thinking': False}, 'prefix_cache': False,
        'truncated_answers_scored_incorrect': True, 'primary_filter': 'flexible-extract',
        'gsm8k_scoring': 'exact decimal equality after official extraction; currency, commas and final punctuation normalized',
        'sampling': 'fixed stratified random subset, identical for every profile and model',
        'profile_order': orders, 'model_ids': [m.id for m in models],
        'paper_reproduction': False, 'official_full_benchmark': False}
    plan = destination / 'protocol.json'
    if plan.exists() and json.loads(plan.read_text()) != protocol:
        raise ValueError('Output already holds another protocol; choose another output directory')
    atomic_json(plan, protocol)
    atomic_json(destination / 'excluded.json', excluded)
    if not (destination / 'storage-before.json').exists():
        atomic_json(destination / 'storage-before.json', storage_snapshot(args.state))
    try:
        compare(app, models, destination, tasks, protocol, args.passes)
    finally:
        app.worker.close()
        atomic_json(destination / 'storage-after.json', storage_snapshot(args.state))


if __name__ == '__main__':
    main()
