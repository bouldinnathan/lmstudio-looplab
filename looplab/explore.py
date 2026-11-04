"""Best-effort local compatibility survey with repair handoffs, not quality claims."""
# SPDX-License-Identifier: AGPL-3.0-or-later
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import tempfile
import time

from loop_gguf import GGUF
from .benchmark import Cancelled, check_cancel, distribution_difference, probe
from .models import Profile, atomic_json, resource_reason, storage_snapshot

VERSION = 'looplab-explore-v1'
RECOVERY_VERSION = 'looplab-precision-recovery-v1'
EFFECT_VERSION = 'shared-token-probability-v1'


def clean_error(text, model, worker):
    text = str(text).replace(model.path, '<model.gguf>').replace(worker.token, '<redacted>')
    text = text.replace(str(Path.home()), '<home>')
    # Diagnostics can contain absolute runtime/library paths; keep source basenames only.
    return re.sub(r'(?<![\w<])/(?:[^\s:]+/)+([^\s:]+)', r'<path>/\1', text)[-6000:]


def failure(exc, model, worker):
    details = clean_error(f'{type(exc).__name__}: {exc}', model, worker)
    path = getattr(worker, 'log_path', None)
    if path and Path(path).exists():
        # Startup failures otherwise only report the log filename.
        lines = Path(path).read_text(errors='replace').splitlines()[-100:]
        matching = [line for line in lines if re.search(r'error|fail|exception|assert|unknown|unsupported|missing', line, re.I)]
        details += '\n' + clean_error('\n'.join(matching[-12:]), model, worker)
    return {'error_type': type(exc).__name__, 'diagnostic': details.strip()}


def finite_probe(value):
    probabilities = value.get('completion_probabilities', [])
    return bool(value.get('tokens')) and bool(probabilities) and all(
        row.get('top_logprobs') and all(isinstance(p.get('logprob'), (float, int)) and math.isfinite(p['logprob'])
        for p in row['top_logprobs']) for row in probabilities)


def distribution_changed(baseline, looped):
    """Find a measured loop effect even when the top-k token set changes.

    The strict comparator remains mandatory for identity and repeatability.
    Here, one shared token with a changed probability establishes an effect.
    """
    if baseline.get('tokens') != looped.get('tokens'):
        return True
    for a, b in zip(baseline.get('completion_probabilities', []), looped.get('completion_probabilities', [])):
        left = {p['id']: p['logprob'] for p in a['top_logprobs']}
        right = {p['id']: p['logprob'] for p in b['top_logprobs']}
        if any(abs(left[token] - right[token]) > 1e-6 for token in left.keys() & right.keys()):
            return True
    return False


def repair_handoff(model, report):
    gguf = GGUF(model.path)
    # No tokenizer strings, templates, generated text, credentials, or tensor data.
    metadata = {key: value for key, value in gguf.values.items()
                if key.startswith(model.architecture + '.') and isinstance(value, (int, float, bool))}
    representative = [t for t in gguf.tensors if not t.name.startswith('blk.') or t.name.startswith('blk.0.')]
    return {
        'model_id': model.id, 'architecture': model.architecture, 'status': report['status'],
        'adapter': model.adapter or report.get('runtime', {}).get('adapter_rejections', {}).get(model.architecture, {}),
        'metadata': metadata, 'tensor_count': len(gguf.tensors),
        'representative_tensors': [{'name': t.name, 'dimensions': t.dims, 'ggml_type': t.kind} for t in representative[:160]],
        'runtime_revision': report.get('runtime', {}).get('llama_cpp_revision'),
        'patch_sha256': report.get('runtime', {}).get('patch_sha256'),
        'reasons': report.get('reasons', model.reasons),
        'failures': [text for text in [report.get('diagnostic', '')] +
                     [p.get('diagnostic', '') for p in report.get('profiles', [])] if text],
        'profile_failures': [p for p in report.get('profiles', []) if p.get('status') != 'compatible_experimental'],
        'reproduce': f'python -m looplab explore {model.id} --no-resume',
        'next_steps': [
            'Check the pinned native model loader and original graph before changing metadata or tensor names.',
            'For an unrecognized but similar graph, extend runtime/auto_adapters.py and preserve its original layer body.',
            'For recurrent, MLA, shared, or convolution state, implement and test the state snapshot/restore semantics explicitly.',
            'For MoE, retain the first expert IDs and mixture weights through the shared routing helper.',
            'For repeat-request instability, inspect cache clearing, graph reuse and backend numerical behavior with cache_prompt=false.',
            'Rebuild, pass the independent native reference tests, then rerun this exact model and its K=2/3/4 probes.',
            'Treat successful smoke tests as experimental compatibility, never proof of quality gains or full paper reproduction.'
        ]}


def current(report, model, info):
    stale_effect_check = report.get('effect_check_version') != EFFECT_VERSION and any(
        p.get('loop_smoke', {}).get('finite') and p['loop_smoke'].get('repeat_stable') and
        not p['loop_smoke'].get('changed_from_baseline') for p in report.get('profiles', []))
    return (not stale_effect_check and report.get('suite') == VERSION and report.get('status') not in {'running', 'cancelled', 'resource_skipped', 'preflight_skipped'}
            and report.get('model', {}).get('header_sha256') == model.header_sha256
            and report.get('model', {}).get('size_bytes') == model.size_bytes
            and report.get('source_mtime_ns') == model.mtime_ns
            and report.get('runtime', {}).get('patch_sha256') == info.get('patch_sha256')
            and report.get('runtime', {}).get('execution_policy') == info.get('execution_policy'))


def public_exploration(report):
    keys = ('schema', 'suite', 'status', 'reasons', 'diagnostic', 'error_type', 'model', 'baseline_check',
            'profiles', 'paper_reproduction', 'quality_tested', 'recommended_profile', 'repair', 'seconds',
            'storage_before', 'storage_after', 'backend_recovery', 'effect_check_version')
    result = {key: report[key] for key in keys if key in report}
    if 'runtime' in report:
        result['runtime'] = {key: report['runtime'][key] for key in ('llama_cpp_revision', 'patch_sha256', 'execution_policy')
                             if key in report['runtime']}
    return result


def passing_counts(report):
    return {p['passes'] for p in report.get('profiles', []) if p.get('status') == 'compatible_experimental'}


def explore_model(worker, model, state, cancel=None, progress=None, resume=True):
    """Retry numerical failures without weakening checks or replacing good profiles."""
    report = _explore_once(worker, model, state, cancel, progress, resume)
    recoverable = any(not p.get('anchor_identity', {}).get('passed') or
                      (p.get('loop_smoke', {}).get('finite') and not p['loop_smoke'].get('repeat_stable'))
                      for p in report.get('profiles', []))
    if (not recoverable or report['status'] == 'cancelled' or not model.eligible or
            not report.get('baseline_check', {}).get('passed') or worker.cpu_mode(model) or
            (resume and report.get('backend_recovery', {}).get('version') == RECOVERY_VERSION)):
        return report
    state = Path(state)
    target = state / 'exploration' / f'{model.id}.json'
    modes = ['vulkan_f32', 'cpu'] if worker.execution_mode(model) == 'default' else ['cpu']
    recovery = {'version': RECOVERY_VERSION, 'selected': worker.execution_mode(model), 'attempts': []}
    def record_attempt(value, mode):
        recovery['attempts'].append({'mode': mode, 'status': value['status'],
            'execution_policy': value.get('runtime', {}).get('execution_policy'),
            'profiles': value.get('profiles', []),
            **({'diagnostic': value['diagnostic']} if 'diagnostic' in value else {})})
    record_attempt(report, worker.execution_mode(model))
    for mode in modes:
        if cancel and cancel.is_set():
            return {**report, 'status': 'cancelled'}
        if progress: progress(f'{model.name}: retrying numerical checks with {mode}')
        # Failed trials cannot replace an existing report or expose untested settings.
        with tempfile.TemporaryDirectory(prefix='precision-trial-', dir=state) as directory:
            with worker.execution_trial(model, mode):
                candidate = _explore_once(worker, model, Path(directory), cancel, progress, resume=False)
        if candidate['status'] == 'cancelled' or (cancel and cancel.is_set()):
            return {**report, 'status': 'cancelled'}
        record_attempt(candidate, mode)
        if passing_counts(candidate) > passing_counts(report):
            recovery['selected'] = mode
            candidate['backend_recovery'] = recovery
            candidate['storage_before'] = report['storage_before']
            candidate['storage_after'] = storage_snapshot(state)
            worker.remember_execution_mode(model, mode)
            atomic_json(target, candidate)
            repair = state / 'repairs' / f'{model.id}.json'
            if candidate.get('repair'):
                atomic_json(repair, candidate['repair'])
            else:
                repair.unlink(missing_ok=True)
            report = candidate
            if passing_counts(candidate) == {2, 3, 4}:
                return candidate
    report['backend_recovery'] = recovery
    atomic_json(target, report)
    return report


def _explore_once(worker, model, state, cancel=None, progress=None, resume=True):
    state = Path(state)
    target = state / 'exploration' / f'{model.id}.json'
    info = worker.build_info(model)
    if resume and target.exists():
        previous = json.loads(target.read_text())
        if current(previous, model, info):
            if progress: progress(f'{model.name}: reused completed {previous["status"]} result')
            return previous
    # A competing process must not overwrite the active model's report with a
    # spurious startup failure. Refuse it before writing any report.
    worker.close()
    if model.loadable:
        worker.claim()
    started = time.monotonic()
    report = {'schema': 1, 'suite': VERSION, 'status': 'running', 'model': model.public(),
              'effect_check_version': EFFECT_VERSION,
              'source_mtime_ns': model.mtime_ns, 'runtime': info, 'profiles': [], 'paper_reproduction': False,
              'quality_tested': False, 'recommended_profile': 'baseline', 'storage_before': storage_snapshot(state)}
    def save():
        report['seconds'] = time.monotonic() - started
        atomic_json(target, report)
    try:
        check_cancel(cancel)
        if not model.loadable:
            report.update(status='not_applicable' if model.role == 'projector' else 'preflight_skipped', reasons=model.load_reasons)
            return report
        if reason := resource_reason(model):
            report.update(status='resource_skipped', reasons=[reason])
            return report
        if progress: progress(f'{model.name}: original model ({model.role})')
        with worker.use(model, Profile.for_model(model), diagnostic=True):
            if model.role == 'embedding':
                result = worker.json('/v1/embeddings', {'input': 'A short local compatibility test.', 'model': model.id})
                vectors = [item.get('embedding', []) for item in result.get('data', [])]
                passed = bool(vectors) and all(v and all(isinstance(x, (int,float)) and math.isfinite(x) for x in v) for v in vectors)
                report.update(status='embedding_passed' if passed else 'baseline_failed',
                              baseline_check={'passed': bool(passed), 'embedding_dimensions': [len(v) for v in vectors]})
                return report
            baseline = probe(worker)
            report['baseline_check'] = {'passed': finite_probe(baseline), 'tokens': len(baseline.get('tokens', [])),
                                        'effective_gpu_layers': 0 if worker.cpu_mode(model) else worker.gpu_layers}
            if not report['baseline_check']['passed']:
                report.update(status='baseline_failed', reasons=['No finite token probability distribution'])
                return report
        save()
        if not model.eligible:
            report.update(status='baseline_only', reasons=model.reasons)
            return report
        for count in (2, 3, 4):
            check_cancel(cancel)
            entry = {'passes': count, 'profile': asdict(Profile.for_model(model, count)),
                     'status': 'conformance_failed', 'anchor_identity': {'passed': False}}
            report['profiles'].append(entry)
            try:
                if progress: progress(f'{model.name}: {count} passes, identity and loop smoke test')
                with worker.use(model, Profile.for_model(model, count, anchor=1.0)):
                    identity = probe(worker)
                delta = distribution_difference(baseline, identity)
                passed = bool(baseline.get('tokens')) and baseline.get('tokens') == identity.get('tokens') and delta is not None and delta < .005
                entry['anchor_identity'] = {'passed': passed, 'max_logprob_error': delta}
                if passed:
                    with worker.use(model, Profile.for_model(model, count)):
                        looped = probe(worker)
                        repeat = probe(worker)  # Independent request: no cross-request cache reuse.
                    finite = finite_probe(looped)
                    repeat_delta = distribution_difference(looped, repeat)
                    stable = looped.get('tokens') == repeat.get('tokens') and repeat_delta is not None and repeat_delta < .005
                    effect = distribution_difference(baseline, looped)
                    changed = distribution_changed(baseline, looped)
                    entry['loop_smoke'] = {'passed': finite and stable and changed, 'finite': finite,
                        'repeat_stable': stable, 'repeat_max_logprob_error': repeat_delta,
                        'changed_from_baseline': changed, 'same_sequence_logprob_difference': effect}
                    entry['status'] = 'compatible_experimental' if entry['loop_smoke']['passed'] else 'smoke_failed'
            except Exception as exc:
                entry.update(status='failed', **failure(exc, model, worker))
                worker.close()
            save()
        compatible = sum(p['status'] == 'compatible_experimental' for p in report['profiles'])
        report['status'] = 'compatible_experimental' if compatible == 3 else 'partially_compatible' if compatible else 'conformance_failed'
    except Cancelled:
        report['status'] = 'cancelled'
    except KeyboardInterrupt:
        report['status'] = 'cancelled'
        raise
    except MemoryError as exc:
        report.update(status='resource_skipped', reasons=[str(exc)])
    except Exception as exc:
        report.update(status='baseline_failed', **failure(exc, model, worker))
    finally:
        worker.close()
        if report['status'] not in {'compatible_experimental', 'embedding_passed', 'not_applicable', 'cancelled'}:
            report['repair'] = repair_handoff(model, report)
            atomic_json(state / 'repairs' / f'{model.id}.json', report['repair'])
        report['storage_after'] = storage_snapshot(state)
        save()
    return report
