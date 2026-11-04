# SPDX-License-Identifier: AGPL-3.0-or-later
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import importlib.util

import pytest

spec = importlib.util.spec_from_file_location('auto_adapters', Path(__file__).resolve().parents[1] / 'runtime/auto_adapters.py')
adapters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapters)
transform = adapters.transform
from looplab import benchmark, explore, models
from looplab.models import Model, Profile, atomic_json
from looplab.server import App
from looplab.worker import Worker


GRAPH = '''
llama_model_example::graph::graph(const llama_model & model) : llm_graph_context(params) {
    auto * inp_attn = build_attn_inp_kv();
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * cur;
    for (int il = 0; il < n_layer; ++il) {
        // return and braces { } in comments must not affect parsing.
        cur = native_attention(inpL, inp_attn, il);
        cur = native_ffn(cur, model.layers[il]);
        inpL = cur;
    }
    cur = build_norm(inpL);
}
'''


def test_generated_adapter_preserves_layer_and_output_code():
    result, detail = transform(GRAPH)
    assert detail['strategy'] == 'preserve_native_layer_body'
    assert 'cur = native_attention(inpL, inp_attn, il);' in result
    assert 'cur = native_ffn(cur, model.layers[il]);' in result
    assert 'cur = build_norm(inpL);' in result
    assert 'inpL = looplab::run(' in result
    assert 'return cur;' in result
    result, detail = transform(GRAPH.replace('build_attn_inp_kv()', 'build_attn_inp_kv_iswa()'))
    assert result and detail['cache_interfaces'] == ['kv_iswa']


@pytest.mark.parametrize('change', [
    lambda s: s.replace('build_attn_inp_kv()', 'build_inp_mem_hybrid()'),
    lambda s: s.replace('build_attn_inp_kv()', 'build_attn_inp_k()'),
    lambda s: s.replace('inpL = cur;', 'if (il == 3) continue; inpL = cur;'),
    lambda s: s.replace('inpL = cur;', 'inpL = another;'),
])
def test_unknown_state_and_boundaries_require_repair(change):
    result, reason = transform(change(GRAPH))
    assert result is None and reason


def model():
    return Model('tiny', 'tiny', '/private/model.gguf', 'example', 8, 100, 1, 'hash', True, [], 'block', tier='experimental')


def test_ram_guard_leaves_reserve(monkeypatch):
    monkeypatch.setattr(models, 'available_memory', lambda: 10 * 1024**3)
    assert 'reserve' in models.resource_reason(model())
    monkeypatch.setattr(models, 'available_memory', lambda: 40 * 1024**3)
    assert models.resource_reason(model()) is None


def test_cpu_fallback_policy_invalidates_only_affected_models(tmp_path):
    binary = tmp_path / 'llama-server'
    binary.touch()
    atomic_json(tmp_path / 'looplab-build.json', {'method': 'anchored_damped_euler', 'version': 2})
    worker = Worker(tmp_path / 'state', binary)
    small, large = model(), model()
    large.size_bytes = 55 * 1024**3
    small_policy = worker.build_info(small)['execution_policy']
    large_policy = worker.build_info(large)['execution_policy']
    assert not worker.cpu_mode(small) and worker.cpu_mode(large)
    assert 'cpu_offload_disabled' not in small_policy
    assert large_policy['cpu_offload_disabled'] and large_policy['auto_fit'] is False
    worker.gpu_layers = 0
    assert worker.build_info(small)['execution_policy']['cpu_offload_disabled']


def test_manifest_discovery_deduplicates_shared_files(tmp_path, monkeypatch):
    studio = tmp_path / 'studio'
    ollama = tmp_path / 'ollama'
    studio.mkdir()
    blobs = ollama / 'blobs'
    blobs.mkdir(parents=True)
    digest = '0' * 64
    blob = blobs / ('sha256-' + digest)
    blob.write_bytes(b'fake')
    (studio / 'original.gguf').symlink_to(blob)
    manifest = ollama / 'manifests/registry.ollama.ai/library/test/latest.gguf'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'layers': [{'mediaType': 'application/vnd.ollama.image.model', 'digest': 'sha256:' + digest}]}))
    monkeypatch.setattr(models, 'inspect', lambda *args: model())
    rows, errors = models.discover([studio, ollama])
    assert not errors and len(rows) == 1
    assert rows[0].aliases == ['library/test/latest.gguf']


class FakeWorker:
    token = 'secret-token'
    gpu_layers = 999
    def __init__(self): self.profile = None; self.calls = 0
    def build_info(self, model=None): return {'patch_sha256': 'build', 'execution_policy': {'context': 2048}}
    def close(self): pass
    def claim(self): pass
    def cpu_mode(self, model=None): return True
    @contextmanager
    def use(self, m, p, diagnostic=False):
        self.profile = p
        yield self
    def json(self, path, payload):
        self.calls += 1
        if self.profile.passes == 3:
            raise RuntimeError('unsupported kernel at /private/model.gguf secret-token')
        probability = -1 if self.profile.passes == 1 or self.profile.anchor == 1 else -.9
        return {'tokens': [1], 'completion_probabilities': [{'id': 1, 'top_logprobs': [{'id': 1, 'logprob': probability}]}]}


def test_survey_continues_after_profile_failure_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(explore, 'resource_reason', lambda m: None)
    monkeypatch.setattr(explore, 'repair_handoff', lambda m, r: {'status': r['status']})
    worker = FakeWorker()
    report = explore.explore_model(worker, model(), tmp_path)
    assert report['status'] == 'partially_compatible'
    assert [p['status'] for p in report['profiles']] == ['compatible_experimental', 'failed', 'compatible_experimental']
    assert 'secret-token' not in json.dumps(explore.public_exploration(report))
    assert '/private/model.gguf' not in json.dumps(explore.public_exploration(report))
    calls = worker.calls
    assert explore.explore_model(worker, model(), tmp_path)['status'] == 'partially_compatible'
    assert worker.calls == calls
    changed = model()
    changed.mtime_ns += 1
    assert not explore.current(report, changed, worker.build_info())


def test_experimental_chat_requires_actual_loop_smoke(tmp_path):
    app = App(tmp_path, roots=[tmp_path / 'absent'])
    m = model()
    app.worker.build_info = lambda model=None: FakeWorker().build_info()
    report = {'suite': explore.VERSION, 'model': m.public(), 'source_mtime_ns': m.mtime_ns,
              'runtime': app.worker.build_info(), 'profiles': [{'passes': 2, 'profile': asdict(Profile.for_model(m, 2)),
              'anchor_identity': {'passed': True}, 'loop_smoke': {'passed': False}}]}
    atomic_json(tmp_path / 'exploration/tiny.json', report)
    with pytest.raises(ValueError, match='conformance'):
        app.profile(m, 'loop2')
    report['profiles'][0]['loop_smoke']['passed'] = True
    atomic_json(tmp_path / 'exploration/tiny.json', report)
    assert app.profile(m, 'loop2').passes == 2


def test_precision_choice_persists_and_invalidates_changed_sources(tmp_path):
    binary = tmp_path / 'llama-server'
    binary.touch()
    marker = {'method': 'anchored_damped_euler', 'version': 2, 'patch_sha256': 'build'}
    atomic_json(tmp_path / 'looplab-build.json', marker)
    worker = Worker(tmp_path / 'workers', binary)
    m = model()
    default = worker.build_info(m)['execution_policy']
    with worker.execution_trial(m, 'vulkan_f32'):
        assert worker.build_info(m)['execution_policy']['vulkan_f16'] is False
    assert worker.build_info(m)['execution_policy'] == default
    worker.remember_execution_mode(m, 'vulkan_f32')
    fresh = Worker(worker.state, binary)
    assert fresh.execution_mode(m) == 'vulkan_f32'
    m.mtime_ns += 1
    assert fresh.build_info(m)['execution_policy'] == default
    m.mtime_ns -= 1
    atomic_json(tmp_path / 'looplab-build.json', {**marker, 'patch_sha256': 'new-build'})
    assert fresh.build_info(m)['execution_policy'] == default


class RecoveryWorker(FakeWorker):
    def __init__(self):
        super().__init__()
        self.mode = 'default'
        self.remembered = None
    def cpu_mode(self, model=None): return self.mode == 'cpu'
    def execution_mode(self, model=None): return self.mode
    def remember_execution_mode(self, model, mode): self.remembered = self.mode = mode
    @contextmanager
    def execution_trial(self, model, mode):
        previous = self.mode
        self.mode = mode
        try: yield
        finally: self.mode = previous


@pytest.mark.parametrize('recovered', [True, False])
def test_recovery_keeps_passed_profiles_and_only_publishes_tested_modes(tmp_path, monkeypatch, recovered):
    worker = RecoveryWorker()
    m = model()
    def result(counts):
        return {'status': 'partially_compatible', 'storage_before': {}, 'baseline_check': {'passed': True}, 'profiles': [
            {'passes': k, 'anchor_identity': {'passed': True},
             'loop_smoke': {'finite': True, 'repeat_stable': k in counts},
             'status': 'compatible_experimental' if k in counts else 'smoke_failed'} for k in (2, 3, 4)]}
    original = result({2, 4})
    target = tmp_path / 'exploration/tiny.json'
    atomic_json(target, original)
    def trial(worker, model, state, *args, **kwargs):
        if worker.mode == 'default': return original.copy()
        # Two different passing profiles must not displace the original two.
        if worker.mode == 'vulkan_f32': return result({2, 3})
        return result({2, 3, 4} if recovered else {2})
    monkeypatch.setattr(explore, '_explore_once', trial)
    report = explore.explore_model(worker, m, tmp_path)
    assert worker.remembered == ('cpu' if recovered else None)
    assert explore.passing_counts(report) == ({2, 3, 4} if recovered else {2, 4})
    assert json.loads(target.read_text()) == report
    assert len(report['backend_recovery']['attempts']) == 3


def test_cancelled_recovery_preserves_active_settings_and_report(tmp_path, monkeypatch):
    worker = RecoveryWorker()
    original = {'status': 'conformance_failed', 'baseline_check': {'passed': True},
                'profiles': [{'passes': 2, 'anchor_identity': {'passed': False}}]}
    target = tmp_path / 'exploration/tiny.json'
    atomic_json(target, original)
    monkeypatch.setattr(explore, '_explore_once', lambda worker, *a, **k:
                        original.copy() if worker.mode == 'default' else {'status': 'cancelled'})
    assert explore.explore_model(worker, model(), tmp_path)['status'] == 'cancelled'
    assert worker.remembered is None and worker.mode == 'default'
    assert json.loads(target.read_text()) == original


def test_changed_topk_still_detects_measured_loop_effect():
    def probe(probabilities):
        return {'tokens': [1], 'completion_probabilities': [{'id': 1,
            'top_logprobs': [{'id': token, 'logprob': value} for token, value in probabilities]}]}
    baseline = probe([(1, -.2), (2, -2), (3, -3)])
    looped = probe([(1, -.3), (4, -2), (5, -3)])
    assert explore.distribution_difference(baseline, looped) is None
    assert explore.distribution_changed(baseline, looped)
    assert not explore.distribution_changed(baseline, baseline)


def test_quality_tests_reuse_recovery_and_skip_failed_loop_profiles(tmp_path, monkeypatch):
    worker = FakeWorker()
    checked = {'status': 'partially_compatible', 'profiles': [
        {'passes': k, 'anchor_identity': {'passed': True}, 'loop_smoke': {'passed': k == 2}}
        for k in (2, 3)]}
    calls = []
    monkeypatch.setattr(explore, 'explore_model', lambda *a, **k: calls.append('preflight') or checked)
    monkeypatch.setattr(benchmark, 'evaluate_cases', lambda *a, **k: [{'passed': True}])
    report = benchmark.test_model(worker, model(), tmp_path, passes=(2, 3), quick=True)
    assert calls == ['preflight'] and report['status'] == 'tested'
    assert report['profiles'][0]['loop_smoke']['passed']
    assert report['profiles'][1]['status'] == 'conformance_failed'
    # FakeWorker raises if K=3 inference is attempted; the failed profile stays blocked.
    assert report['recommended_profile'] == 'baseline'
    app = App(tmp_path, roots=[tmp_path / 'absent'])
    app.worker.build_info = worker.build_info
    assert app.profile(model(), 'loop2').passes == 2
    with pytest.raises(ValueError, match='conformance'):
        app.profile(model(), 'loop3')
