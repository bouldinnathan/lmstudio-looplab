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
from looplab import explore, models
from looplab.models import Model, Profile, atomic_json
from looplab.server import App


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
    manifest = ollama / 'manifests/registry.ollama.ai/library/test/latest'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'layers': [{'mediaType': 'application/vnd.ollama.image.model', 'digest': 'sha256:' + digest}]}))
    monkeypatch.setattr(models, 'inspect', lambda *args: model())
    rows, errors = models.discover([studio, ollama])
    assert not errors and len(rows) == 1
    assert rows[0].aliases == ['library/test/latest']


class FakeWorker:
    token = 'secret-token'
    gpu_layers = 999
    def __init__(self): self.profile = None; self.calls = 0
    def build_info(self): return {'patch_sha256': 'build', 'execution_policy': {'context': 2048}}
    def close(self): pass
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
    app.worker.build_info = lambda: FakeWorker().build_info()
    report = {'suite': explore.VERSION, 'model': m.public(), 'source_mtime_ns': m.mtime_ns,
              'runtime': app.worker.build_info(), 'profiles': [{'passes': 2, 'profile': asdict(Profile.for_model(m, 2)),
              'anchor_identity': {'passed': True}, 'loop_smoke': {'passed': False}}]}
    atomic_json(tmp_path / 'exploration/tiny.json', report)
    with pytest.raises(ValueError, match='conformance'):
        app.profile(m, 'loop2')
    report['profiles'][0]['loop_smoke']['passed'] = True
    atomic_json(tmp_path / 'exploration/tiny.json', report)
    assert app.profile(m, 'loop2').passes == 2
