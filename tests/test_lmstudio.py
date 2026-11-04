# SPDX-License-Identifier: AGPL-3.0-or-later
from dataclasses import asdict
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import struct
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from loop_gguf import GGUF, Tensor, align, encode_value, pack_string
from looplab.explore import VERSION
from looplab.lmstudio import digest, install_engine, install_models, select_profiles, remove_installation
from looplab.lmstudio_engine import argument, proxy_handler, resolve_profile, validate_load_options
from looplab.models import Profile, atomic_json, discover
from looplab.server import App


@pytest.fixture
def installation(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    path = tmp_path / 'weights/original/model.gguf'
    path.parent.mkdir(parents=True)
    fields = [pack_string(k) + struct.pack('<I', kind) + encode_value(kind, value)
              for k, kind, value in [('general.architecture', 8, 'llama'),
                                    ('llama.block_count', 4, 12), ('general.name', 8, 'Original')]]
    fields.append(pack_string('tokenizer.ggml.tokens') + struct.pack('<IIQ', 9, 8, 3) +
                  b''.join(pack_string(v) for v in ['one', 'two', 'three']))
    data = b'GGUF' + struct.pack('<IQQ', 3, 1, len(fields)) + b''.join(fields)
    data += Tensor('source.weight', (16,), 0, 0).descriptor()
    path.write_bytes(data + b'\0' * (align(len(data), 32) - len(data)) + b'SOURCE WEIGHT!!! ' * 4)
    app = App(tmp_path / 'state', roots=[tmp_path / 'weights'])
    app.worker.build_info = lambda model=None: {'patch_sha256': 'tested-build', 'execution_policy': {'context': 2048}}
    model = next(iter(app.models.values()))
    report = {'suite': VERSION, 'model': model.public(), 'source_mtime_ns': model.mtime_ns,
              'runtime': app.worker.build_info(), 'profiles': [
                  {'passes': k, 'profile': asdict(Profile.for_model(model, k)),
                   'anchor_identity': {'passed': True}, 'loop_smoke': {'passed': k != 3}}
                  for k in (2, 3, 4)]}
    atomic_json(app.state / 'exploration' / f'{model.id}.json', report)
    return app, model, tmp_path / 'entries'


def test_descriptors_have_no_weights_and_only_current_passing_profiles(installation):
    app, model, root = installation
    original = Path(model.path).read_bytes()
    registry = install_models(app, root)
    assert len(registry['entries']) == 2
    assert sorted(e['profile']['passes'] for e in registry['entries'].values()) == [2, 4]
    for name in registry['entries']:
        descriptor = GGUF(name)
        assert [t.name for t in descriptor.tensors] == ['looplab.descriptor']
        assert 'tokenizer.ggml.tokens' not in descriptor.values
        assert b'SOURCE WEIGHT!!!' not in Path(name).read_bytes()
        resolved, profile = resolve_profile(app, name)
        assert resolved.id == model.id
        assert profile.passes in (2, 4)
    assert Path(model.path).read_bytes() == original
    assert discover([root]) == ([], [])
    assert install_models(app, root) == registry  # Idempotent installation.


def test_descriptors_reject_edits_and_stale_tests(installation):
    app, model, root = installation
    registry = install_models(app, root)
    path = Path(next(iter(registry['entries'])))
    original = path.read_bytes()
    path.write_bytes(original[:-4] + struct.pack('<f', 1))
    with pytest.raises(ValueError, match='unregistered or changed'):
        resolve_profile(app, path)
    with pytest.raises(ValueError, match='untracked or edited'):
        install_models(app, root)
    path.write_bytes(original)
    app.worker.build_info = lambda model=None: {'patch_sha256': 'different-build', 'execution_policy': {'context': 2048}}
    with pytest.raises(ValueError, match='conformance'):
        resolve_profile(app, path)
    with pytest.raises(ValueError, match='No current passing'):
        select_profiles(app)


def test_sync_retires_failed_profiles_and_preserves_user_settings(installation):
    app, model, root = installation
    registry = install_models(app, root)
    entry = next(iter(registry['entries'].values()))
    defaults = Path(entry['defaults_path'])
    defaults.write_text('{"custom":"user setting"}')
    install_models(app, root)
    assert defaults.read_text() == '{"custom":"user setting"}'
    report_path = app.state / 'exploration' / f'{model.id}.json'
    report = json.loads(report_path.read_text())
    for profile in report['profiles']:
        profile['loop_smoke']['passed'] = False
    atomic_json(report_path, report)
    registry = install_models(app, root, allow_empty=True)
    assert not registry['entries']
    assert defaults.exists()
    assert not list(root.rglob('*.gguf'))
    assert Path(model.path).exists()


def test_engine_install_preserves_official_files_and_shares_storage(installation):
    app, _, root = installation
    backends = root / 'backends'
    source = backends / 'official'
    source.mkdir(parents=True)
    manifest = {'name': 'official', 'version': '2.69.0', 'platform': 'linux',
                'cpu': {'architecture': 'x86_64'}, 'engine': 'llama.cpp',
                'engine_protocol_server': {'runtime_kind': 'llama-server', 'executable_relative_path': 'llama-server'}}
    atomic_json(source / 'backend-manifest.json', manifest)
    (source / 'display-data.json').write_text('[]')
    (source / 'llama-server').write_text('official executable')
    (source / 'binding.node').write_text('official binding')
    before = {p.name: digest(p) for p in source.iterdir()}
    installed = install_engine(app, source, backends)
    assert {p.name: digest(p) for p in source.iterdir()} == before
    assert (installed / 'binding.node').stat().st_ino == (source / 'binding.node').stat().st_ino
    assert (installed / 'stock-llama-server').stat().st_ino == (source / 'llama-server').stat().st_ino
    assert (installed / 'backend-manifest.json').stat().st_ino != (source / 'backend-manifest.json').stat().st_ino
    assert install_engine(app, source, backends) == installed


def test_removal_only_removes_managed_entries_and_runtime(installation, monkeypatch):
    import looplab.lmstudio as module
    app, model, root = installation
    registry = install_models(app, root)
    backends = root / 'backends'
    destination = backends / f'{module.ENGINE_NAME}-{module.ENGINE_VERSION}'
    destination.mkdir(parents=True)
    atomic_json(destination / 'looplab-engine.json', {'managed_by': module.MARKER, 'state': str(app.state.resolve())})
    owned_path = next(iter(registry['entries'].values()))['indexed_model_identifier']
    instances = [
        {'modelKey': 'normalized-key', 'indexedModelIdentifier': owned_path, 'identifier': 'owned', 'status': 'idle'},
        {'modelKey': 'normalized-key', 'indexedModelIdentifier': 'other/model/file.gguf', 'identifier': 'unrelated'},
    ]
    calls = []
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: json.dumps(instances))
    monkeypatch.setattr(module.subprocess, 'run', lambda args, **kwargs: calls.append(args))
    result = remove_installation(app.state, backends)
    assert result['removed']
    assert not destination.exists()
    assert not (app.state / 'lmstudio-models.json').exists()
    assert Path(model.path).exists()
    assert all(not Path(p).exists() for p in registry['entries'])
    assert calls == [['lms', 'unload', 'owned']]


def test_engine_protocol_proxy_preserves_stream_and_disables_prefix_reuse():
    captured = []
    class Response(io.BytesIO):
        status = 200
        headers = {'Content-Type': 'text/event-stream'}
    class Worker:
        def request(self, path, payload):
            captured.append((path, payload))
            return Response(b'data: {"content":"4"}\n\ndata: [DONE]\n\n')
    server = ThreadingHTTPServer(('127.0.0.1', 0), proxy_handler(Worker(), 'real-model', 4, ('runtime-secret',)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    headers = {'Authorization': 'Bearer runtime-secret', 'Content-Type': 'application/json'}
    try:
        with pytest.raises(HTTPError) as exc:
            urlopen(base + '/props')
        assert exc.value.code == 401
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(base + '/props', headers={**headers, 'Origin': 'https://example.invalid'}))
        assert exc.value.code == 403
        payload = {'prompt': 'two plus two', 'cache_prompt': True, 'model': 'alias', 'stream': True}
        with urlopen(Request(base + '/completion', data=json.dumps(payload).encode(), headers=headers)) as response:
            assert response.headers['X-LoopLab-Passes'] == '4'
            assert response.read() == b'data: {"content":"4"}\n\ndata: [DONE]\n\n'
        assert captured[-1][1]['cache_prompt'] is False
        assert captured[-1][1]['model'] == 'real-model'
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(base + '/models/load', data=b'{}', headers=headers))
        assert exc.value.code == 404
        assert len(captured) == 1
    finally:
        server.shutdown()
        server.server_close()


def test_launcher_argument_parser_handles_both_forms():
    assert argument(['-m', 'original', '--model=descriptor'], '-m', '--model') == 'descriptor'
    assert argument(['--port=9999'], '--port') == '9999'
    with pytest.raises(ValueError, match='Missing value'):
        argument(['--model'], '--model')
    validate_load_options(['-c', '2048', '--parallel=1'], 2048)
    with pytest.raises(ValueError, match='context length 2048'):
        validate_load_options(['--ctx-size=4096'], 2048)
    with pytest.raises(ValueError, match='Concurrent Predictions = 1'):
        validate_load_options(['--parallel=4'], 2048)
