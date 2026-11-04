"""Install ordinary LM Studio model entries backed by the tested LoopLab runtime.

The small GGUF descriptors contain metadata and one dummy tensor, never weights.
Stock engines fail to load them, preventing a loop-labelled baseline response.
LM Studio's own bindings are reused locally, not redistributed by this project.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

from loop_gguf import GGUF, Tensor, align, encode_value, pack_string
from .models import atomic_json, slug, state_directory
from .server import App

ENGINE_NAME = 'looplab-linux-x86_64'
# LM Studio uses this version for llama.cpp feature gates. Preserve the version
# of the pinned official bindings; the bridge has its own independent version.
ENGINE_VERSION = '2.69.0'
BRIDGE_VERSION = '0.3.0'
MARKER = 'lmstudio-looplab-engine-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def descriptor_bytes(model, profile):
    source = GGUF(model.path)
    # Token vocabularies can be megabytes long. Inference gets the real tokenizer
    # from the source GGUF; the indexer only needs descriptive metadata here.
    fields = {f.key: f.raw for f in source.fields
              if not f.key.startswith(('tokenizer.', 'split.', 'looplab.'))}
    for f in source.fields:
        if f.key in {'tokenizer.ggml.model', 'tokenizer.ggml.pre', 'tokenizer.chat_template'}:
            fields[f.key] = f.raw
    values = {
        'general.name': (8, f'{model.name} Looped {profile.passes}x (LoopLab experimental)'),
        'general.alignment': (4, 32),
        'looplab.descriptor_version': (4, 1),
        'looplab.model_id': (8, model.id),
        'looplab.passes': (4, profile.passes),
        'looplab.source_header_sha256': (8, model.header_sha256),
    }
    for key, (kind, value) in values.items():
        fields[key] = pack_string(key) + struct.pack('<I', kind) + encode_value(kind, value)
    header = b'GGUF' + struct.pack('<IQQ', 3, 1, len(fields)) + b''.join(fields.values())
    header += Tensor('looplab.descriptor', (1,), 0, 0).descriptor()
    return header + b'\0' * (align(len(header), 32) - len(header)) + struct.pack('<f', 0)


def write_owned(path, payload, previous_hash=None):
    """Never overwrite an existing file unless its previous managed hash matches."""
    path = Path(path)
    if path.exists() and digest(path) != previous_hash:
        if path.read_bytes() == payload:
            return
        raise ValueError(f'Refusing to overwrite an untracked or edited file: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.looplab-', delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(payload)
            f.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def select_profiles(app, selection='all', require_any=True):
    models = list(app.models.values()) if selection == 'all' else [app.model(selection)]
    chosen = []
    for model in models:
        if not model.eligible or model.role != 'generation':
            continue
        for passes in (2, 3, 4):
            try:
                chosen.append((model, app.profile(model, f'loop{passes}')))
            except ValueError:
                pass
    if not chosen and require_any:
        raise ValueError('No current passing loop profiles. Run compatibility testing first.')
    return chosen


def remove_owned_entry(path, entry):
    path = Path(path)
    if path.exists():
        if digest(path) != entry['descriptor_sha256']:
            raise ValueError(f'Refusing to remove an edited model descriptor: {path}')
        path.unlink()
        if not any(path.parent.iterdir()):
            path.parent.rmdir()
    defaults = Path(entry['defaults_path'])
    if defaults.exists() and digest(defaults) == entry['defaults_sha256']:
        defaults.unlink()


def install_models(app, model_root, selection='all', allow_empty=False):
    registry_path = app.state / 'lmstudio-models.json'
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {'managed_by': MARKER, 'entries': {}}
    if registry.get('managed_by') != MARKER:
        raise ValueError('Unrecognized LM Studio model registry')
    registry['model_root'] = str(Path(model_root).resolve())
    selected = select_profiles(app, selection, require_any=not allow_empty)
    valid = {(model.id, profile.passes) for model, profile in selected}
    for path, entry in list(registry['entries'].items()):
        if selection != 'all' and entry['model_id'] != app.model(selection).id:
            continue
        if (entry['model_id'], entry['profile']['passes']) not in valid:
            remove_owned_entry(path, entry)
            del registry['entries'][path]
            atomic_json(registry_path, registry)
    for model, profile in selected:
        name = f'{slug(model.name)}-looped-{profile.passes}x'
        relative = Path('looplab') / name / (name + '.gguf')
        path = (Path(model_root) / relative).absolute()
        old = registry['entries'].get(str(path), {})
        payload = descriptor_bytes(model, profile)
        write_owned(path, payload, old.get('descriptor_sha256'))
        defaults = Path.home() / '.lmstudio/.internal/user-concrete-model-default-config' / (str(relative) + '.json')
        defaults_payload = (json.dumps({'preset': '', 'operation': {'fields': []}, 'load': {'fields': [
            {'key': 'llm.load.contextLength', 'value': app.worker.context},
            {'key': 'llm.load.numParallelSessions', 'value': 1},
            {'key': 'llm.load.llama.acceleration.offloadRatio', 'value': 0 if app.worker.cpu_mode(model) else 1},
            {'key': 'llm.load.llama.flashAttention', 'value': False},
            {'key': 'llm.load.llama.evalBatchSize', 'value': app.worker.context},
            {'key': 'llm.load.llama.physicalBatchSize', 'value': app.worker.context},
            {'key': 'llm.load.llama.contextCheckpoints', 'value': 0},
            {'key': 'llm.load.llama.autoFit', 'value': not app.worker.cpu_mode(model)},
            {'key': 'llm.load.offloadKVCacheToGpu', 'value': not app.worker.cpu_mode(model)},
        ]}}, indent=2) + '\n').encode()
        # Preserve load defaults the user edited in LM Studio. Context and
        # parallelism are still checked before starting the fixed test profile.
        defaults_hash = old.get('defaults_sha256')
        if not defaults.exists() or digest(defaults) == defaults_hash or defaults.read_bytes() == defaults_payload:
            write_owned(defaults, defaults_payload, defaults_hash)
            defaults_hash = digest(defaults)
        elif not defaults_hash:
            raise ValueError(f'An unrelated model configuration already exists: {defaults}')
        registry['entries'][str(path)] = {
            'model_id': model.id, 'file_stem': name, 'indexed_model_identifier': relative.as_posix(),
            'profile': asdict(profile),
            'descriptor_sha256': digest(path), 'descriptor_bytes': len(payload),
            'defaults_path': str(defaults), 'defaults_sha256': defaults_hash,
        }
        atomic_json(registry_path, registry)
    return registry


def sync_if_installed(app):
    path = app.state / 'lmstudio-models.json'
    engine = Path.home() / '.lmstudio/extensions/backends' / f'{ENGINE_NAME}-{ENGINE_VERSION}'
    if not path.exists() or not (engine / 'looplab-engine.json').exists():
        return None
    registry = json.loads(path.read_text())
    registry = install_models(app, registry['model_root'], allow_empty=True)
    return {'profiles': len(registry['entries']), 'descriptor_bytes': sum(
        e['descriptor_bytes'] for e in registry['entries'].values())}


def install_engine(app, source, backends):
    source, backends = Path(source).resolve(), Path(backends).resolve()
    manifest = json.loads((source / 'backend-manifest.json').read_text())
    if manifest.get('engine') != 'llama.cpp' or manifest.get('engine_protocol_server', {}).get('runtime_kind') != 'llama-server':
        raise ValueError('An installed llama.cpp Engine Protocol runtime is required')
    if manifest.get('version') != ENGINE_VERSION:
        raise ValueError(f'This bridge is tested with official runtime {ENGINE_VERSION}')
    if manifest.get('platform') != 'linux' or manifest.get('cpu', {}).get('architecture') != 'x86_64':
        raise ValueError('This installer currently supports Linux x86_64 only')
    if source.name.startswith('looplab-'):
        raise ValueError('Use an official LM Studio runtime as the source')
    if source.stat().st_dev != backends.stat().st_dev:
        raise ValueError('Source and destination must share a filesystem for storage-free hard links')
    app.worker.build_info()
    destination = backends / f'{ENGINE_NAME}-{ENGINE_VERSION}'
    if destination.exists():
        config = json.loads((destination / 'looplab-engine.json').read_text())
        if config.get('managed_by') != MARKER or config.get('state') != str(app.state.resolve()):
            raise ValueError('Refusing to replace an unrelated runtime')
        if (config.get('binary') != str(app.worker.binary) or config.get('context') != app.worker.context
                or config.get('gpu_layers') != app.worker.gpu_layers):
            raise ValueError('Installed runtime uses different tested settings; remove it before reinstalling')
        return destination
    staging = Path(tempfile.mkdtemp(prefix='.looplab-stage-', dir=backends))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True, copy_function=os.link, symlinks=True)
        # Replace metadata atomically; the linked official files stay untouched.
        manifest.update(name=ENGINE_NAME, version=ENGINE_VERSION)
        manifest['engine_protocol_server']['executable_relative_path'] = 'llama-server'
        atomic_json(staging / 'backend-manifest.json', manifest)
        atomic_json(staging / 'display-data.json', [['en', {
            'langKey': 'en', 'displayName': 'LoopLab (tested loops + standard models)',
            'description': 'LoopLab profiles use the tested runtime; ordinary models use the pinned official runtime.',
            'releaseNotes': [],
        }]])
        (staging / 'engine-protocol-server-artifacts.json').unlink(missing_ok=True)
        (staging / 'llama-server').rename(staging / 'stock-llama-server')
        config = {'managed_by': MARKER, 'bridge_version': BRIDGE_VERSION, 'state': str(app.state.resolve()),
                  'binary': str(app.worker.binary), 'context': app.worker.context,
                  'gpu_layers': app.worker.gpu_layers, 'source_runtime': str(source),
                  'max_model_gib': app.max_model_bytes / 1024**3}
        atomic_json(staging / 'looplab-engine.json', config)
        launcher = f'#!{sys.executable}\nfrom looplab.lmstudio_engine import main\nmain()\n'
        (staging / 'llama-server').write_text(launcher)
        (staging / 'llama-server').chmod(0o755)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def remove_installation(state, backends):
    """Explicit rollback; unload only our instances and preserve edited defaults."""
    state, backends = Path(state), Path(backends)
    destination = backends / f'{ENGINE_NAME}-{ENGINE_VERSION}'
    config = json.loads((destination / 'looplab-engine.json').read_text())
    if config.get('managed_by') != MARKER or config.get('state') != str(state.resolve()):
        raise ValueError('Refusing to remove an unrelated runtime')
    registry_path = state / 'lmstudio-models.json'
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {'managed_by': MARKER, 'entries': {}}
    if registry.get('managed_by') != MARKER:
        raise ValueError('Unrecognized model registry')
    # Preflight ownership before touching the running application or any files.
    for path, entry in registry['entries'].items():
        if Path(path).exists() and digest(path) != entry['descriptor_sha256']:
            raise ValueError(f'Refusing to remove an edited model descriptor: {path}')
    preferences = Path.home() / '.lmstudio/.internal/backend-preferences-v1.json'
    selected = json.loads(preferences.read_text()) if preferences.exists() else []
    if any(p.get('name') == ENGINE_NAME for p in selected):
        source = Path(config['source_runtime']) / 'backend-manifest.json'
        if not source.exists():
            raise ValueError('Select an installed official runtime with lms runtime select before removing LoopLab')
        official = json.loads(source.read_text())
        subprocess.run(['lms', 'runtime', 'select', f'{official["name"]}@{official["version"]}'], check=True)
    # LM Studio normalizes API model keys (for example, it strips Q4 tags).
    # Match exact indexed paths so rollback also handles those names and never
    # unloads a different file that happens to have the same normalized key.
    indexed_paths = {entry.get('indexed_model_identifier') or Path(path).relative_to(
        Path(registry['model_root'])).as_posix() for path, entry in registry['entries'].items()}
    instances = json.loads(subprocess.check_output(['lms', 'ps', '--json'], text=True))
    for instance in instances:
        if instance.get('indexedModelIdentifier') in indexed_paths:
            if instance.get('status') not in (None, 'idle'):
                raise ValueError('A LoopLab model is generating; finish its request before uninstalling')
            subprocess.run(['lms', 'unload', instance['identifier']], check=True)
    for path, entry in list(registry['entries'].items()):
        remove_owned_entry(path, entry)
        del registry['entries'][path]
        atomic_json(registry_path, registry)
    registry_path.unlink(missing_ok=True)
    shutil.rmtree(destination)
    return {'removed': True, 'source_weights_removed': False}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=state_directory())
    parser.add_argument('--source-backend', type=Path)
    parser.add_argument('--model-root', type=Path, default=Path.home() / '.lmstudio/models')
    parser.add_argument('--model', default='all')
    parser.add_argument('--remove', action='store_true', help='Restore the official runtime and remove managed profile entries')
    args = parser.parse_args()
    if args.remove:
        print(json.dumps(remove_installation(args.state, Path.home() / '.lmstudio/extensions/backends')))
        return
    if args.source_backend is None:
        parser.error('--source-backend is required for installation')
    app = App(args.state)
    try:
        # Validate the reports before installing any entries.
        select_profiles(app, args.model)
        destination = install_engine(app, args.source_backend, Path.home() / '.lmstudio/extensions/backends')
        registry = install_models(app, args.model_root, args.model)
        print(json.dumps({'runtime': str(destination), 'runtime_alias': f'{ENGINE_NAME}@{ENGINE_VERSION}',
                          'profiles': len(registry['entries']),
                          'descriptor_bytes': sum(e['descriptor_bytes'] for e in registry['entries'].values()),
                          'weight_bytes_copied': 0}, indent=2))
    finally:
        app.worker.close()


if __name__ == '__main__':
    main()
