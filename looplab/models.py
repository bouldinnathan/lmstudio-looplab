"""Read-only GGUF discovery and conservative architecture eligibility."""
# SPDX-License-Identifier: AGPL-3.0-or-later
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

from loop_gguf import GGUF

ARCHITECTURES = {"llama", "qwen2", "qwen3", "qwen3moe"}
ROOT = Path(__file__).resolve().parents[1]


def runtime_binary():
    auto = ROOT / '.runtime/auto/build/bin/llama-server'
    return auto if auto.with_name('looplab-build.json').exists() else ROOT / '.runtime/build/bin/llama-server'


def adapter_catalog(binary=None):
    marker = Path(binary or runtime_binary()).parent / 'looplab-build.json'
    if marker.exists():
        info = json.loads(marker.read_text())
        return info.get('adapters', {name: {'tier': 'reference_validated'} for name in info.get('architectures', [])}), info.get('adapter_rejections', {})
    return {name: {'tier': 'reference_validated'} for name in ARCHITECTURES}, {}


def available_memory():
    try:
        values = dict(re.findall(r'^(\w+):\s+(\d+)', Path('/proc/meminfo').read_text(), re.M))
        return int(values['MemAvailable']) * 1024
    except (OSError, KeyError):
        return None


def memory_required(model):
    # Leave room for dequantization, graph buffers and KV; this is a guard, not a profiler.
    return int(model.size_bytes * 1.15) + 4 * 1024**3


def resource_reason(model):
    free = available_memory()
    needed = memory_required(model)
    if free is not None and needed + 8 * 1024**3 > free:
        return f'Estimated {needed / 1024**3:.1f} GiB plus 8 GiB reserve exceeds {free / 1024**3:.1f} GiB available RAM'
    return None


def state_directory():
    return Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))) / "looplab"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", prefix=path.name + ".", suffix=".tmp",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:100] or "model"


@dataclass
class Model:
    id: str
    name: str
    path: str
    architecture: str
    layers: int
    size_bytes: int
    mtime_ns: int
    header_sha256: str
    eligible: bool
    reasons: list[str]
    mode: str
    tier: str = 'reference_validated'
    role: str = 'generation'
    adapter: dict = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    loadable: bool = True
    load_reasons: list[str] = field(default_factory=list)

    def public(self):
        value = asdict(self)
        del value["path"]
        del value["mtime_ns"]
        return value


def inspect(path, max_model_bytes=80 * 1024**3, catalog=None, decisions=None, name=None):
    source = Path(path)
    model = GGUF(source)
    name = name or (source.parent.name if source.suffix == ".gguf" else str(model.values.get("general.name", source.name)))
    if catalog is None:
        catalog, decisions = adapter_catalog()
    adapter = dict(catalog.get(model.arch, {}))
    if model.arch != 'qwen3moe' and model.values.get(f'{model.arch}.expert_count', 0) and adapter:
        adapter['tier'] = 'experimental'
    reasons, load_reasons = [], []
    role = 'generation'
    if model.arch in {'clip', 'siglip', 'whisper'} or model.layers == 0:
        role = 'projector'
        load_reasons.append('Auxiliary vision/audio component; not a standalone language model')
    elif 'embed' in name.lower() or model.arch in {'bert', 'nomic-bert', 'jina-bert-v2'} or model.values.get(f'{model.arch}.pooling_type', -1) not in (-1, None):
        role = 'embedding'
    if not adapter:
        reasons.append((decisions or {}).get(model.arch, {}).get('reason', f'No generated native adapter for {model.arch or "unknown architecture"}'))
    if model.layers < 6:
        reasons.append("Not enough decoder layers for an internal loop window")
    if any(k.startswith("looped.") for k in model.values):
        load_reasons.append("Static repeated-layer conversion; select the original checkpoint")
    if model.values.get("split.count", 1) != 1:
        load_reasons.append("Split GGUF requires a separate adapter")
    if role == 'embedding':
        reasons.append("Embedding model; the generation test suite does not apply")
    if model.values.get(f'{model.arch}.attention.sliding_window', 0) and 'kv_iswa' not in adapter.get('cache_interfaces', []):
        reasons.append('Sliding-window state is not handled by this adapter')
    if model.stat.st_size > max_model_bytes:
        load_reasons.append("Exceeds configured model size budget")
    reasons += load_reasons
    identity = hashlib.sha256((model.header_sha256 + str(model.stat.st_size)).encode()).hexdigest()[:8]
    return Model(f"{slug(name)}-{identity}", name, str(model.path), model.arch, model.layers,
                 model.stat.st_size, model.stat.st_mtime_ns, model.header_sha256,
                 not reasons, reasons, "layer" if model.values.get(f'{model.arch}.expert_count', 0) else "block",
                 adapter.get('tier', 'unadapted'), role, adapter or (decisions or {}).get(model.arch, {}),
                 [name], not load_reasons, load_reasons)


def discover(roots=None, max_model_bytes=80 * 1024**3, binary=None):
    roots = roots if roots is not None else [Path.home() / '.lmstudio/models',
        Path(os.environ.get('OLLAMA_MODELS', str(Path.home() / '.ollama/models'))),
        Path('/usr/share/ollama/.ollama/models'), Path.home() / '.cache/huggingface/hub']
    rows, errors, seen = [], [], {}
    catalog, decisions = adapter_catalog(binary)
    for root in map(Path, roots):
        manifests = root / 'manifests'
        # Ollama tags can themselves end in .gguf; their contents are JSON manifests.
        paths = [(root, None)] if root.is_file() else [(p, None) for p in sorted(root.rglob('*.gguf'))
                                                       if manifests not in p.parents]
        if manifests.is_dir():
            for manifest in sorted(manifests.rglob('*')):
                if not manifest.is_file():
                    continue
                try:
                    content = json.loads(manifest.read_text())
                    alias = '/'.join(manifest.relative_to(manifests).parts[1:])
                    for layer in content.get('layers', []):
                        if layer.get('mediaType') == 'application/vnd.ollama.image.model':
                            digest = layer['digest']
                            if not re.fullmatch(r'sha256:[0-9a-f]{64}', digest):
                                raise ValueError('Invalid model blob digest')
                            paths.append((root / 'blobs' / digest.replace(':', '-'), alias))
                except (ValueError, OSError, KeyError) as exc:
                    errors.append({'name': manifest.name, 'error': str(exc)[:300]})
        for path, alias in paths:
            try:
                resolved = path.resolve(strict=True)
                stat = resolved.stat()
                key = (stat.st_dev, stat.st_ino)
                if key in seen:
                    if alias and alias not in seen[key].aliases:
                        seen[key].aliases.append(alias)
                    continue
                row = inspect(path, max_model_bytes, catalog, decisions, alias)
                rows.append(row)
                seen[key] = row
            except (ValueError, OSError, KeyError) as exc:
                errors.append({"name": path.name, "error": str(exc)[:300]})
    return rows, errors


@dataclass(frozen=True)
class Profile:
    passes: int = 1
    anchor: float = 0.5
    mode: str = "block"
    cache: str = "first"
    start: int = 0
    end: int = 0

    @property
    def id(self):
        if self.passes == 1:
            return "baseline"
        return f"k{self.passes}-a{self.anchor:g}-{self.mode}-{self.cache}-{self.start}-{self.end}"

    @classmethod
    def for_model(cls, model, passes=1, anchor=0.5, cache="first"):
        if isinstance(passes, bool) or not isinstance(passes, int) or not 1 <= passes <= 8:
            raise ValueError("Passes must be an integer from 1 to 8")
        if isinstance(anchor, bool) or not isinstance(anchor, (int, float)) or not 0 <= anchor <= 1:
            raise ValueError("Anchor must be a finite number between 0 and 1")
        if cache not in {"first", "last"}:
            raise ValueError("Cache policy must be first or last")
        start = model.layers // 2 - 2
        return cls(passes, float(anchor), model.mode, cache, start, start + 4)

    def environment(self):
        return {"LOOPLAB_PASSES": str(self.passes), "LOOPLAB_ANCHOR": str(self.anchor),
                "LOOPLAB_MODE": self.mode, "LOOPLAB_CACHE": self.cache,
                "LOOPLAB_START": str(self.start), "LOOPLAB_END": str(self.end)}


def storage_snapshot(state):
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    total = 0
    for path in state.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # A worker may rotate a log during this read-only snapshot.
    return {"time": time.time(), "free_bytes": shutil.disk_usage(state).free,
            "state_bytes": total,
            "model_weight_bytes_written": 0}
