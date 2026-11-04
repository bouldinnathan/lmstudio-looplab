"""Read-only GGUF discovery and conservative architecture eligibility."""
# SPDX-License-Identifier: AGPL-3.0-or-later
from dataclasses import asdict, dataclass
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

    def public(self):
        value = asdict(self)
        del value["path"]
        del value["mtime_ns"]
        return value


def inspect(path, max_model_bytes=32 * 1024**3):
    source = Path(path)
    model = GGUF(source)
    name = source.parent.name if source.suffix == ".gguf" else str(model.values.get("general.name", source.name))
    reasons = []
    if model.arch not in ARCHITECTURES:
        reasons.append(f"No validated native adapter for {model.arch or 'unknown architecture'}")
    if model.layers < 6:
        reasons.append("Not enough decoder layers for an internal loop window")
    if any(k.startswith("looped.") for k in model.values):
        reasons.append("Static repeated-layer conversion; select the original checkpoint")
    if model.values.get("split.count", 1) != 1:
        reasons.append("Split GGUF requires a separate adapter")
    if model.values.get(f"{model.arch}.attention.sliding_window", 0):
        reasons.append("Sliding-window cache semantics have not been validated")
    if "embed" in name.lower() or model.values.get(f"{model.arch}.pooling_type", -1) not in (-1, None):
        reasons.append("Embedding model; the generation test suite does not apply")
    if model.arch != "qwen3moe" and model.values.get(f"{model.arch}.expert_count", 0):
        reasons.append("This MoE variant has no routing-reuse adapter")
    if model.stat.st_size > max_model_bytes:
        reasons.append("Exceeds configured model size budget")
    identity = hashlib.sha256((model.header_sha256 + str(model.stat.st_size)).encode()).hexdigest()[:8]
    return Model(f"{slug(name)}-{identity}", name, str(model.path), model.arch, model.layers,
                 model.stat.st_size, model.stat.st_mtime_ns, model.header_sha256,
                 not reasons, reasons, "layer" if model.arch == "qwen3moe" else "block")


def discover(roots=None, max_model_bytes=32 * 1024**3):
    roots = roots or [Path.home() / ".lmstudio/models"]
    rows, errors, seen = [], [], set()
    for root in map(Path, roots):
        paths = [root] if root.is_file() else sorted(root.rglob("*.gguf"))
        for path in paths:
            try:
                resolved = path.resolve(strict=True)
                if resolved in seen:
                    continue
                seen.add(resolved)
                rows.append(inspect(path, max_model_bytes))
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
