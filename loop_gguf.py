#!/usr/bin/env python3
"""Create reversible, reflink-backed experimental repeated-layer GGUF variants.

The source is always opened read-only. This is static layer repetition, not a
training-free damped loop algorithm or shared in-memory model weights.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass


FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
LAYER_RE = re.compile(r"^blk\.(\d+)\.(.+)$")
SUPPORTED_ARCHITECTURES = {"llama", "qwen2", "qwen3", "qwen3moe"}
PAGE = 4096
FICLONERANGE = 0x4020940D


def align(n: int, alignment: int) -> int:
    return (n + alignment - 1) // alignment * alignment


def pack_string(value: str) -> bytes:
    b = value.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def read_exact(f, count: int) -> bytes:
    b = f.read(count)
    if len(b) != count:
        raise ValueError("Truncated GGUF header")
    return b


def read_string(f) -> str:
    length = struct.unpack("<Q", read_exact(f, 8))[0]
    if length > 512 * 1024 * 1024:
        raise ValueError("Unreasonably large GGUF string")
    return read_exact(f, length).decode("utf-8")


def read_value(f, kind: int, decode=True):
    if kind in FORMATS:
        fmt = "<" + FORMATS[kind]
        value = struct.unpack(fmt, read_exact(f, struct.calcsize(fmt)))[0]
        return value if decode else None
    if kind == 8:
        value = read_string(f)
        return value if decode else None
    if kind == 9:
        subtype, count = struct.unpack("<IQ", read_exact(f, 12))
        if subtype == 9 or count > 10_000_000:
            raise ValueError("Unsupported GGUF array")
        keep = decode and count <= 256
        if subtype in FORMATS and not keep:
            read_exact(f, count * struct.calcsize("<" + FORMATS[subtype]))
            return {"array_type": subtype, "count": count}
        values = [read_value(f, subtype, keep) for _ in range(count)]
        return values if keep else {"array_type": subtype, "count": count}
    raise ValueError(f"Unknown GGUF metadata type {kind}")


def encode_value(kind: int, value) -> bytes:
    if kind in FORMATS:
        return struct.pack("<" + FORMATS[kind], value)
    if kind == 8:
        return pack_string(value)
    raise ValueError("Only scalar metadata edits are supported")


@dataclass
class Field:
    key: str
    kind: int
    value: object
    raw: bytes


@dataclass
class Tensor:
    name: str
    dims: tuple[int, ...]
    kind: int
    offset: int
    span: int = 0

    def descriptor(self, name=None, offset=None) -> bytes:
        return (pack_string(name if name is not None else self.name)
                + struct.pack("<I", len(self.dims))
                + struct.pack("<" + "Q" * len(self.dims), *self.dims)
                + struct.pack("<IQ", self.kind, self.offset if offset is None else offset))


class GGUF:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.stat = self.path.stat()
        self.fields: list[Field] = []
        self.tensors: list[Tensor] = []
        with self.path.open("rb") as f:
            if read_exact(f, 4) != b"GGUF":
                raise ValueError("Not a little-endian GGUF")
            self.version, tensor_count, field_count = struct.unpack("<IQQ", read_exact(f, 20))
            if self.version not in (2, 3) or tensor_count > 1_000_000 or field_count > 100_000:
                raise ValueError("Unsupported GGUF header")
            for _ in range(field_count):
                begin = f.tell()
                key = read_string(f)
                kind = struct.unpack("<I", read_exact(f, 4))[0]
                value = read_value(f, kind)
                end = f.tell()
                f.seek(begin)
                raw = read_exact(f, end - begin)
                self.fields.append(Field(key, kind, value, raw))
            self.values = {v.key: v.value for v in self.fields}
            if len(self.values) != len(self.fields):
                raise ValueError("Duplicate metadata keys")
            self.alignment = int(self.values.get("general.alignment", 32))
            if self.alignment <= 0 or self.alignment & (self.alignment - 1):
                raise ValueError("Invalid GGUF alignment")
            for _ in range(tensor_count):
                name = read_string(f)
                ndims = struct.unpack("<I", read_exact(f, 4))[0]
                if not 1 <= ndims <= 4:
                    raise ValueError("Unsupported tensor dimensions")
                dims = struct.unpack("<" + "Q" * ndims, read_exact(f, 8 * ndims))
                kind, offset = struct.unpack("<IQ", read_exact(f, 12))
                self.tensors.append(Tensor(name, dims, kind, offset))
            self.header_end = f.tell()
            self.data_offset = align(self.header_end, self.alignment)
            f.seek(0)
            self.header_sha256 = hashlib.sha256(read_exact(f, self.data_offset)).hexdigest()
        if not self.tensors or len({t.name for t in self.tensors}) != len(self.tensors):
            raise ValueError("Empty model or duplicate tensor names")
        if [t.offset for t in self.tensors] != sorted(t.offset for t in self.tensors):
            raise ValueError("Tensor descriptors must follow storage order")
        data_size = self.stat.st_size - self.data_offset
        for i, tensor in enumerate(self.tensors):
            stop = self.tensors[i + 1].offset if i + 1 < len(self.tensors) else data_size
            tensor.span = stop - tensor.offset
            if tensor.span <= 0 or tensor.offset % self.alignment:
                raise ValueError("Invalid tensor storage layout")
        self.arch = str(self.values.get("general.architecture", ""))
        self.layers = int(self.values.get(f"{self.arch}.block_count", 0))

    def inspect(self):
        layers = {}
        for tensor in self.tensors:
            match = LAYER_RE.match(tensor.name)
            if match:
                layers.setdefault(int(match[1]), []).append(tensor)
        return {
            "source": str(self.path), "architecture": self.arch,
            "layers": self.layers, "tensor_count": len(self.tensors),
            "size_bytes": self.stat.st_size, "data_offset": self.data_offset,
            "data_offset_mod_4096": self.data_offset % PAGE,
            "end_mod_4096": self.stat.st_size % PAGE,
            "layer_layout": [{"layer": i, "start": ts[0].offset,
                "absolute_start_mod_4096": (self.data_offset + ts[0].offset) % PAGE,
                "bytes": sum(t.span for t in ts), "count": len(ts),
                "bytes_mod_4096": sum(t.span for t in ts) % PAGE}
                for i, ts in layers.items()],
            "architecture_fields": {k: v for k, v in self.values.items() if k.startswith(self.arch + ".")},
        }


def make_field(key, kind, value):
    return Field(key, kind, value, pack_string(key) + struct.pack("<I", kind) + encode_value(kind, value))


def make_plan(model: GGUF, start: int, end: int, passes: int, name: str):
    if model.arch not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Architecture {model.arch!r} is not supported by this converter")
    if any(f.key.startswith("looped.") for f in model.fields):
        raise ValueError("Source is already a looped conversion")
    if model.values.get("split.count", 1) != 1:
        raise ValueError("Convert only a complete single-file GGUF")
    if not 0 < start < end < model.layers or not 2 <= passes <= 4:
        raise ValueError("Use an internal layer window and between 2 and 4 passes")
    for field in model.fields:
        if field.key.startswith(model.arch + ".") and field.kind == 9:
            raise ValueError(f"Architecture array requires explicit remapping: {field.key}")
    if model.values.get(f"{model.arch}.attention.sliding_window", 0):
        raise ValueError("Sliding-window architectures are not enabled in this converter")
    layer_names = {i: set() for i in range(model.layers)}
    for tensor in model.tensors:
        match = LAYER_RE.match(tensor.name)
        if match:
            if int(match[1]) not in layer_names:
                raise ValueError("Layer index exceeds block_count")
            layer_names[int(match[1])].add(match[2])
    if not all(layer_names.values()) or any(layer_names[i] != layer_names[start] for i in range(start, end)):
        raise ValueError("Loop window has missing or heterogeneous layer tensors")
    width = end - start
    extra = width * (passes - 1)
    layer_map = list(range(end)) + list(range(start, end)) * (passes - 1) + list(range(end, model.layers))
    entries = []
    for tensor in model.tensors:
        match = LAYER_RE.match(tensor.name)
        new_name = tensor.name
        if match and int(match[1]) >= end:
            new_name = f"blk.{int(match[1]) + extra}.{match[2]}"
        entries.append((tensor, new_name, tensor.offset))
    offset = align(model.stat.st_size - model.data_offset, model.alignment)
    # Tensor lookup is by name. Keeping original tensors in storage order lets
    # the complete original payload stay reflinked, with repeats appended.
    appended = []
    for loop in range(1, passes):
        for tensor in model.tensors:
            match = LAYER_RE.match(tensor.name)
            if match and start <= int(match[1]) < end:
                new_name = f"blk.{int(match[1]) + loop * width}.{match[2]}"
                entries.append((tensor, new_name, offset))
                appended.append((tensor, offset))
                offset += align(tensor.span, model.alignment)
    if len({e[1] for e in entries}) != len(entries):
        raise ValueError("Output tensor names collide")
    updates = {
        "general.name": (8, name),
        "general.description": (8, "Experimental static repetition of an internal layer block; not trained for looping. Original quantized weights and chat template preserved. Separate KV slots and memory allocations per repeated layer."),
        f"{model.arch}.block_count": (4, model.layers + extra),
        "looped.method": (8, "static_block_repetition"),
        "looped.experimental": (7, True),
        "looped.passes": (4, passes),
        "looped.window_start": (4, start),
        "looped.window_end_exclusive": (4, end),
        "looped.original_block_count": (4, model.layers),
        "looped.original_header_sha256": (8, model.header_sha256),
        "looped.layer_map": (8, json.dumps(layer_map)),
    }
    if "general.parameter_count" in model.values:
        updates["general.parameter_count"] = (10, sum(math.prod(t.dims) for t, _, _ in entries))
    fields = [make_field(f.key, *updates.pop(f.key)) if f.key in updates else f for f in model.fields
              if f.key not in {"split.no", "split.count", "split.tensors.count"}]
    fields.extend(make_field(k, *v) for k, v in updates.items())
    prefix = b"GGUF" + struct.pack("<IQQ", model.version, len(entries), len(fields) + 1)
    descriptors = b"".join(t.descriptor(n, o) for t, n, o in entries)
    raw = prefix + b"".join(f.raw for f in fields)
    padding_empty = make_field("looped.header_padding", 8, "").raw
    header_len = len(raw) + len(padding_empty) + len(descriptors)
    data_start = header_len + ((model.data_offset - header_len) % PAGE)
    if data_start % model.alignment:
        raise ValueError("Source alignment does not permit a shared storage layout")
    pad = make_field("looped.header_padding", 8, " " * (data_start - header_len)).raw
    header = raw + pad + descriptors
    assert len(header) == data_start
    ranges = [(model.data_offset, data_start, model.stat.st_size - model.data_offset)]
    ranges.extend((model.data_offset + t.offset, data_start + dst, t.span) for t, dst in appended)
    literal_bytes = len(header)
    clone_bytes = 0
    for src, dst, length in ranges:
        if src % PAGE != dst % PAGE:
            raise ValueError(f"Unaligned weight range would require a large copy: source {src}, destination {dst}. Choose another loop window.")
        head = min(length, (-src) % PAGE)
        bulk = (length - head) // PAGE * PAGE
        clone_bytes += bulk
        literal_bytes += length - bulk
    return {
        "header": header, "entries": entries, "ranges": ranges,
        "source": str(model.path), "display_name": name, "architecture": model.arch,
        "original_layers": model.layers, "effective_layers": model.layers + extra,
        "window": [start, end], "passes": passes, "layer_map": layer_map,
        "source_bytes": model.stat.st_size, "output_bytes": data_start + offset,
        "reflink_bytes": clone_bytes, "literal_write_bytes": literal_bytes,
        "source_header_sha256": model.header_sha256,
    }


def public_plan(plan):
    return {k: v for k, v in plan.items() if k not in {"header", "entries", "ranges"}}


def free_bytes(path):
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def btrfs_usage(path):
    result = subprocess.run(["btrfs", "filesystem", "du", "--raw", "-s", str(path)], capture_output=True, text=True, check=True)
    row = result.stdout.strip().splitlines()[-1].split(maxsplit=3)
    return {"total_extent_bytes": int(row[0]), "exclusive_extent_bytes": int(row[1]),
            "shared_extent_bytes": int(row[2])}


def validate_loader(path, library):
    class Params(ctypes.Structure):
        _fields_ = [("no_alloc", ctypes.c_bool), ("ctx", ctypes.c_void_p)]
    lib = ctypes.CDLL(str(library))
    lib.gguf_init_from_file.argtypes = [ctypes.c_char_p, Params]
    lib.gguf_init_from_file.restype = ctypes.c_void_p
    lib.gguf_free.argtypes = [ctypes.c_void_p]
    context = lib.gguf_init_from_file(os.fsencode(path), Params(True, None))
    if not context:
        raise ValueError("Installed GGUF loader rejected the generated file")
    lib.gguf_free(context)


def copy_literal(src_fd, dst_fd, src, dst, length):
    while length:
        block = os.pread(src_fd, min(length, 1024 * 1024), src)
        if not block:
            raise ValueError("Source shortened during conversion")
        done = 0
        while done < len(block):
            done += os.pwrite(dst_fd, block[done:], dst + done)
        src += len(block)
        dst += len(block)
        length -= len(block)


def convert(source, output, start, end, passes, name, library,
            max_literal=64 * 1024**2, max_exclusive=128 * 1024**2,
            min_free=64 * 1024**3):
    model = GGUF(source)
    plan = make_plan(model, start, end, passes, name)
    output = Path(output).absolute()
    manifest = output.with_suffix(output.suffix + ".json")
    if output.exists() or output.is_symlink() or manifest.exists():
        raise FileExistsError("Output or conversion manifest already exists")
    if plan["literal_write_bytes"] > max_literal:
        raise ValueError("Planned physical writes exceed the storage cap")
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.statvfs(output.parent).f_frsize != PAGE:
        raise ValueError("This converter requires a 4096-byte filesystem block size")
    before_free = free_bytes(output.parent)
    if before_free < min_free + max_literal:
        raise ValueError("Insufficient free-space reserve")
    # Also verifies the destination filesystem supports Btrfs accounting.
    btrfs_usage(model.path)
    fd, temporary_name = tempfile.mkstemp(prefix="." + output.name + ".", suffix=".partial", dir=output.parent)
    temporary = Path(temporary_name)
    published = False
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w+b", buffering=0) as dest, model.path.open("rb", buffering=0) as src:
            dest.write(plan["header"])
            dest.truncate(plan["output_bytes"])
            last_update = time.monotonic()
            done_bytes = 0
            for source_offset, dest_offset, length in plan["ranges"]:
                head = min(length, (-source_offset) % PAGE)
                copy_literal(src.fileno(), dest.fileno(), source_offset, dest_offset, head)
                source_offset += head
                dest_offset += head
                length -= head
                bulk = length // PAGE * PAGE
                while bulk:
                    amount = min(bulk, 1024**3)
                    # No fallback to deep copying if reflink fails.
                    fcntl.ioctl(dest.fileno(), FICLONERANGE,
                                struct.pack("<qQQQ", src.fileno(), source_offset, amount, dest_offset))
                    source_offset += amount
                    dest_offset += amount
                    length -= amount
                    bulk -= amount
                    done_bytes += amount
                    if free_bytes(output.parent) < min_free:
                        raise ValueError("Free-space reserve reached; conversion cancelled")
                    if time.monotonic() - last_update > 10:
                        print(json.dumps({"event": "reflink_progress", "bytes": done_bytes}), flush=True)
                        last_update = time.monotonic()
                copy_literal(src.fileno(), dest.fileno(), source_offset, dest_offset, length)
            os.fsync(dest.fileno())
        validate_loader(temporary, library)
        converted = GGUF(temporary)
        if converted.layers != plan["effective_layers"]:
            raise ValueError("Output block_count mismatch")
        if converted.values.get("tokenizer.chat_template") != model.values.get("tokenizer.chat_template"):
            raise ValueError("Chat template changed")
        # Check every tensor's identity and both ends of its storage region.
        # Most bytes are kernel-cloned, not copied through userspace.
        with model.path.open("rb") as src, temporary.open("rb") as dst:
            for original, new_name, new_offset in plan["entries"]:
                count = min(4096, original.span)
                for inner_offset in {0, max(0, original.span - count)}:
                    a = os.pread(src.fileno(), count, model.data_offset + original.offset + inner_offset)
                    b = os.pread(dst.fileno(), count, converted.data_offset + new_offset + inner_offset)
                    if a != b:
                        raise ValueError(f"Tensor verification failed: {new_name}")
        current = model.path.stat()
        if (current.st_ino, current.st_size, current.st_mtime_ns) != (model.stat.st_ino, model.stat.st_size, model.stat.st_mtime_ns):
            raise ValueError("Source changed during conversion")
        if GGUF(model.path).header_sha256 != model.header_sha256:
            raise ValueError("Source header changed during conversion")
        usage = btrfs_usage(temporary)
        if usage["exclusive_extent_bytes"] > max_exclusive:
            raise ValueError("Actual exclusive storage exceeds the cap")
        # Publishing via link fails rather than overwriting an existing path.
        os.link(temporary, output)
        published = True
        temporary.unlink()
        record = {**public_plan(plan), "output": str(output), "status": "converted",
            "method": "static_block_repetition", "experimental": True,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_mtime_ns": model.stat.st_mtime_ns,
            "output_header_sha256": converted.header_sha256,
            "storage": usage, "filesystem_free_before": before_free,
            "filesystem_free_after": free_bytes(output.parent),
            "validation": {"installed_gguf_loader": "passed", "all_tensor_edge_samples": "passed", "source_unchanged": True},
            "rollback": "Unload and unregister this variant, then delete only this output and its manifest; select the original model."}
        with manifest.open("x") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
        return record
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        if published:
            output.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Read only the GGUF header and tensor directory")
    inspect.add_argument("source", type=Path)
    for command in ["plan", "convert"]:
        convert_parser = sub.add_parser(command, help="Plan or create a separate repeated-layer variant")
        convert_parser.add_argument("source", type=Path)
        convert_parser.add_argument("--start", type=int, required=True, help="First repeated layer, zero-based")
        convert_parser.add_argument("--end", type=int, required=True, help="Exclusive end of the repeated window")
        convert_parser.add_argument("--passes", type=int, default=2)
        convert_parser.add_argument("--name", required=True, help="Display name, preferably including Looped and Experimental")
        if command == "convert":
            convert_parser.add_argument("--output", type=Path, required=True)
            convert_parser.add_argument("--loader-library", type=Path, required=True)
            convert_parser.add_argument("--max-literal-mib", type=int, default=64)
            convert_parser.add_argument("--max-exclusive-mib", type=int, default=128)
            convert_parser.add_argument("--min-free-gib", type=int, default=64)
    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps(GGUF(args.source).inspect(), indent=2))
    elif args.command == "plan":
        print(json.dumps(public_plan(make_plan(GGUF(args.source), args.start, args.end, args.passes, args.name)), indent=2))
    else:
        result = convert(args.source, args.output, args.start, args.end, args.passes, args.name,
                         args.loader_library, args.max_literal_mib * 1024**2,
                         args.max_exclusive_mib * 1024**2, args.min_free_gib * 1024**3)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
