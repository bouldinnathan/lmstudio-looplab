#!/usr/bin/env python3
"""Apply the small LoopLab adapter to an exact, independently installed llama.cpp."""
# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_adapters import generate

REVISION = "0893ec98f5eafad36a43a72cd1c3723d72b0d03a"
ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Patch context is ambiguous or missing: {old[:100]!r}")
    return text.replace(old, new, 1)


def apply(source: Path):
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f"Expected llama.cpp {REVISION}, found {revision}")
    header = (ROOT / "runtime/looplab.h").read_bytes()
    patch_id = hashlib.sha256(header + Path(__file__).read_bytes() + (ROOT / "runtime/auto_adapters.py").read_bytes()).hexdigest()
    marker = source / ".looplab-patch.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get("patch_sha256") == patch_id:
            return previous
        managed = previous.get("managed_files", {})
        if not managed or any(hashlib.sha256((source / p).read_bytes()).hexdigest() != digest
                              for p, digest in managed.items()):
            raise RuntimeError("Managed runtime files changed; use a fresh checkout to preserve edits")
        for name in managed:
            path = source / name
            if name == "src/looplab.h":
                path.unlink()
            else:
                path.write_bytes(subprocess.check_output(["git", "-C", str(source), "show", "HEAD:" + name]))
        marker.unlink()
    generated, catalog, decisions = generate(source)
    changes = {path: (path.read_text(), updated) for path, updated in generated.items()}

    path = source / "src/llama-graph.cpp"
    original = path.read_text()
    text = '#include "looplab.h"\n' + original
    needle = "            ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));\n            ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));"
    # Patch only the standard KV attention overload; other architectures fail preflight.
    pos = text.index("ggml_tensor * llm_graph_context::build_attn(\n        llm_graph_input_attn_kv * inp,")
    before, body = text[:pos], text[pos:]
    addition = """            if (looplab::active && !looplab::active->restore_kv.count(il)) {
                auto * saved_k = ggml_dup(ctx0, k_cur);
                auto * saved_v = ggml_dup(ctx0, v_cur);
                ggml_build_forward_expand(gf, saved_k);
                ggml_build_forward_expand(gf, saved_v);
                auto * ki = k_idxs;
                auto * vi = v_idxs;
                looplab::active->restore_kv.emplace(il, [this, mctx_cur, saved_k, saved_v, ki, vi, il]() {
                    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, saved_k, ki, il));
                    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, saved_v, vi, il));
                });
            }

"""
    if needle not in body:
        raise RuntimeError("Standard KV attention context missing")
    text = before + body.replace(needle, addition + needle, 1)
    # SWA uses the same indexed writes, with separate full and sliding-window caches.
    needle = "    // optionally store to KV cache\n    if (store_kv && k_cur) {"
    addition = """    if (looplab::active && store_kv && (k_cur || v_cur) && !looplab::active->restore_kv.count(il)) {
        auto * saved_k = k_cur ? ggml_dup(ctx0, k_cur) : nullptr;
        auto * saved_v = v_cur ? ggml_dup(ctx0, v_cur) : nullptr;
        if (saved_k) { ggml_build_forward_expand(gf, saved_k); }
        if (saved_v) { ggml_build_forward_expand(gf, saved_v); }
        auto * ki = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();
        auto * vi = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();
        looplab::active->restore_kv.emplace(il, [this, mctx_cur, saved_k, saved_v, ki, vi, il]() {
            if (saved_k) { ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, saved_k, ki, il)); }
            if (saved_v) { ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, saved_v, vi, il)); }
        });
    }

"""
    text = replace_once(text, needle, addition + needle)
    needle = "    //call early so that topk-moe can be used\n    ggml_build_forward_expand(gf, weights);"
    addition = """    if (looplab::active && looplab::active->pin_routes) {
        auto & routes = looplab::active->routes;
        auto found = routes.find(il);
        if (found == routes.end()) {
            routes.emplace(il, std::make_pair(selected_experts, weights));
        } else {
            selected_experts = found->second.first;
            weights = found->second.second;
        }
    }

"""
    text = replace_once(text, needle, addition + needle)
    changes[path] = (original, text)

    path = source / "src/llama-context.cpp"
    original = path.read_text()
    text = '#include "looplab.h"\n' + original
    needle = "    if (cparams.training) {\n        res *= 4;\n    }\n\n    return res;"
    addition = """    if (looplab::config::integer("LOOPLAB_PASSES", 1) > 1) {
        res *= 4;
    }

"""
    text = replace_once(text, needle, addition + needle)
    changes[path] = (original, text)
    # Validate every context before touching the checkout.
    for path, (_, updated) in changes.items():
        path.write_text(updated)
    shutil.copyfile(ROOT / "runtime/looplab.h", source / "src/looplab.h")
    result = {"llama_cpp_revision": revision, "patch_sha256": patch_id,
              "method": "anchored_damped_euler", "architectures": sorted(catalog), "adapters": catalog, "adapter_rejections": decisions,
              "cache": ["first", "last"], "moe_routing": "pinned_ids_and_weights", "version": 2}
    result["managed_files"] = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [*changes, source / "src/looplab.h"]}
    marker.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(apply(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / ".runtime/llama.cpp")), indent=2))
