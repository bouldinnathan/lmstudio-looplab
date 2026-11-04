"""Generate experimental adapters from recognizable native graph structures.

This preserves the architecture's layer code. It never relabels GGUF weights.
Unknown stateful graphs are left intact and recorded for a later repair pass.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
import hashlib
import re

VALIDATED = {"llama", "qwen2", "qwen3", "qwen3moe"}


def code_only(text):
    # Keep offsets stable while ignoring braces and identifiers inside comments/literals.
    pattern = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(pattern, lambda m: ''.join('\n' if c == '\n' else ' ' for c in m[0]), text)


def closing_brace(code, start):
    depth = 0
    for i in range(start, len(code)):
        if code[i] == '{': depth += 1
        elif code[i] == '}':
            depth -= 1
            if depth == 0: return i
    raise ValueError("Unbalanced C++ body")


def transform(text):
    code = code_only(text)
    pattern = r'llama_model_\w+::graph(?:<\w+>)?::(?:graph|build_backbone)\s*\('
    constructors = list(re.finditer(pattern, code))
    # Qwen3 shares its decoder backbone with its embedding implementation.
    backbones = [m for m in constructors if 'build_backbone' in m[0]]
    if backbones:
        constructors = backbones
    if len(constructors) != 1:
        return None, "No single standard graph constructor"
    begin = code.find('{', constructors[0].end())
    end = closing_brace(code, begin)
    constructor = code[begin:end+1]
    if re.search(r'build_inp_mem|is_recr|build_ssm|build_rwkv|build_layer_attn_linear|build_mamba|build_conv', constructor):
        return None, "Recurrent/SSM/convolution state needs an explicit snapshot adapter"
    inputs = set(re.findall(r'build_attn_inp_([a-z0-9_]+)\s*\(', constructor))
    allowed = {"kv", "kv_iswa"}
    if 'llama_model_llama::graph<embed>' in code:
        allowed.add('no_cache')
    if not inputs or not inputs <= allowed:
        return None, "Attention state is not the supported KV/SWA interface: " + ','.join(sorted(inputs))
    loops = list(re.finditer(r'for\s*\(int\s+il\s*=\s*0;\s*il\s*<\s*n_layer;\s*\+\+il\s*\)\s*\{', constructor))
    if len(loops) != 1:
        return None, "No single recognizable decoder-layer loop"
    loop = loops[0]
    start = begin + loop.start()
    body_start = begin + loop.end()
    finish = closing_brace(code, body_start-1)
    body = code[body_start:finish]
    if re.search(r'\b(?:return|break|continue)\b', body):
        return None, "Layer body has control flow requiring review"
    assignment = re.search(r'inpL\s*=\s*cur\s*;\s*$', body)
    if not assignment:
        return None, "Residual-stream input/output boundaries are not recognized"
    moe = bool(re.search(r'build_moe_ffn|n_expert', code))
    copied = text[body_start:body_start+assignment.start()] + "        return cur;\n"
    replacement = ("auto looplab_layer = [&](ggml_tensor * layer_input, int il) {\n"
                   "        inpL = layer_input;" + copied + "    };\n"
                   "    const auto looplab_config = looplab::config::read(n_layer, n_expert > 0);\n"
                   "    inpL = looplab::run(ctx0, gf, looplab_config, inpL, n_layer, n_expert > 0, looplab_layer);")
    patched = text[:start] + replacement + text[finish+1:]
    patched = '#include "looplab.h"\n' + patched
    return patched, {"family": "moe" if moe else "dense", "cache_interfaces": sorted(inputs),
                     "strategy": "preserve_native_layer_body", "source_sha256": hashlib.sha256(text.encode()).hexdigest()}


def generate(source):
    names = dict(re.findall(r'\{\s*(LLM_ARCH_\w+),\s*"([^"]+)"', (source/'src/llama-arch.cpp').read_text()))
    factory = dict(re.findall(r'case\s+(LLM_ARCH_\w+)\s*:\s*return new llama_model_(\w+)\(',
                              (source/'src/llama-model.cpp').read_text()))
    classes = {}
    for path in sorted((source/'src/models').glob('*.cpp')):
        text = path.read_text()
        for name in re.findall(r'llama_model_(\w+)::build_arch_graph\(', code_only(text)):
            classes[name] = path
    changes, catalog, decisions = {}, {}, {}
    for enum, name in sorted(names.items()):
        path = classes.get(factory.get(enum))
        if not path:
            decisions[name] = {"reason": "Native graph source not directly discoverable"}
            continue
        patched, detail = transform(path.read_text())
        relative = path.relative_to(source).as_posix()
        if patched is None:
            decisions[name] = {"source": relative, "reason": detail}
            continue
        changes[path] = patched
        catalog[name] = {**detail, "source": relative,
                         "tier": "reference_validated" if name in VALIDATED else "experimental"}
    missing = VALIDATED - catalog.keys()
    if missing:
        raise RuntimeError("Required adapters no longer match: " + ','.join(sorted(missing)))
    return changes, catalog, decisions
