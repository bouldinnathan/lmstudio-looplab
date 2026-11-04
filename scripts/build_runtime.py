#!/usr/bin/env python3
"""Build a pinned local runtime without modifying LM Studio or model weights."""
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VULKAN_REV = "9fbc2a782ee3f8a4db64e7d991e23da75e893f44"
SPIRV_REV = "71747ddd16f20e9b379ce293dba2954931d3b3c1"
spec = importlib.util.spec_from_file_location("looplab_patch", ROOT / "runtime/apply_patch.py")
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


def run(args):
    print("+ " + " ".join(map(str,args)), flush=True)
    subprocess.run(list(map(str,args)), check=True)


def checkout(url, destination, revision):
    if not destination.exists():
        run(["git", "clone", "--filter=blob:none", "--no-checkout", url, destination])
        run(["git", "-C", destination, "checkout", "--detach", revision])
    actual=subprocess.check_output(["git","-C",str(destination),"rev-parse","HEAD"],text=True).strip()
    if actual!=revision:
        raise RuntimeError(f"{destination} is at {actual}; expected {revision}. Use a fresh checkout.")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true", help="Build without Vulkan")
    p.add_argument("--jobs", type=int, default=min(8,os.cpu_count() or 2))
    args=p.parse_args()
    cmake=shutil.which("cmake") or str(ROOT/".venv/bin/cmake")
    if not Path(cmake).is_file():
        raise SystemExit("Install CMake, or run .venv/bin/pip install cmake")
    runtime=ROOT/".runtime"
    runtime.mkdir(exist_ok=True)
    cache=runtime/"llama.cpp"
    checkout("https://github.com/ggml-org/llama.cpp.git",cache,patch.REVISION)
    source=runtime/"auto/source"
    source.parent.mkdir(exist_ok=True)
    if not source.exists():
        run(["git", "clone", "--shared", "--no-checkout", cache, source])
        run(["git", "-C", source, "checkout", "--detach", patch.REVISION])
    build=runtime/"auto/build"
    info=patch.apply(source)
    flags=["-DCMAKE_BUILD_TYPE=Release","-DGGML_NATIVE=OFF","-DLLAMA_BUILD_TESTS=OFF",
           "-DLLAMA_BUILD_EXAMPLES=ON","-DLLAMA_BUILD_SERVER=ON","-DLLAMA_BUILD_APP=OFF",
           "-DLLAMA_BUILD_UI=OFF","-DLLAMA_USE_PREBUILT_UI=OFF"]
    if not args.cpu:
        if not shutil.which("glslc"):
            raise SystemExit("Vulkan build needs glslc and a Vulkan loader. Use --cpu for a CPU build.")
        checkout("https://github.com/KhronosGroup/Vulkan-Headers.git",runtime/"vulkan-headers",VULKAN_REV)
        checkout("https://github.com/KhronosGroup/SPIRV-Headers.git",runtime/"spirv-headers",SPIRV_REV)
        deps=runtime/"deps"
        run([cmake,"-S",runtime/"spirv-headers","-B",runtime/"spirv-build",f"-DCMAKE_INSTALL_PREFIX={deps}","-DSPIRV_HEADERS_ENABLE_TESTS=OFF"])
        run([cmake,"--install",runtime/"spirv-build"])
        flags += ["-DGGML_VULKAN=ON",f"-DVulkan_INCLUDE_DIR={runtime/'vulkan-headers/include'}",
                  f"-DCMAKE_PREFIX_PATH={deps}",f"-DCMAKE_CXX_FLAGS=-I{deps/'include'}"]
    else:
        flags.append("-DGGML_VULKAN=OFF")
    run([cmake,"-S",source,"-B",build,"-G","Ninja",*flags])
    run([cmake,"--build",build,"--target","llama-server","llama-perplexity","--parallel",args.jobs])
    info["backend"]="cpu" if args.cpu else "vulkan"
    info["dependencies"]={"vulkan_headers":VULKAN_REV,"spirv_headers":SPIRV_REV}
    (build/"bin/looplab-build.json").write_text(json.dumps(info,indent=2)+"\n")
    run(["g++","-std=c++17","-O2",f"-I{ROOT/'runtime'}",f"-I{source/'ggml/include'}",
         ROOT/"runtime/test_integrator.cpp","-o",runtime/"test-integrator"])
    run([runtime/"test-integrator"])
    print("Built. Run tests/native_reference.py before relying on model test results.")


if __name__=="__main__":
    main()
