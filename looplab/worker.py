"""Own one pinned llama-server process. Models and profiles never share KV state."""
# SPDX-License-Identifier: AGPL-3.0-or-later
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import secrets
import resource
import shutil
import socket
import subprocess
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from .models import ROOT, Profile, resource_reason, runtime_binary

MIN_FREE_BYTES = 2 * 1024**3
MAX_LOG_BYTES = 8 * 1024**2
MAX_LOG_FILES = 64


class Worker:
    def __init__(self, state, binary=None, context=2048, gpu_layers=999, timeout=300):
        self.state = Path(state)
        self.binary = Path(binary or runtime_binary()).resolve()
        self.context = context
        self.gpu_layers = gpu_layers
        self.timeout = timeout
        self.lock = threading.RLock()
        self.process = None
        self.log_thread = None
        self.lease = None
        self.key = None
        self.token = secrets.token_urlsafe(24)
        self.base = None
        self.last_used = time.monotonic()

    def cpu_mode(self, model=None):
        return self.gpu_layers == 0 or (model is not None and model.size_bytes > 32 * 1024**3)

    def build_info(self, model=None):
        marker = self.binary.parent / "looplab-build.json"
        if not self.binary.is_file() or not marker.is_file():
            raise RuntimeError("LoopLab runtime is not built. Run python scripts/build_runtime.py")
        info = json.loads(marker.read_text())
        if info.get("method") != "anchored_damped_euler" or info.get("version") not in (1, 2):
            raise RuntimeError("Runtime does not declare the required LoopLab capabilities")
        return {**info, "execution_policy": {"context": self.context, "gpu_layers": self.gpu_layers,
                "kv_type": "f16", "flash_attention": False, "vulkan_fusion": False,
                "prefix_cache": False, "swa_full": info.get('version') == 2,
                "cpu_above_gib": 32, "runner_version": "0.2.0",
                **({'cpu_offload_disabled': True, 'auto_fit': False} if self.cpu_mode(model) else {})}}

    def _stop_process(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
            if self.log_thread:
                self.log_thread.join(timeout=10)
            self.process.stdout.close()
            self.process = None
        self.log_thread = None
        self.key = None

    def close(self):
        with self.lock:
            self._stop_process()
            if self.lease:
                fcntl.flock(self.lease, fcntl.LOCK_UN)
                self.lease.close()
                self.lease = None

    def claim(self):
        """CLI and companion instances sharing a state directory share one worker lease."""
        if self.lease:
            return
        self.state.mkdir(parents=True, exist_ok=True)
        lease = (self.state / "inference.lock").open("a+")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            raise RuntimeError("Another LoopLab worker is active. Wait for its tests or chat to finish.") from None
        self.lease = lease

    def check_storage(self):
        self.state.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.state).free < MIN_FREE_BYTES:
            raise RuntimeError("LoopLab requires at least 2 GiB of free space; stopped before inference")

    @staticmethod
    def drain_log(pipe, path):
        # Always drain the pipe, even if disk writes fail, so the child cannot deadlock.
        try:
            output = path.open("wb")
        except OSError:
            output = None
        try:
            while data := pipe.read1(64 * 1024):
                if output:
                    try:
                        if output.tell() + len(data) > MAX_LOG_BYTES:
                            output.seek(0)
                            output.truncate()
                        output.write(data)
                        output.flush()
                    except OSError:
                        output.close()
                        output = None
        finally:
            if output:
                output.close()

    def ensure(self, model, profile, diagnostic=False):
        info = self.build_info(model)
        if not model.eligible and not (diagnostic and profile.passes == 1 and model.loadable):
            raise ValueError("Model is not eligible: " + "; ".join(model.reasons))
        stat = Path(model.path).stat()
        if stat.st_size != model.size_bytes or stat.st_mtime_ns != model.mtime_ns:
            raise ValueError("Model changed since discovery; refresh the inventory")
        self.check_storage()
        self.claim()
        key = (model.id, model.header_sha256, profile, self.context, info["patch_sha256"])
        if self.key == key and self.process and self.process.poll() is None:
            return
        self._stop_process()
        if reason := resource_reason(model):
            raise MemoryError(reason)
        self.state.mkdir(parents=True, exist_ok=True)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        self.log_path = self.state / f"{model.id}-{profile.id}.log"
        logs = sorted(self.state.glob("*.log"), key=lambda p: p.stat().st_mtime)
        for old in logs[:max(0, len(logs) - MAX_LOG_FILES + 1)]:
            old.unlink()
        env = dict(os.environ)
        for name in list(env):
            if name.startswith("LOOPLAB_"):
                del env[name]
        env.update(profile.environment())
        # Fusion changes the quantized anchor identity when intermediate K/V are retained.
        # Keep the same unfused kernels for baseline, conformance probes, and chat.
        env["GGML_VK_DISABLE_FUSION"] = "1"
        env["LLAMA_API_KEY"] = self.token
        args = [str(self.binary), "-m", model.path, "--host", "127.0.0.1", "--port", str(port),
                "--alias", model.id, "-ngl", str(0 if self.cpu_mode(model) else self.gpu_layers),
                "-c", str(self.context), "-b", str(self.context), "-ub", str(self.context),
                "--parallel", "1", "--no-context-shift", "--jinja", "--flash-attn", "off",
                "--cache-type-k", "f16", "--cache-type-v", "f16"]
        if info.get('version') == 2:
            args.append('--swa-full')
        if model.role == 'embedding':
            args.append('--embedding')
        if self.cpu_mode(model):
            # -ngl 0 alone still permits automatic fitting and operation/KV offload.
            args += ['--device', 'none', '--no-op-offload', '--no-kv-offload', '--fit', 'off']
        self.log_path.write_text('')
        self.process = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # A failed experimental graph must not dump tens of GiB of mapped weights.
        if hasattr(resource, 'prlimit'):
            try:
                resource.prlimit(self.process.pid, resource.RLIMIT_CORE, (0, 0))
            except ProcessLookupError:
                pass
        self.log_thread = threading.Thread(target=self.drain_log, args=(self.process.stdout, self.log_path), daemon=True)
        self.log_thread.start()
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Runtime failed to start; inspect " + str(self.log_path))
            try:
                with self.request("/health", timeout=2) as response:
                    if response.status == 200:
                        self.key = key
                        self.last_used = time.monotonic()
                        return
            except (URLError, TimeoutError, OSError):
                pass
            time.sleep(0.15)
        self.close()
        raise TimeoutError("Runtime startup exceeded its time limit")

    def request(self, path, payload=None, timeout=None):
        self.check_storage()
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = Request(self.base + path, data=data, headers={"Content-Type": "application/json",
                          "Authorization": "Bearer " + self.token})
        self.last_used = time.monotonic()
        return urlopen(request, timeout=timeout or self.timeout)

    def json(self, path, payload=None):
        with self.request(path, payload) as response:
            return json.load(response)

    @contextmanager
    def use(self, model, profile, diagnostic=False):
        with self.lock:
            try:
                self.ensure(model, profile, diagnostic)
            except BaseException:
                self.close()
                raise
            try:
                yield self
            finally:
                self.last_used = time.monotonic()
