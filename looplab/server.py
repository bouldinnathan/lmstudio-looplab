"""Authenticated loopback-only API for the LM Studio generator."""
# SPDX-License-Identifier: AGPL-3.0-or-later
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import threading
import time
from urllib.error import HTTPError
from urllib.parse import unquote

from . import __version__
from .benchmark import public_summary, test_model
from .models import Profile, atomic_json, discover, storage_snapshot
from .worker import Worker
from .explore import explore_model, VERSION as EXPLORE_VERSION


class App:
    def __init__(self, state, roots=None, binary=None, max_model_gib=80, context=2048, gpu_layers=999):
        self.state = state
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.roots = roots
        self.max_model_bytes = int(max_model_gib * 1024**3)
        self.worker = Worker(state / "workers", binary, context, gpu_layers)
        token_path = state / "token"
        if not token_path.exists():
            token_path.write_text(secrets.token_urlsafe(32))
        token_path.chmod(0o600)
        self.token = token_path.read_text().strip()
        self.models = {}
        self.errors = []
        self.jobs = {}
        self.jobs_lock = threading.RLock()
        self.refresh()

    def refresh(self):
        models, self.errors = discover(self.roots, self.max_model_bytes, self.worker.binary)
        self.models = {m.id: m for m in models}
        atomic_json(self.state / "inventory.json", {"models": [asdict(m) for m in models], "errors": self.errors})
        return {"models": [m.public() for m in models], "errors": self.errors}

    def model(self, model_id):
        if not isinstance(model_id, str):
            raise ValueError("model must be a model identifier")
        if model_id in self.models:
            return self.models[model_id]
        matches = [m for m in self.models.values() if m.name == model_id or model_id in m.aliases]
        if len(matches) == 1:
            return matches[0]
        raise ValueError("Unknown or ambiguous model. Use /looplab/models to select an exact identifier.")

    def report(self, model):
        paths = [self.state / directory / f'{model.id}.json' for directory in ('reports', 'exploration')]
        paths = [path for path in paths if path.exists()]
        return json.loads(max(paths, key=lambda p: p.stat().st_mtime_ns).read_text()) if paths else None

    def profile(self, model, requested):
        report = self.report(model)
        requested = requested or "recommended"
        if requested == "recommended":
            requested = report.get("recommended_profile", "baseline") if report else "baseline"
        if requested == "baseline":
            return Profile.for_model(model)
        if requested not in {"loop2", "loop3", "loop4"}:
            raise ValueError("profile must be recommended, baseline, loop2, loop3, or loop4")
        passes = int(requested[-1])
        profile = Profile.for_model(model, passes)
        info = self.worker.build_info()
        tested = report and report.get("model", {}).get("header_sha256") == model.header_sha256
        tested = tested and report.get("model", {}).get("size_bytes") == model.size_bytes
        tested = tested and report.get("source_mtime_ns") == model.mtime_ns
        tested = tested and report.get("runtime", {}).get("patch_sha256") == info["patch_sha256"]
        tested = tested and report.get("runtime", {}).get("execution_policy") == info.get("execution_policy")
        tested = tested and any(p["passes"] == passes and p.get("profile") == asdict(profile)
                                and p.get("anchor_identity", {}).get("passed")
                                and (report.get('suite') != EXPLORE_VERSION or p.get('loop_smoke', {}).get('passed'))
                                for p in report.get("profiles", []))
        if not tested:
            raise ValueError("This loop profile has no current conformance pass. Run Auto-test first.")
        return profile

    def busy(self):
        with self.jobs_lock:
            return any(j["status"] in {"queued", "running"} for j in self.jobs.values())

    def all_jobs(self):
        with self.jobs_lock:
            return [self.job(j) for j in self.jobs]

    def start_job(self, selection, quick=False, explore=False):
        with self.jobs_lock:
            if self.busy():
                raise ValueError("An evaluation is already running; check its progress or cancel it")
            self.refresh()  # Pick up models added since the companion started.
            models = list(self.models.values()) if selection == "all" else [self.model(selection)]
            models.sort(key=lambda m: m.size_bytes)
            with self.worker.lock:
                self.worker.claim()
            job_id = secrets.token_hex(8)
            cancel = threading.Event()
            job = {"id": job_id, "status": "queued", "progress": "Queued", "reports": [], "cancel": cancel}
            self.jobs[job_id] = job
            def progress(message):
                job["progress"] = message
            def run():
                job["status"] = "running"
                try:
                    for model in models:
                        if cancel.is_set():
                            break
                        if explore:
                            report = explore_model(self.worker, model, self.state, cancel=cancel, progress=progress)
                        else:
                            report = test_model(self.worker, model, self.state, quick=quick, cancel=cancel, progress=progress)
                        job["reports"].append(public_summary(report))
                    job["status"] = "cancelled" if cancel.is_set() else "complete"
                    job["progress"] = "Finished"
                except Exception as exc:
                    job.update(status="failed", progress=str(exc))
                finally:
                    self.worker.close()
            threading.Thread(target=run, name="looplab-evaluation", daemon=True).start()
            return self.job(job_id)

    def job(self, job_id):
        if job_id not in self.jobs:
            raise ValueError("Unknown job")
        return {k: v for k, v in self.jobs[job_id].items() if k != "cancel"}


def handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "LoopLab/" + __version__

        def log_message(self, fmt, *args):
            # No conversation bodies, credentials, or URLs in ordinary logs.
            pass

        def send_json(self, status, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self):
            if self.headers.get("Origin"):
                self.send_json(403, {"error": "Browser-origin requests are disabled"})
                return False
            wanted = "Bearer " + app.token
            if not secrets.compare_digest(self.headers.get("Authorization", ""), wanted):
                self.send_json(401, {"error": "A LoopLab local access token is required"})
                return False
            return True

        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, {"status": "ok", "version": __version__})
                return
            if not self.authorized():
                return
            try:
                if self.path == "/looplab/models":
                    result = {"models": [m.public() for m in app.models.values()], "errors": app.errors}
                elif self.path == "/v1/models":
                    result = {"object": "list", "data": [{"id": m.id, "object": "model", "owned_by": "local"}
                              for m in app.models.values() if m.eligible]}
                elif self.path == "/looplab/status":
                    result = {"storage": storage_snapshot(app.state), "busy": app.busy(),
                              "jobs": app.all_jobs()}
                elif self.path.startswith("/looplab/jobs/"):
                    result = app.job(self.path.rsplit("/", 1)[1])
                elif self.path.startswith("/looplab/reports/"):
                    report = app.report(app.model(unquote(self.path.rsplit("/", 1)[1])))
                    result = public_summary(report) if report else {"status": "untested"}
                else:
                    self.send_json(404, {"error": "Unknown endpoint"})
                    return
                self.send_json(200, result)
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)})

        def do_POST(self):
            if not self.authorized():
                return
            streaming = False
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2*1024*1024:
                    self.send_json(413, {"error": "Expected a JSON request of at most 2 MiB"})
                    return
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/looplab/refresh":
                    if app.busy():
                        raise ValueError("Cannot refresh during evaluation")
                    self.send_json(200, app.refresh())
                elif self.path == "/looplab/jobs":
                    if any(type(body.get(k, False)) is not bool for k in ('quick', 'explore')):
                        raise ValueError("quick and explore must be booleans")
                    self.send_json(202, app.start_job(body.get("model", "all"), body.get("quick", False), body.get('explore', False)))
                elif self.path.endswith("/cancel") and self.path.startswith("/looplab/jobs/"):
                    job_id = self.path.split("/")[-2]
                    app.job(job_id)
                    app.jobs[job_id]["cancel"].set()
                    self.send_json(200, {"status": "cancellation_requested"})
                elif self.path == "/v1/chat/completions":
                    if app.busy():
                        self.send_json(409, {"error": "Evaluation is using the worker. Wait or cancel the job."})
                        return
                    model = app.model(body.get("model"))
                    profile = app.profile(model, body.pop("looplab_profile", "recommended"))
                    if not isinstance(body.get("messages"), list) or not body["messages"]:
                        raise ValueError("messages must be a nonempty array")
                    # Only documented inference parameters reach the native worker.
                    allowed = {"model", "messages", "stream", "temperature", "top_p", "max_tokens", "seed",
                               "tools", "tool_choice", "response_format", "stop", "frequency_penalty", "presence_penalty"}
                    payload = {k: v for k, v in body.items() if k in allowed}
                    payload["model"] = model.id
                    payload["cache_prompt"] = False
                    payload.setdefault("max_tokens", 512)
                    if type(payload["max_tokens"]) is not int or not 1 <= payload["max_tokens"] <= app.worker.context:
                        raise ValueError("max_tokens exceeds the configured context budget")
                    with app.worker.use(model, profile):
                        with app.worker.request("/v1/chat/completions", payload) as response:
                            if payload.get("stream"):
                                streaming = True
                                self.send_response(200)
                                self.send_header("Content-Type", "text/event-stream")
                                self.send_header("Cache-Control", "no-cache")
                                self.send_header("X-LoopLab-Passes", str(profile.passes))
                                self.end_headers()
                                for line in response:
                                    self.wfile.write(line)
                                    self.wfile.flush()
                            else:
                                self.send_json(200, json.load(response))
                else:
                    self.send_json(404, {"error": "Unknown endpoint"})
            except (BrokenPipeError, ConnectionResetError):
                return
            except (ValueError, KeyError, TypeError) as exc:
                if not streaming:
                    self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                if not streaming:
                    self.send_json(502, {"error": f"{type(exc).__name__}: {exc}"})
    return Handler


def serve(app, port=7341):
    server = ThreadingHTTPServer(("127.0.0.1", port), handler(app))
    print(f"LoopLab listening at http://127.0.0.1:{server.server_port}; token stored in {app.state / 'token'}", flush=True)
    stop = threading.Event()
    def idle_unload():
        while not stop.wait(30):
            if not app.busy() and time.monotonic()-app.worker.last_used > 300:
                if app.worker.lock.acquire(blocking=False):
                    try:
                        app.worker.close()
                    finally:
                        app.worker.lock.release()
    threading.Thread(target=idle_unload, daemon=True).start()
    try:
        server.serve_forever()
    finally:
        stop.set()
        for job in app.jobs.values():
            job["cancel"].set()
        app.worker.close()
        server.server_close()
