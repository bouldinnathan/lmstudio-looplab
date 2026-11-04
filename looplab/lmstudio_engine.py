"""LM Studio Engine Protocol launcher and local transport for loop descriptors."""
# SPDX-License-Identifier: AGPL-3.0-or-later
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import signal
import sys
import threading
from urllib.error import HTTPError
from urllib.parse import urlsplit

from loop_gguf import GGUF
from .lmstudio import MARKER, digest
from .server import App

PREDICTION_PATHS = {'/completion', '/completions', '/v1/completions', '/chat/completions', '/v1/chat/completions'}
POST_PATHS = PREDICTION_PATHS | {'/tokenize', '/detokenize', '/apply-template'}
GET_PATHS = {'/health', '/props', '/slots', '/metrics', '/v1/models', '/models'}


def argument(args, *names, default=None):
    value = default
    for index, arg in enumerate(args):
        if arg in names:
            if index + 1 >= len(args):
                raise ValueError(f'Missing value for {arg}')
            value = args[index + 1]
        else:
            for name in names:
                if arg.startswith(name + '='):
                    value = arg.split('=', 1)[1]
    return value


def resolve_profile(app, path):
    path = Path(path).resolve(strict=True)
    descriptor = GGUF(path)
    if descriptor.values.get('looplab.descriptor_version') != 1:
        raise ValueError('Not a LoopLab profile descriptor')
    registry = json.loads((app.state / 'lmstudio-models.json').read_text())
    entry = registry.get('entries', {}).get(str(path))
    if registry.get('managed_by') != MARKER or not entry or digest(path) != entry['descriptor_sha256']:
        raise ValueError('LoopLab descriptor is unregistered or changed; reinstall its model entry')
    model = app.model(entry['model_id'])
    if descriptor.values['looplab.source_header_sha256'] != model.header_sha256:
        raise ValueError('The source model changed; retest and reinstall its LoopLab entries')
    profile = app.profile(model, f'loop{entry["profile"]["passes"]}')
    from dataclasses import asdict
    if asdict(profile) != entry['profile']:
        raise ValueError('LoopLab profile settings changed; reinstall its model entry')
    return model, profile


def validate_load_options(args, context):
    requested_context = argument(args, '-c', '--ctx-size')
    if requested_context is not None and int(requested_context) != context:
        raise ValueError(f'This tested LoopLab entry requires context length {context}; restore its load default')
    parallel = argument(args, '-np', '--parallel')
    if parallel is not None and int(parallel) != 1:
        raise ValueError('Tested LoopLab entries require Max Concurrent Predictions = 1')


def proxy_handler(worker, model_id, passes, api_keys=()):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_):
            pass

        def error(self, status, message):
            data = json.dumps({'error': {'message': message, 'type': 'looplab_runtime_error'}}).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(data)
            self.close_connection = True

        def forward(self):
            streaming = False
            try:
                path = urlsplit(self.path).path
                if self.headers.get('Origin'):
                    return self.error(403, 'Browser-origin requests are disabled')
                if api_keys and path != '/health' and not any(secrets.compare_digest(
                    self.headers.get('Authorization', ''), 'Bearer ' + key) for key in api_keys):
                    return self.error(401, 'The runtime access token is required')
                allowed = GET_PATHS if self.command == 'GET' else POST_PATHS
                if path not in allowed:
                    return self.error(404, 'This endpoint is not available for a fixed LoopLab profile')
                payload = None
                if self.command == 'POST':
                    if self.headers.get('Transfer-Encoding'):
                        return self.error(400, 'A Content-Length is required')
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 2 * 1024**2:
                        return self.error(413, 'Request must contain at most 2 MiB of JSON')
                    payload = json.loads(self.rfile.read(size))
                    if not isinstance(payload, dict):
                        return self.error(400, 'Expected a JSON object')
                    if path in PREDICTION_PATHS:
                        payload['cache_prompt'] = False
                        if 'model' in payload:
                            payload['model'] = model_id
                # Forward only to the owned worker, with its own credential.
                try:
                    response = worker.request(self.path, payload)
                except HTTPError as exc:
                    response = exc
                with response:
                    self.send_response(response.status)
                    self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
                    self.send_header('Connection', 'close')
                    self.send_header('X-LoopLab-Passes', str(passes))
                    self.end_headers()
                    streaming = True
                    self.close_connection = True
                    while chunk := response.read1(64 * 1024):
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (ValueError, TypeError) as exc:
                if not streaming:
                    self.error(400, str(exc))
            except Exception as exc:
                if not streaming:
                    self.error(502, str(exc))

        do_GET = forward
        do_POST = forward
    return Handler


def main():
    try:
        _main()
    except Exception as exc:
        print(f'LoopLab runtime: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1) from None


def _main():
    directory = Path(sys.argv[0]).resolve().parent
    config = json.loads((directory / 'looplab-engine.json').read_text())
    if config.get('managed_by') != MARKER:
        raise ValueError('Unrecognized runtime configuration')
    args = sys.argv[1:]
    model_path = argument(args, '-m', '--model')
    is_descriptor = False
    if model_path and Path(model_path).is_file():
        try:
            is_descriptor = GGUF(model_path).values.get('looplab.descriptor_version') == 1
        except ValueError:
            # The official engine can support layouts the LoopLab inspector
            # does not. Preserve its normal loading behavior for those files.
            pass
    if not is_descriptor:
        # Ordinary models retain the official engine and all its arguments.
        stock = directory / 'stock-llama-server'
        os.execv(str(stock), [str(stock), *args])
    host = argument(args, '--host', default='127.0.0.1')
    if host not in {'127.0.0.1', 'localhost'}:
        raise ValueError('The Engine Protocol transport must bind to loopback; use LM Studio to serve externally')
    port = int(argument(args, '--port', default='8080'))
    if not 1 <= port <= 65535:
        raise ValueError('Invalid runtime port')
    app = App(Path(config['state']), binary=Path(config['binary']), context=config['context'],
              gpu_layers=config['gpu_layers'], max_model_gib=config['max_model_gib'])
    server = None
    try:
        model, profile = resolve_profile(app, model_path)
        validate_load_options(args, app.worker.context)
        api_keys = argument(args, '--api-key', default=os.environ.get('LLAMA_API_KEY', '')).split(',')
        api_key_file = argument(args, '--api-key-file')
        if api_key_file:
            api_keys += Path(api_key_file).read_text().splitlines()
        api_keys = tuple(key.strip() for key in api_keys if key.strip())
        server = ThreadingHTTPServer(('127.0.0.1', port), proxy_handler(app.worker, model.id, profile.passes, api_keys))
        # LM Studio owns this process's lifetime; the same lease prevents its
        # worker from competing with companion inference or compatibility tests.
        def cancel_loading(*_):
            raise InterruptedError('Model loading was cancelled')
        signal.signal(signal.SIGTERM, cancel_loading)
        signal.signal(signal.SIGINT, cancel_loading)
        app.worker.ensure(model, profile)
        print(f'LoopLab active: {model.id}, {profile.passes} passes, context={app.worker.context}, '
              f'CPU={app.worker.cpu_mode(model)}, prefix_cache=false', flush=True)
        def stop(*_):
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        server.serve_forever()
    finally:
        if server:
            server.server_close()
        app.worker.close()
