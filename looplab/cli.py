# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
from pathlib import Path

from .benchmark import public_summary, test_model
from .models import atomic_json, discover, state_directory
from .server import App, serve


def main():
    parser = argparse.ArgumentParser(prog="looplab")
    parser.add_argument("--state", type=Path, default=state_directory())
    parser.add_argument("--models", type=Path, action="append", help="Model directory or single GGUF; repeatable")
    parser.add_argument("--max-model-gib", type=float, default=32)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--gpu-layers", type=int, default=999)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan")
    run = sub.add_parser("serve")
    run.add_argument("--port", type=int, default=7341)
    test = sub.add_parser("test")
    test.add_argument("model", nargs="?", default="all")
    test.add_argument("--quick", action="store_true")
    test.add_argument("--passes", type=int, nargs="+", default=[2,3,4])
    export = sub.add_parser("export-report")
    export.add_argument("input", type=Path)
    export.add_argument("output", type=Path)
    args = parser.parse_args()
    if not 128 <= args.context <= 32768 or not 0 < args.max_model_gib <= 1024:
        parser.error("Context must be 128..32768 and model budget 0..1024 GiB")
    if args.command == "export-report":
        atomic_json(args.output, public_summary(json.loads(args.input.read_text())))
        return
    if args.command == "scan":
        models, errors = discover(args.models, int(args.max_model_gib*1024**3))
        print(json.dumps({"models": [m.public() for m in models], "errors": errors}, indent=2))
        return
    app = App(args.state, args.models, args.binary, args.max_model_gib, args.context, args.gpu_layers)
    if args.command == "serve":
        try:
            serve(app, args.port)
        except KeyboardInterrupt:
            pass
    else:
        if any(p not in (2,3,4) for p in args.passes):
            parser.error("Tests support passes 2, 3, and 4")
        models = list(app.models.values()) if args.model == "all" else [app.model(args.model)]
        try:
            for model in sorted(models, key=lambda m: m.size_bytes):
                report = test_model(app.worker, model, args.state, args.passes, args.quick,
                                    progress=lambda message: print(message, flush=True))
                print(json.dumps(public_summary(report), indent=2), flush=True)
        except KeyboardInterrupt:
            print("Evaluation cancelled.", flush=True)
        finally:
            app.worker.close()


if __name__ == "__main__":
    main()
