#!/usr/bin/env python3
"""Install or remove an optional, unprivileged systemd user service."""
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MARKER = "# Managed by lmstudio-looplab/scripts/service.py\n"


def quoted(value):
    # systemd quoting is not shell quoting; percent signs expand specifiers.
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'


def run(*args):
    subprocess.run(["systemctl", "--user", *args], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["install", "remove"])
    parser.add_argument("--enable", action="store_true", help="Also start automatically at user login")
    args = parser.parse_args()
    directory = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home()/".config"))) / "systemd/user"
    target = directory/"looplab.service"
    if target.exists() and not target.read_text().startswith(MARKER):
        raise SystemExit(f"Refusing to overwrite an unrelated service: {target}")
    if args.action == "remove":
        if target.exists():
            run("disable", "--now", "looplab.service")
            target.unlink()
            run("daemon-reload")
        return
    if not (ROOT/".runtime/build/bin/looplab-build.json").exists():
        raise SystemExit("Build the native runtime first: python scripts/build_runtime.py")
    directory.mkdir(parents=True, exist_ok=True)
    target.write_text(MARKER + f"""[Unit]
Description=LoopLab local inference companion

[Service]
Type=simple
WorkingDirectory={str(ROOT).replace('%', '%%')}
ExecStart={quoted(sys.executable)} -m looplab serve
Environment=PYTHONUNBUFFERED=1
UMask=0077
Restart=on-failure
RestartSec=3
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=default.target
""")
    run("daemon-reload")
    if args.enable:
        run("enable", "looplab.service")
    run("restart", "looplab.service")
    run("is-active", "looplab.service")
    print(f"Installed {target}. Remove with: {sys.executable} scripts/service.py remove")


if __name__ == "__main__":
    main()
