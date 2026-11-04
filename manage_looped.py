#!/usr/bin/env python3
"""Inspect physical storage or safely undo one conversion.

  python manage_looped.py status
  python manage_looped.py rollback deepcoder-14b-looped-2x-mid4
  python manage_looped.py rollback deepcoder-14b-looped-2x-mid4 --apply

Rollback defaults to a preview. It checks ownership of the generated paths,
unloads only the selected variant, removes its LM Studio link and generated
file, and retains the audit manifest. Original models are never deleted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from loop_gguf import GGUF, btrfs_usage, free_bytes


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"


def records():
    for path in sorted(MODELS.glob("*/*.gguf.json")):
        yield path, json.loads(path.read_text())


def status():
    items = []
    for path, record in records():
        output = Path(record['output'])
        row = {'identifier': record['identifier'], 'status': record['status'],
               'source': record['source'], 'output': str(output),
               'original_layers': record['original_layers'],
               'effective_layers': record['effective_layers'],
               'window_zero_based': record['window'], 'passes': record['passes']}
        if output.is_file():
            row.update({'logical_bytes': output.stat().st_size, **btrfs_usage(output)})
        row['lmstudio_link_present'] = Path(record['lmstudio_link']).is_symlink()
        row['validation'] = record.get('validation', {})
        items.append(row)
    result = {'checked_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'filesystem_free_bytes': free_bytes(ROOT),
              'logical_bytes': sum(x.get('logical_bytes', 0) for x in items),
              'exclusive_extent_bytes': sum(x.get('exclusive_extent_bytes', 0) for x in items),
              'models': items,
              'storage_note': 'Btrfs file extents are measured here; filesystem metadata and unrelated background writes are not included in exclusive_extent_bytes.'}
    return result


def rollback(identifier, apply=False):
    matches = [(p, r) for p, r in records() if r['identifier'] == identifier]
    if len(matches) != 1:
        raise ValueError('Expected exactly one recorded conversion')
    manifest, record = matches[0]
    if record['status'] == 'rolled_back':
        return {'identifier': identifier, 'status': 'already_rolled_back'}
    output = Path(record['output'])
    if output.parent.resolve() != manifest.parent.resolve() or output.is_symlink():
        raise ValueError('Generated output path was replaced or moved')
    if output.resolve().parent.parent != MODELS.resolve():
        raise ValueError('Output is outside this project\'s managed model directory')
    if GGUF(output).header_sha256 != record['output_header_sha256']:
        raise ValueError('Generated model was modified; refusing automatic removal')
    if output.stat().st_mtime_ns != record['output_mtime_ns']:
        raise ValueError('Generated model timestamp changed; refusing automatic removal')
    if GGUF(record['source']).header_sha256 != record['source_header_sha256']:
        raise ValueError('Original model is missing or has changed')
    link = Path(record['lmstudio_link'])
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != output.resolve():
            raise ValueError('LM Studio link has changed; refusing to remove it')
    loaded = json.loads(subprocess.run(['lms', 'ps', '--json'], check=True, capture_output=True, text=True).stdout)
    unload = []
    for item in loaded:
        candidate = Path.home() / '.lmstudio/models' / item['path']
        if candidate.resolve() == output.resolve():
            if item.get('status') != 'idle' or item.get('queued', 0):
                raise ValueError('Selected variant is busy; wait for generation to finish')
            unload.append(item['identifier'])
    plan = {'identifier': identifier, 'unload': unload,
            'remove_link': str(link), 'remove_generated_file': str(output),
            'preserve_original': record['source'], 'apply': apply}
    if not apply:
        return plan
    for name in unload:
        subprocess.run(['lms', 'unload', name], check=True, capture_output=True, text=True)
    if link.is_symlink():
        link.unlink()
        if not any(link.parent.iterdir()):
            link.parent.rmdir()
    output.unlink()
    record['status'] = 'rolled_back'
    record['rolled_back_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    stat = sub.add_parser('status', help='Report logical and exclusive physical storage')
    stat.add_argument('--output', type=Path, help='Optionally save the JSON report')
    undo = sub.add_parser('rollback', help='Preview or undo one generated variant')
    undo.add_argument('identifier')
    undo.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.command == 'status':
        result = status()
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + '\n')
    else:
        result = rollback(args.identifier, args.apply)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
