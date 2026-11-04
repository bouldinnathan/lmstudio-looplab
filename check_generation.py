#!/usr/bin/env python3
"""Run a small functional generation check, saving results beside each variant.

This checks loading and a simple Python function, not general model quality.
Only temporary model instances created by this script are unloaded.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request

from manage_looped import records


def check_answer(response):
    choice = response['choices'][0]
    content = choice['message'].get('content', '')
    candidates = re.findall(r'```(?:python)?\s*\n(.*?)```', content, re.S)
    if not candidates:
        candidates = [content.strip()]
    allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
               ast.BinOp, ast.Add, ast.Name, ast.Load, ast.Constant, ast.Expr)
    for code in candidates:
        try:
            tree = ast.parse(code)
            if not all(isinstance(node, allowed) for node in ast.walk(tree)):
                continue
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                continue
            function = tree.body[0]
            if function.name != 'add' or function.decorator_list or len(function.args.args) != 2:
                continue
            namespace = {'__builtins__': {}}
            exec(compile(tree, '<generated-add-check>', 'exec'), namespace)
            cases = [(2, 3), (-2, 7), (0, 0), (1.5, 2.25), (-4, -9)]
            passed = all(namespace['add'](a, b) == a + b for a, b in cases)
            if passed:
                outside = re.sub(r'```(?:python)?\s*\n.*?```', '', content, flags=re.S).strip() if '```' in content else ''
                return {'functional_python_add': 'passed', 'cases': len(cases),
                        'completed': choice.get('finish_reason') == 'stop',
                        'only_requested_code': not bool(outside)}
        except Exception:
            continue
    return {'functional_python_add': 'failed', 'completed': choice.get('finish_reason') == 'stop'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('identifiers', nargs='+')
    parser.add_argument('--existing-response', action='store_true', help='Validate a previously saved lmstudio-smoke.json')
    args = parser.parse_args()
    available = {r['identifier']: (p, r) for p, r in records()}
    for identifier in args.identifiers:
        manifest, record = available[identifier]
        artifact = manifest.parent / 'lmstudio-smoke.json'
        if args.existing_response:
            result = json.loads(artifact.read_text())
        else:
            alias = 'loop-check-' + record['original_lmstudio_key']
            started = time.monotonic()
            load = subprocess.run(['lms', 'load', identifier, '--gpu', 'max', '--context-length', '2048',
                                   '--parallel', '1', '--identifier', alias, '--ttl', '600', '-y'],
                                  capture_output=True, text=True, timeout=180)
            (manifest.parent / 'load.log').write_text(load.stdout + load.stderr)
            if load.returncode:
                raise RuntimeError(f'{identifier} failed to load; see load.log')
            load_seconds = time.monotonic() - started
            print(json.dumps({'event': 'loaded', 'identifier': identifier, 'seconds': load_seconds}), flush=True)
            try:
                body = {'model': alias, 'messages': [{'role': 'user', 'content':
                        'Write a Python function add(a, b) that returns their sum. Output only the function.'}],
                        'temperature': 0, 'max_tokens': 512, 'stream': False}
                request = urllib.request.Request('http://127.0.0.1:1234/v1/chat/completions',
                          data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                started = time.monotonic()
                with urllib.request.urlopen(request, timeout=240) as response:
                    data = json.load(response)
                result = {'test': 'simple_python_add', 'load_seconds': load_seconds,
                          'elapsed_seconds': time.monotonic() - started, 'response': data}
            finally:
                subprocess.run(['lms', 'unload', alias], check=True, capture_output=True, text=True)
        result['checks'] = check_answer(result['response'])
        artifact.write_text(json.dumps(result, indent=2) + '\n')
        record['validation']['generation'] = result['checks']
        record['validation']['generation_artifact'] = str(artifact.resolve())
        manifest.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'identifier': identifier, 'checks': result['checks'],
                          'elapsed_seconds': result.get('elapsed_seconds')}), flush=True)
        if result['checks']['functional_python_add'] != 'passed' or not result['checks']['completed']:
            raise RuntimeError('Generation did not pass the functional completion check')


if __name__ == '__main__':
    main()
