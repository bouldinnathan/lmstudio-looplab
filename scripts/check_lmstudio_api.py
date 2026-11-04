#!/usr/bin/env python3
"""Smoke-test installed loop profiles through LM Studio's real HTTP server."""
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


def chat(base, model, messages, stream):
    headers = {'Content-Type': 'application/json'}
    if token := os.environ.get('LMSTUDIO_API_TOKEN'):
        headers['Authorization'] = 'Bearer ' + token
    body = {'model': model, 'messages': messages, 'temperature': 0, 'max_tokens': 256, 'stream': stream}
    start = time.monotonic()
    with urlopen(Request(base.rstrip('/') + '/v1/chat/completions', data=json.dumps(body).encode(),
                         headers=headers), timeout=300) as response:
        if stream:
            content, done = '', False
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                for choice in event.get('choices', []):
                    content += choice.get('delta', {}).get('content') or ''
            if not done:
                raise RuntimeError('Stream ended without [DONE]')
        else:
            content = json.load(response)['choices'][0]['message']['content']
    return {'model': model, 'content': content, 'stream': stream,
            'seconds': round(time.monotonic() - start, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:1234')
    parser.add_argument('--model', default='gpt-oss-120b-abliterated-latest')
    parser.add_argument('--passes', type=int, nargs='+', default=[2, 3, 4], choices=[2, 3, 4])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'transport': 'LM Studio HTTP API', 'checks': []}
    question = {'role': 'user', 'content': 'What is 17 multiplied by 23? Reply with only the integer.'}
    for index, passes in enumerate(args.passes):
        model = f'{args.model}-looped-{passes}x'
        result = chat(args.base_url, model, [question], stream=index > 0)
        result['passed'] = result['content'].strip() == '391'
        report['checks'].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        if index == 0:
            result = chat(args.base_url, model, [question, {'role': 'assistant', 'content': '391'},
                {'role': 'user', 'content': 'Subtract 91 from your previous answer. Reply with only the integer.'}], True)
            result['passed'] = result['content'].strip() == '300'
            report['checks'].append(result)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(result), flush=True)
    report['passed'] = all(c['passed'] for c in report['checks'])
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise SystemExit('One or more chat smoke tests failed')


if __name__ == '__main__':
    main()
