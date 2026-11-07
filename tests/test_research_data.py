# SPDX-License-Identifier: AGPL-3.0-or-later
import hashlib
import json
from pathlib import Path

import pytest

from scripts.export_research_data import Export, digest, joined_requests
from scripts.verify_research_data import verify


def test_published_research_records_reproduce_reported_results():
    result = verify(Path(__file__).resolve().parents[1] / 'reports')
    assert result['main_responses'] == 1600
    assert result['pilot_responses'] == 8
    assert result['diagnostic_answers_regraded'] == 312
    assert result['model_files_covered'] == 54
    assert result['performance_rows'] == 54


def test_export_refuses_credentials_and_unreviewed_local_paths(tmp_path):
    state = tmp_path / 'state'
    state.mkdir()
    (state / 'token').write_text('test-private-token-value')
    exporter = Export(tmp_path / 'public', state)
    for value in ({'authorization': 'secret'}, {'content': 'test-private-token-value'},
                  {'content': 'File /home/example/private.txt'}, {'password': 'example'}):
        with pytest.raises(ValueError):
            exporter.clean(value, 'record.json')
    answer = {'response': {'content': 'The answer is 42.', 'tokens': [1, 2]}}
    assert exporter.clean(answer, 'record.json') == answer
    task = {'metadata': {'config_source': '/home/example/env/lib/site-packages/lm_eval/tasks/math.yaml'}}
    assert exporter.clean(task, 'harness.json')['metadata']['config_source'] == 'lm_eval/tasks/math.yaml'
    assert exporter.redactions == [{'file': 'harness.json', 'json_pointer': '/metadata/config_source',
        'change': 'Removed local installation directory; kept package-relative task path.'}]


def test_missing_response_and_changed_payload_cannot_be_exported(tmp_path):
    messages = [{'role': 'user', 'content': 'Question'}]
    harness = {'samples': {'task': [{'doc_id': 12, 'filter': 'flexible-extract',
                                    'arguments': [[[json.dumps(messages)], {}]]}]}}
    requests = tmp_path / 'requests'
    requests.mkdir()
    with pytest.raises(ValueError, match='Missing request'):
        joined_requests(tmp_path, harness)
    record = {'payload': {'messages': messages}, 'response': {'content': 'Answer'}, 'seconds': 1.25}
    path = requests / (digest(record['payload']) + '.json')
    path.write_text(json.dumps(record))
    exported = joined_requests(tmp_path, harness)
    assert exported[0]['doc_id'] == 12
    assert exported[0]['record'] == record
    record['payload']['messages'][0]['content'] = 'Different question'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='hash mismatch'):
        joined_requests(tmp_path, harness)


def test_manifest_detects_modified_data(tmp_path):
    data = tmp_path / 'research-data'
    data.mkdir()
    content = b'{"score": 1}\n'
    (data / 'sample.json').write_bytes(content.replace(b'1', b'0'))
    (data / 'manifest.json').write_text(json.dumps({'files': {'sample.json': {
        'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}}}))
    with pytest.raises(ValueError, match='SHA256'):
        verify(tmp_path)
