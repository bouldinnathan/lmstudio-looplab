# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path
import io
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from looplab.benchmark import parse_answer, equal_json, paired_pvalue, choose_recommendation, public_summary, distribution_difference
from looplab.models import Model, Profile
from looplab.server import App, handler
from looplab.worker import Worker


def test_profiles_validate_and_key_every_semantic_setting():
    m=Model('m','m','unused','qwen3moe',48,10,0,'x',True,[],'layer')
    p=Profile.for_model(m,4)
    assert p.mode=='layer' and p.start==22 and p.end==26
    assert Profile.for_model(m,2).id != p.id
    assert Profile.for_model(m,4,cache='last').id != p.id
    assert Profile.for_model(m,4,anchor=0).id != p.id
    for value in [True,0,9,2.5]:
        with pytest.raises(ValueError): Profile.for_model(m,value)
    for value in [True,float('nan'),float('inf'),-1,2]:
        with pytest.raises(ValueError): Profile.for_model(m,2,value)


def test_grading_is_typed_and_never_executes_code():
    assert parse_answer('{"answer":[1,2]}')==[1,2]
    assert not equal_json(True,1)
    assert not equal_json('2',2)
    assert parse_answer('<think>irrelevant</think>\n```json\n{"answer":2}\n```')==2
    for text in ['__import__("os").system("touch /tmp/no")','{"answer":2,"extra":3}','prefix {"answer":2}']:
        with pytest.raises((ValueError,TypeError)): parse_answer(text)


def test_holdout_requires_real_paired_evidence():
    rows=lambda values:[{'passed':v} for v in values]
    assert not choose_recommendation(rows([False]*4),rows([True]*4),True)['recommend_loop']
    assert choose_recommendation(rows([False]*8),rows([True]*8),True)['recommend_loop']
    assert not choose_recommendation(rows([False]*8),rows([True]*8),False)['recommend_loop']
    assert paired_pvalue([True]*8,[True]*8)['one_sided_sign_test_p']==1


def test_distribution_alignment_tolerates_topk_tie_order():
    a={'completion_probabilities':[{'id':1,'top_logprobs':[{'id':1,'logprob':-1.0},{'id':2,'logprob':-1.01}]}]}
    b={'completion_probabilities':[{'id':1,'top_logprobs':[{'id':2,'logprob':-1.02},{'id':1,'logprob':-1.0}]}]}
    assert abs(distribution_difference(a,b)-0.01)<1e-10


def test_public_export_omits_private_artifacts():
    report={'status':'tested','profiles':[],'token':'secret','path':'/home/private',
        'model':{'id':'public','path':'/home/private','token':'secret'},'baseline':{
        'selection':[{'passed':True,'output':'private'}],'holdout':[],'probe':{'content':'private'}}}
    rendered=json.dumps(public_summary(report))
    assert 'secret' not in rendered and 'private' not in rendered


def test_loopback_api_auth_origin_and_input_limits(tmp_path):
    app=App(tmp_path/'state', roots=[tmp_path/'models'])
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(app))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base+'/health') as r: assert json.load(r)['status']=='ok'
        with pytest.raises(HTTPError) as e: urlopen(base+'/looplab/models')
        assert e.value.code==401
        headers={'Authorization':'Bearer '+app.token}
        with urlopen(Request(base+'/looplab/models',headers=headers)) as r: assert json.load(r)['models']==[]
        with pytest.raises(HTTPError) as e:
            urlopen(Request(base+'/looplab/models',headers={**headers,'Origin':'http://evil.test'}))
        assert e.value.code==403
        with pytest.raises(HTTPError) as e:
            urlopen(Request(base+'/looplab/jobs',data=b'[]',headers=headers))
        assert e.value.code==400
        with pytest.raises(HTTPError) as e:
            urlopen(Request(base+'/v1/chat/completions',data=b'{"model":"../../etc/passwd","messages":[]}',headers=headers))
        assert e.value.code==400
    finally:
        server.shutdown();server.server_close();app.worker.close()


def test_unverified_loop_profiles_fail_closed(tmp_path):
    app=App(tmp_path/'state',roots=[tmp_path/'models'])
    m=Model('m','m','unused','qwen2',48,10,0,'x',True,[],'block')
    app.worker.build_info=lambda model=None:{'patch_sha256':'build'}
    assert app.profile(m,'recommended').passes==1
    with pytest.raises(ValueError,match='conformance'): app.profile(m,'loop2')


def test_worker_lease_prevents_competing_loads_and_recovers(tmp_path):
    first, second = Worker(tmp_path), Worker(tmp_path)
    try:
        first.claim()
        with pytest.raises(RuntimeError,match='Another LoopLab'): second.claim()
        first.close()
        second.claim()
    finally:
        first.close()
        second.close()


def test_logs_stay_bounded_and_low_disk_prevents_inference(tmp_path,monkeypatch):
    import looplab.worker as module
    monkeypatch.setattr(module,'MAX_LOG_BYTES',128*1024)
    log=tmp_path/'worker.log'
    Worker.drain_log(io.BytesIO(b'x'*(256*1024)+b'final diagnostics'),log)
    assert log.stat().st_size <= 128*1024
    assert log.read_bytes().endswith(b'final diagnostics')
    monkeypatch.setattr(module.shutil,'disk_usage',lambda p:type('Usage',(),{'free':100})())
    with pytest.raises(RuntimeError,match='2 GiB'): Worker(tmp_path).check_storage()


def test_test_record_is_invalidated_by_execution_settings(tmp_path):
    from dataclasses import asdict
    from looplab.models import atomic_json
    app=App(tmp_path/'state',roots=[tmp_path/'models'])
    m=Model('m','m','unused','qwen2',48,10,0,'hash',True,[],'block')
    app.worker.build_info=lambda model=None:{'patch_sha256':'build','execution_policy':{'context':2048}}
    report={'model':{'header_sha256':'hash','size_bytes':10},'source_mtime_ns':0,
            'runtime':app.worker.build_info(),
            'profiles':[{'passes':2,'profile':asdict(Profile.for_model(m,2)),'anchor_identity':{'passed':True}}]}
    atomic_json(app.state/'reports/m.json',report)
    assert app.profile(m,'loop2').passes==2
    m.mtime_ns=1
    with pytest.raises(ValueError,match='conformance'):app.profile(m,'loop2')
    m.mtime_ns=0
    report['profiles'][0]['profile']['cache']='last'
    atomic_json(app.state/'reports/m.json',report)
    with pytest.raises(ValueError,match='conformance'):app.profile(m,'loop2')
    report['profiles'][0]['profile']['cache']='first'
    atomic_json(app.state/'reports/m.json',report)
    app.worker.build_info=lambda model=None:{'patch_sha256':'build','execution_policy':{'context':4096}}
    with pytest.raises(ValueError,match='conformance'):app.profile(m,'loop2')
