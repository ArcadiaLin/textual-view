import json
import numpy as np
import pandas as pd
import pytest
from semceb_dm.client import DecisionClient
from semceb_dm.data import load_query, reference_cardinality, render
from semceb_dm.estimators import sampled_indices, bilinear_total, q_error
from semceb_dm.runner import ROOT


def test_pair_space_is_ordered_with_diagonal():
    pairs = sampled_indices(3, 3, 9, 42)
    assert set(pairs) == {(i,j) for i in range(3) for j in range(3)}
    assert pairs == sampled_indices(3,3,9,42)
    assert sampled_indices(0,3,2,42) == []
    with pytest.raises(ValueError): sampled_indices(3,3,0,42)


def test_aggregation_equals_explicit_pairs():
    r=np.array([[.8,.2],[.1,.9]])
    s=np.array([[.4,.6],[1.,0.],[0.,1.]])
    k=np.array([[.9,.1],[.3,.7]])
    assert bilinear_total(r,k,s) == pytest.approx(sum(a @ k @ b for a in r for b in s))


def test_response_contract():
    question={'type':'choice','criteria':{'a':'A','b':'B'}}
    assert DecisionClient._parse({'answers':{'decision':{'probabilities':{'b':.7,'a':.3}}}},question)==[.3,.7]
    with pytest.raises(ValueError): DecisionClient._parse({'answers':{'decision':{'probabilities':{'a':.3}}}},question)
    with pytest.raises(ValueError): DecisionClient._parse({'answers':{'decision':{'noul':float('nan')}}},{'type':'noul'})


def test_reference_key_and_query_rendering():
    repo=ROOT/'references/repos/SemCEB'
    for qid in range(62,102):
        q=load_query(repo,qid)
        assert reference_cardinality(repo,q,10) is not None
        assert reference_cardinality(repo,q,100) is not None
    q=load_query(repo,63)
    text=render(q,{'p1':{'description_json':'{literal}'},'p2':{'description_json':'tools'}})
    assert '{literal}' in text and 'tools' in text
    assert q_error(0,100)==100
    assert q_error(0,0)==1


def test_state_proxy_end_to_end_with_stub(tmp_path):
    from semceb_dm.estimators import state_proxy
    repo=ROOT/'references/repos/SemCEB'
    q=load_query(repo,63)
    config=json.loads((ROOT/'experiments/semceb-dm/configs/q63-states.json').read_text())
    class Stub:
        def choice(self,text,instructions,states): return [1/len(states)]*len(states)
        def noul(self,*args): return .25
    tables={q.datasets[0].table_ref:pd.DataFrame({'description_json':['x','y']})}
    result=state_proxy(q,tables,Stub(),config,tmp_path/'states.jsonl')
    assert result['population']==4
    assert result['soft_estimate']==pytest.approx(1.)
    config['left']['fields']=['undeclared_field']
    with pytest.raises(ValueError): state_proxy(q,tables,Stub(),config,tmp_path/'bad.jsonl')


def test_client_cache_and_budget(tmp_path, monkeypatch):
    import io
    monkeypatch.setenv('X-goog-api-key','test-secret-do-not-log')
    client=DecisionClient(tmp_path/'cache',tmp_path/'trace.jsonl',max_requests=1)
    class Opener:
        def open(self,req,timeout):
            return io.BytesIO(json.dumps({'answers':{'decision':{'noul':.7}},'usage':{'total_input_tokens':5}}).encode())
    client.opener=Opener()
    assert client.noul('same')==.7
    assert client.noul('same')==.7
    assert client.stats['http_attempts']==1 and client.stats['cache_hits']==1
    with pytest.raises(RuntimeError, match='budget exhausted'): client.noul('different')
    assert 'test-secret-do-not-log' not in (tmp_path/'trace.jsonl').read_text()
