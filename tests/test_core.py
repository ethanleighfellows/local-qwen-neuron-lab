from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import json
import pytest
import torch
from fastapi.testclient import TestClient
from app.steering import Code, SteeringEngine, parse_codes
from app.store import Store
from app.tools import parse_answer


def test_bundled_presets(tmp_path):
    store = Store(tmp_path / 'presets.db')
    state = store.state()
    assert state['active_profile'] == 'Baseline'
    full, half = state['profiles']['Full dose'], state['profiles']['Half dose']
    assert [(c['layer'], c['neuron'], c['dose']) for c in full['steering']] == [(14, 8949, 1), (13, 7956, -1)]
    assert [c['dose'] for c in half['steering']] == [.5, -.5]
    assert state['profiles']['L14 N8949 +0.4']['steering'] == [
        {'layer': 14, 'neuron': 8949, 'dose': .4, 'enabled': True}]
    state['active_profile'] = 'Half dose'
    store.save_state(state)
    assert Store(tmp_path / 'presets.db').snapshot()['codes'] == half['steering']


def test_parser():
    codes = parse_codes('L15 00150 +1\nL15 00420 -1\nL22 01234 +0.5\nL0 0 0',32,9216)
    assert [c.dose for c in codes] == [1,-1,.5,0]
    for invalid in ['L32 0 +1','L0 9216 1','L0 1 nan','L0 1 +1\nL0 1 -1']:
        with pytest.raises(ValueError): parse_codes(invalid,32,9216)


def test_profile_restart_and_snapshot(tmp_path):
    path=tmp_path/'state.db'
    store=Store(path); state=store.state()
    state['profiles']['A']={'enabled':True,'steering':[{'layer':0,'neuron':1,'dose':1}]}
    state['active_profile']='A';store.save_state(state)
    snapshot=store.snapshot()
    state=store.state();state['profiles']['A']['steering'][0]['dose']=-1;store.save_state(state)
    assert snapshot['codes'][0]['dose']==1
    assert Store(path).snapshot()['codes'][0]['dose']==-1
    state=store.state();state['enabled']=False;store.save_state(state)
    assert store.snapshot()['codes']==[]
    state=store.state();state['enabled']=True;store.save_state(state)
    assert store.snapshot('Baseline')['codes']==[]
    store.save_chat({'messages':[{'role':'user','content':'hi'}]})
    assert Store(path).chats()[0]['messages'][0]['content']=='hi'
    with pytest.raises(ValueError):store.save_state({'version':0})


def test_hooks_all_tokens_no_duplicates_and_baseline():
    layer=SimpleNamespace(mlp=SimpleNamespace(down_proj=torch.nn.Linear(4,3,bias=False)))
    with torch.no_grad():layer.mlp.down_proj.weight.fill_(1)
    engine=SteeringEngine([layer],[6])
    prompt=torch.zeros(1,5,4);decode=torch.zeros(1,1,4)
    baseline=layer.mlp.down_proj(prompt)
    weights=layer.mlp.down_proj.weight.detach().clone()
    codes=[Code(0,1,1),Code(0,2,-.5),Code(0,3,100,False),Code(0,0,0)]
    engine.install(codes);engine.install(codes)
    assert len(layer.mlp.down_proj._forward_pre_hooks)==1
    expected=3/(3**.5)
    assert torch.allclose(layer.mlp.down_proj(prompt), torch.full_like(baseline,expected))
    assert torch.allclose(layer.mlp.down_proj(decode),torch.full((1,1,3),expected))
    assert {o['tokens'] for o in engine.observations}=={1,5}
    engine.remove();assert torch.equal(layer.mlp.down_proj(prompt),baseline)
    assert torch.equal(layer.mlp.down_proj.weight,weights)


def test_native_tool_parser():
    answer=parse_answer('<tool_call><function=read_file><parameter=path>plan.md</parameter></function></tool_call>')
    call=answer['tool_calls'][0]
    assert json.loads(call['function']['arguments'])=={'path':'plan.md'}
    assert call['id'].startswith('call_')
    assert parse_answer('<tool_call>{"name":"add","arguments":{"a":2}}</tool_call>')['tool_calls']


class FakeRuntime:
    """Protocol test only; never used by the production app."""
    ready=True; layers=[None]*32; width=9216; info={}
    def stream(self,request,snapshot):
        if request.get('tools') and request['messages'][-1]['role']!='tool':
            yield '<tool_call>{"name":"read_file","arguments":{"path":"plan.md"}}</tool_call>'
        else:yield 'Received result' if request['messages'][-1]['role']=='tool' else json.dumps(snapshot)


def test_native_llama_ui_and_request_contract(tmp_path):
    from app.server import create_app
    class NativeRuntime(FakeRuntime):
        info = {'backend': 'llama.cpp CUDA'}
        def stream(self, request, snapshot):
            self.last_request = request
            yield 'Hello'
    ui = tmp_path / 'ui'
    ui.mkdir()
    (ui / 'index.html').write_text("<html><head></head><body><script>base: new URL('.', location).pathname.slice(0, -1)</script></body></html>")
    (ui / 'bundle.js').write_text('console.log(1)')
    runtime = NativeRuntime()
    with TestClient(create_app(runtime, tmp_path / 'store', ui), base_url='http://127.0.0.1:8000') as client:
        page = client.get('/')
        assert page.status_code == 200 and '/static/steering-control.js' in page.text
        assert '/static/steering-control.js' in client.get('/index.html').text
        assert '<base href="/">' in client.get('/chat/example').text
        assert client.get('/bundle.js').text == 'console.log(1)'
        assert client.get('/steering').status_code == 200
        assert client.get('/props').json()['role'] == 'model'
        assert client.get('/tools').json() == []
        assert client.post('/v1/streams/lookup', json={}).json() == []
        assert client.get('/api/profiles', headers={'Origin': 'https://example.com'}).status_code == 403
        assert client.get('/api/profiles', headers={'Origin': 'http://127.0.0.1:8000'}).status_code == 200
        state = client.get('/api/profiles').json()
        state['active_profile'] = 'L14 N8949 +0.4'
        assert client.put('/api/profiles', json=state).status_code == 200
        body = {'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True,
            'return_progress': True, 'sse_ping_interval': 1, 'reasoning_format': 'auto',
            'reasoning_control': True, 'backend_sampling': False, 'timings_per_token': True,
            'top_k': 25, 'min_p': .05, 'typ_p': .9, 'repeat_penalty': 1.1, 'repeat_last_n': 64}
        response = client.post('/v1/chat/completions', json=body)
        assert response.status_code == 200 and 'data: [DONE]' in response.text
        assert 'L14 N8949 +0.4' in response.text and runtime.last_request['top_k'] == 25
        assert client.post('/v1/chat/completions', json={**body, 'backend_sampling': True}).status_code == 400


def test_malformed_tool_markup_is_reported(tmp_path):
    from app.server import create_app
    class MalformedRuntime(FakeRuntime):
        def stream(self, request, snapshot):
            yield 'Trying to read. </parameter></function></tool_call>'
    with TestClient(create_app(MalformedRuntime(), tmp_path)) as client:
        response = client.post('/v1/chat/completions', json={
            'messages': [{'role': 'user', 'content': 'Read a file'}],
            'tools': [{'type': 'function', 'function': {'name': 'read_file'}}],
        })
        assert response.status_code == 400
        assert 'malformed tool-call' in response.json()['error']['message']
        streamed = client.post('/v1/chat/completions', json={
            'messages': [{'role': 'user', 'content': 'Read a file'}], 'stream': True,
            'tools': [{'type': 'function', 'function': {'name': 'read_file'}}],
        })
        assert 'Inference error: Model generated malformed tool-call' in streamed.text
        assert '"error":' in streamed.text and 'data: [DONE]' in streamed.text


def test_api_stream_tools_and_concurrency(tmp_path):
    from app.server import create_app
    with TestClient(create_app(FakeRuntime(),tmp_path)) as client:
        assert client.get('/v1/models').json()['data'][0]['id']=='Qwen/Qwen3.5-4B'
        state=client.get('/api/profiles').json()
        state['profiles']['A']={'enabled':True,'steering':[{'layer':15,'neuron':150,'dose':1}]}
        state['active_profile']='A';assert client.put('/api/profiles',json=state).status_code==200
        body={'messages':[{'role':'user','content':'Read plan.md'}],'tools':[{'type':'function','function':{'name':'read_file','parameters':{'type':'object'}}}]}
        answer=client.post('/v1/chat/completions',json=body).json()['choices'][0]['message']
        assert answer['tool_calls'][0]['function']['name']=='read_file'
        body['messages'] += [answer,{'role':'tool','tool_call_id':answer['tool_calls'][0]['id'],'content':'Project plan'}]
        assert client.post('/v1/chat/completions',json=body).json()['choices'][0]['message']['content']=='Received result'
        body['stream']=True
        stream=client.post('/v1/chat/completions',json=body)
        assert 'data: [DONE]' in stream.text and 'chat.completion.chunk' in stream.text
        def call(baseline):
            result=client.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'Hi'}],'baseline':baseline}).json()
            return result['steering']['codes']
        with ThreadPoolExecutor(2) as pool:
            a=pool.submit(call,False);b=pool.submit(call,True)
            assert a.result()[0]['dose']==1 and b.result()==[]
        assert client.get('/').status_code==200
        invalid=client.get('/api/profiles').json()
        invalid['profiles']['A']['steering'][0]['layer']=15.5
        assert client.put('/api/profiles',json=invalid).status_code==422
        assert client.post('/v1/chat/completions',json={'messages':[{'role':'tool','content':'bad','tool_call_id':'unknown'}]}).status_code==400

def test_model_selection_keeps_uncensored_unsteered(tmp_path):
    from app.model_config import MODEL, UNSTEERED_MODEL
    from app.server import create_app
    class TwoModels(FakeRuntime):
        def available(self, model): return model in (MODEL, UNSTEERED_MODEL)
        def stream(self, request, snapshot):
            self.request, self.snapshot = request, snapshot
            yield 'ok'
    runtime = TwoModels()
    with TestClient(create_app(runtime, tmp_path), base_url='http://127.0.0.1:8000') as client:
        state = client.get('/api/profiles').json()
        state.update(enabled=True, active_profile='L14 N8949 +0.4')
        assert client.put('/api/profiles', json=state).status_code == 200
        body = {'model': UNSTEERED_MODEL, 'messages': [{'role': 'user', 'content': 'Hi'}]}
        result = client.post('/v1/chat/completions', json=body).json()
        assert result['model'] == UNSTEERED_MODEL and result['steering']['codes'] == []
        assert runtime.snapshot['codes'] == []
        assert client.put('/api/model', json={'model': UNSTEERED_MODEL}).status_code == 200
        assert client.get('/props').json()['model_alias'] == UNSTEERED_MODEL
        result = client.post('/v1/chat/completions', json={'messages': body['messages']}).json()
        assert result['model'] == UNSTEERED_MODEL and result['steering']['codes'] == []
        result = client.post('/v1/chat/completions', json={**body, 'model': MODEL}).json()
        assert result['steering']['codes'][0]['dose'] == .4
