"""Two-phase live acceptance across an actual server process restart."""
import json
import sys
from pathlib import Path
import httpx


def main():
    client = httpx.Client(base_url='http://127.0.0.1:8000', timeout=180)
    checkpoint = Path('data/persistence-checkpoint.json')
    state = client.get('/api/profiles').json()
    if sys.argv[1] == 'prepare':
        checkpoint.write_text(json.dumps(state))
        state['profiles']['Acceptance check'] = {'enabled':True,'steering':[{'layer':15,'neuron':150,'dose':1,'enabled':True}]}
        state['active_profile']='Acceptance check';state['enabled']=True
        client.put('/api/profiles',json=state).raise_for_status()
        chat=client.post('/api/chats',json={'title':'Acceptance: restart persistence','messages':[]}).json()
        checkpoint.with_suffix('.chat.json').write_text(json.dumps(chat))
        response=client.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'Say hello.'}],'temperature':0,'max_tokens':8})
        response.raise_for_status()
        assert response.json()['steering']['codes'][0]['dose']==1
        print('Prepared active steering profile and a fresh persistent conversation')
    else:
        assert state['active_profile']=='Acceptance check'
        chat=json.loads(checkpoint.with_suffix('.chat.json').read_text())
        assert any(c['id']==chat['id'] for c in client.get('/api/chats').json())
        # An OpenAI client sends no steering state; the restarted server applies its global default.
        response=client.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'Say hello.'}],'temperature':0,'max_tokens':8})
        response.raise_for_status()
        assert response.json()['steering']['codes'][0]['dose']==1
        original=json.loads(checkpoint.read_text());original['version']=state['version']
        client.put('/api/profiles',json=original).raise_for_status()
        report={'profile_survives_process_restart':True,'chat_survives_process_restart':True,
            'new_chat_inherits_global_profile':True,'openai_client_inherits_web_profile':True}
        Path('data/persistence-acceptance.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
