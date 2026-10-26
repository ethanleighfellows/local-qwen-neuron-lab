"""Live API smoke test including a real model -> read-file -> model tool cycle."""
import json
from pathlib import Path
import httpx


def main():
    client=httpx.Client(base_url='http://127.0.0.1:8000',timeout=180)
    report={}
    state=client.get('/api/profiles').json()
    report['ui_http_200']=client.get('/').status_code==200
    report['models']=client.get('/v1/models').json()
    request={'model':'Qwen/Qwen3.5-4B','temperature':0,'max_tokens':160,
        'messages':[{'role':'user','content':'Use read_file to read README.md. After receiving its contents, tell me the project name.'}],
        'tools':[{'type':'function','function':{'name':'read_file','description':'Read a local project file.',
            'parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path']}}}]}
    first=client.post('/v1/chat/completions',json=request)
    first.raise_for_status(); first=first.json()
    report['first_response']=first
    answer=first['choices'][0]['message']
    if answer.get('tool_calls'):
        call=answer['tool_calls'][0]
        arguments=json.loads(call['function']['arguments'])
        # Only this fixed disposable read is authorized by this acceptance script.
        if call['function']['name']!='read_file' or arguments.get('path')!='README.md':
            raise RuntimeError('Unexpected tool call: '+str(call))
        request['messages'] += [answer,{'role':'tool','tool_call_id':call['id'],
            'content':Path('README.md').read_text(encoding='utf-8')[:1000]}]
        second=client.post('/v1/chat/completions',json=request)
        second.raise_for_status();second=second.json();report['second_response']=second
        report['real_model_tool_cycle']=bool(second['choices'][0]['message'].get('content'))
    else:
        report['real_model_tool_cycle']=False
    report['same_global_profile']=first['steering']['name']==state['active_profile']
    request={'messages':[{'role':'user','content':'Say hello.'}],'stream':True,'temperature':0,'max_tokens':8}
    stream=client.post('/v1/chat/completions',json=request)
    stream.raise_for_status()
    report['stream_done']='data: [DONE]' in stream.text
    Path('data/api-acceptance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if isinstance(v,bool)},indent=2))
    assert report['real_model_tool_cycle'] and report['stream_done']


if __name__=='__main__': main()
