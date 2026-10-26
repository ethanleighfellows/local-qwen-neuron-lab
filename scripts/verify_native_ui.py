"""Check the running native UI and a real streamed request without editing profiles."""
import json
import re
from pathlib import Path
import httpx


def main():
    with httpx.Client(base_url='http://127.0.0.1:8000', timeout=120, trust_env=False) as client:
        page = client.get('/')
        page.raise_for_status()
        assert 'steering-control.js' in page.text and '__sveltekit_' in page.text
        script = re.search(r'\./(_app/immutable/bundle\.[A-Za-z0-9_-]+\.js)', page.text)
        assert script and client.get('/' + script[1]).status_code == 200
        assert client.get('/chat/example').status_code == 200
        assert client.get('/steering').status_code == 200
        assert client.get('/props').json()['role'] == 'model'
        assert client.get('/tools').json() == []
        state = client.get('/api/profiles').json()
        expected = state['active_profile']
        payload = {'messages': [{'role': 'user', 'content': 'Say hello.'}], 'stream': True,
            'return_progress': True, 'sse_ping_interval': 1, 'reasoning_format': 'auto',
            'reasoning_control': True, 'backend_sampling': False, 'timings_per_token': True,
            'temperature': 0, 'top_p': .9, 'max_tokens': 16}
        response = client.post('/v1/chat/completions', json=payload,
            headers={'Origin': 'http://127.0.0.1:8000', 'X-Conversation-Id': 'verification'})
        response.raise_for_status()
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: {')]
        assert 'data: [DONE]' in response.text and not any('error' in event for event in events)
        assert any(event.get('steering', {}).get('name') == expected for event in events)
        answer = ''.join(event['choices'][0].get('delta', {}).get('content', '') or ''
                         for event in events if event.get('choices'))
        assert answer
        status = client.get('/api/status').json()
        report = {'frontend_assets_served': True, 'native_request_streamed': True,
            'profile_applied': expected, 'response': answer,
            'last_inference': status.get('last_inference')}
        Path('data/native-ui-acceptance.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
