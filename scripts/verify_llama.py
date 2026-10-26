"""Real GPU/GGUF acceptance; writes data/llama-acceptance.json."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gguf
import numpy as np
from fastapi.testclient import TestClient
from app.llama_runtime import LlamaRuntime


def main():
    os.environ['MODEL_BACKEND'] = 'llama_cpp'
    from app.server import create_app
    runtime = LlamaRuntime()
    runtime.load()
    report = {"model": runtime.info, "checks": {}}
    baseline = {"name": "Baseline", "version": 0, "codes": []}
    codes = [{"layer": 14, "neuron": 8949, "dose": 1.0, "enabled": True},
             {"layer": 13, "neuron": 7956, "dose": -1.0, "enabled": True}]
    steered = {"name": "Full dose", "version": 1, "codes": codes}
    request = {"messages": [{"role": "user", "content": "Reply with the word hello."}],
               "temperature": 0, "max_tokens": 12}
    def run(snapshot):
        return ''.join(runtime.stream(dict(request), snapshot))
    base = run(baseline)
    report["baseline_text"] = base
    report["checks"]["baseline_inference"] = bool(base.strip())
    report["steered_text"] = run(steered)
    report["steered_inspection"] = runtime.last_inspection
    obs = runtime.last_inspection['observations']
    for code in codes:
        layer, neuron, dose = code['layer'], code['neuron'], code['dose']
        records = [o for o in obs if o['layer'] == layer and o['neuron'] == neuron]
        report['checks'][f'L{layer}_{neuron}_prefill_decode'] = (
            any(o['tokens'] > 1 and o['measured_shift'] * dose > 0 for o in records)
            and any(o['tokens'] == 1 and o['measured_shift'] * dose > 0 for o in records))
    tensor = runtime.tensors['blk.14.ffn_down.weight']
    whole_matrix = gguf.dequantize(tensor.data, tensor.tensor_type)
    reference = float(np.linalg.norm(whole_matrix[:, 8949].astype(np.float64)))
    measured = runtime.column_norm(14, 8949)
    report['column_norms'] = {'full_dequantization': reference, 'selected_block': measured}
    report['checks']['quantized_column_norm'] = bool(np.isclose(reference, measured, rtol=1e-6))
    del whole_matrix
    report['checks']['baseline_restored'] = run(baseline) == base
    with ThreadPoolExecutor(2) as pool:
        a, b = pool.submit(run, baseline), pool.submit(run, steered)
        report['checks']['concurrent_baseline_isolated'] = a.result() == base
        b.result()
    with TestClient(create_app(runtime, 'data/llama-api-check')) as client:
        state = client.get('/api/profiles').json()
        state['active_profile'] = 'Full dose'
        client.put('/api/profiles', json=state).raise_for_status()
        response = client.post('/v1/chat/completions', json={**request, 'stream': True})
        response.raise_for_status()
        report['checks']['api_stream_and_profile'] = (
            'data: [DONE]' in response.text and 'Full dose' in response.text and '"error"' not in response.text)
    Path('data/llama-acceptance.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['checks'], indent=2))
    assert all(report['checks'].values())


if __name__ == '__main__':
    main()
