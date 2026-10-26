"""Hardware acceptance tests. Loads the real checkpoint; writes an evidence report."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from app.runtime import Runtime


def main():
    runtime=Runtime();runtime.load()
    report={'model':runtime.info,'checks':{}}
    request={'messages':[{'role':'user','content':'Reply with the word hello.'}],'temperature':0,'max_tokens':12}
    baseline={'name':'Baseline','version':0,'codes':[]}
    def run(snapshot):return ''.join(runtime.stream(request,snapshot))
    text=run(baseline);report['baseline_text']=text
    report['checks']['unsteered_inference']=bool(text.strip())
    for name,codes in [('positive',[{'layer':15,'neuron':150,'dose':1}]),('multiple',[{'layer':15,'neuron':150,'dose':-1},{'layer':15,'neuron':420,'dose':.5}])]:
        report[name+'_text']=run({'name':name,'version':1,'codes':codes})
        info=runtime.last_inspection
        report[name+'_inspection']=info
        observations=info['observations']
        report['checks'][name+'_prefill_and_decode']=all(any(o['neuron']==c['neuron'] and o['tokens']>1 and o['measured_shift']*c['dose']>0 for o in observations) and any(o['neuron']==c['neuron'] and o['tokens']==1 and o['measured_shift']*c['dose']>0 for o in observations) for c in codes)
    report['checks']['baseline_restored']=run(baseline)==text
    with ThreadPoolExecutor(2) as pool:
        a=pool.submit(run,baseline)
        b=pool.submit(run,{'name':'steered','version':2,'codes':[{'layer':15,'neuron':150,'dose':1}]})
        report['checks']['concurrent_baseline_isolated']=a.result()==text
        b.result()
    Path('data/acceptance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['checks'],indent=2))
    assert all(report['checks'].values())


if __name__=='__main__':main()
