"""Account for logged editor tokens and completed GPU tasks, without inventing billing totals."""
from collections import defaultdict
import json
from pathlib import Path
import time
from .spec import make_spec

ROOT=Path(__file__).resolve().parents[3]


def report(root=ROOT):
    formal={j['id'] for j in make_spec()['jobs']};runs=root/'replication/runs'
    groups=defaultdict(lambda:defaultdict(int));missing=[];seen=set()
    paths=[*runs.glob('**/call.json'),*runs.glob('**/call-failure.json')]
    for path in paths:
        if path.parent in seen:continue
        seen.add(path.parent);meta=json.loads(path.read_text())
        case=path.relative_to(runs).parts[0]
        scope='study' if case in formal else 'infrastructure_or_reference'
        key=(scope,meta.get('model_group'),meta.get('transport'),meta.get('resolved_model'))
        row=groups[key];row['logged_responses']+=1
        row['terminal_outputs']+=bool(meta.get('terminal_model_output'))
        row['reported_call_wait_seconds']+=meta.get('seconds',0)
        usage=meta.get('usage',{})
        if not isinstance(usage,dict):missing.append(str(path.relative_to(root)));continue
        inp=usage.get('input_tokens',usage.get('prompt_tokens'))
        out=usage.get('output_tokens',usage.get('completion_tokens'))
        if inp is None or out is None:missing.append(str(path.relative_to(root)));continue
        cached=usage.get('prompt_cache_hit_tokens',usage.get('input_tokens_details',
            usage.get('prompt_tokens_details',{})).get('cached_tokens',0))
        reasoning=usage.get('output_tokens_details',usage.get('completion_tokens_details',{})).get('reasoning_tokens',0)
        assert 0<=cached<=inp and 0<=reasoning<=out
        assert usage.get('total_tokens',inp+out)==inp+out
        row['responses_with_token_usage']+=1
        row['input_tokens']+=inp;row['output_tokens']+=out
        row['cached_input_tokens_subset']+=cached;row['reasoning_output_tokens_subset']+=reasoning
    api=[dict(zip(('scope','model','transport','returned_model'),key),**values)
        for key,values in sorted(groups.items(),key=lambda x:str(x[0]))]
    gpu=defaultdict(lambda:defaultdict(int));queue=root/'replication/artifacts/math-forward-queue'
    for path in queue.glob('*.response.json'):
        uid=path.name.removesuffix('.response.json')
        request=queue/(uid+'.request.processing')
        if not request.exists():request=queue/(uid+'.request.json')
        if not request.exists():continue
        response=json.loads(path.read_text())
        if not response['ok']:continue
        envelope=json.loads(request.read_text());kind=envelope['request']['type']
        scope=('heldout_reference' if uid.startswith(('eval-eval-base-','eval-eval-released-p50-')) else
            'heldout_new_search' if uid.startswith('eval-') else
            'practice_search' if uid.startswith('e8-') else 'infrastructure_smoke')
        row=gpu[(scope,kind)];row['completed_tasks']+=1;row['worker_task_wall_seconds']+=response.get('seconds',0)
        result=response['result']
        if kind=='generate':
            for arm in ('original','edited'):
                if arm in result:
                    value=result[arm];row['generations']+=value['generations']
                    row['generated_tokens']+=round(value['generations']*value['mean_tokens'])
        elif kind=='heldout':row['generations']+=result['generations'];row['generated_tokens']+=result['tokens']
        elif kind=='teacher_forced':row['teacher_forced_tokens']+=result['tokens']
    gpu_rows=[dict(scope=key[0],kind=key[1],**value) for key,value in sorted(gpu.items())]
    legacy=[str(p.parent.relative_to(runs)) for p in runs.glob('*/terminal-failure.json')
        if json.loads(p.read_text()).get('original_usage_available') is False]
    return {'updated':time.time(),'editor_api':api,'gpu_completed_tasks':gpu_rows,
        'missing_usage_metadata':missing,'legacy_terminal_outputs_missing_usage':legacy,
        'monetary_cost':None,
        'limitations':['Token totals cover logged responses only; cancelled/unrecorded calls may have been charged.',
            'This report covers subject-model calls; orchestration/chat and unrelated shared-service usage are outside its scope.',
            'Cached input and reasoning output are subsets, not extra tokens to add again.',
            'Per-call wait seconds include provider-lock waits and overlap across controllers; they are not study elapsed time.',
            'GPU task seconds exclude idle/startup/failed or preempted unfinished work and include task preprocessing/activation export.',
            'Coding-plan tokens do not establish an OpenAI API dollar bill; provider billing statements are unavailable.']}


def main():
    data=report();path=ROOT/'replication/artifacts/usage-summary.json'
    path.parent.mkdir(exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
    print(json.dumps(data,indent=2))


if __name__=='__main__':main()
