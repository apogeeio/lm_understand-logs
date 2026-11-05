"""Read-only case inventory; no prompts, histories, credentials or model requests in output."""
from collections import Counter
from .execution_scope import math_paused
import json
from pathlib import Path
import time
from .aggregate import aggregate
from .audit_records import review_status
from .spec import make_spec

ROOT=Path(__file__).resolve().parents[3]


def snapshot(root=ROOT):
    runs=root/'replication/runs'
    scope=root/'replication/configs/search-deferrals.json'
    deferred_budgets=json.loads(scope.read_text())['math_pass_budgets'] if scope.exists() else []
    game_states={a['job']:a['status'] for row in aggregate(root/'replication') for a in row['attempts']}
    cases=[]
    for job in make_spec()['jobs']:
        folder=runs/job['id'];exp=job['experiment']
        state='started' if folder.exists() else 'not_started'
        if exp==8 and job['passes'] in deferred_budgets:state='deferred'
        audit=None;benchmarks={}
        if exp==5:
            status=game_states[job['id']]
            if status!='pending':state=status
            elif (folder/'artifact.json').exists():state='frozen_pending_evaluation'
        elif exp in (1,2,3) and (folder/'result.json').exists():
            state='scored';audit=review_status(folder)
        elif exp in (4,6) and (folder/'evaluation.json').exists():state='scored'
        elif (folder/'frozen.json').exists():state='frozen_pending_evaluation'
        if exp in (6,8) and (folder/'frozen.json').exists():
            audit=review_status(folder)
        summary=folder/'heldout/summary.json'
        if exp==8 and summary.exists():
            report=json.loads(summary.read_text())
            state='scored' if report['all_requested_benchmarks_complete'] else 'evaluating'
            if report.get('current_phase_benchmarks_complete') and report.get('deferred_benchmarks'):
                state='scored_with_deferred_benchmarks'
            benchmarks={name:{k:s[k] for k in ('completed_problems','expected_problems','complete')}
                for name,s in report['sets'].items()}
        if (folder/'terminal-failure.json').exists():state='model_failure'
        if (folder/'evaluation-held-format.json').exists():state='held_invalid_format'
        if audit not in (None,'passed','pending'):state='protocol_violation'
        row={'id':job['id'],'experiment':exp,'model':job['model'],'state':state,'audit':audit}
        if exp==8 and math_paused(root):
            row['state_before_pause']=row['state'];row['state']='paused'
        search=folder/'search-state.json'
        if search.exists():
            row['search']={k:v for k,v in json.loads(search.read_text()).items() if k!='history'}
        if benchmarks:row['benchmarks']=benchmarks
        if exp==8 and summary.exists() and report.get('deferred_benchmarks'):
            row['deferred_benchmarks']=report['deferred_benchmarks']
        for name in ('search-error.json','evaluation-error.json'):
            path=folder/name
            if path.exists():
                error=json.loads(path.read_text())
                row.setdefault('recorded_errors',{})[name]={k:error[k] for k in ('type','error_type','updated') if k in error}
        cases.append(row)
    return {'updated':time.time(),'total':len(cases),'states':dict(Counter(c['state'] for c in cases)),
        'cases':cases,'not_completion_claim':True}


def main():
    report=snapshot();path=ROOT/'replication/artifacts/progress-snapshot.json';path.parent.mkdir(exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
    print(json.dumps({'total':report['total'],'states':report['states'],
        'active':[c for c in report['cases'] if c['state'] in ('started','evaluating','held_invalid_format','infrastructure_error')],
        'scored_pending_semantic_review':[c['id'] for c in report['cases']
            if c['state'] in ('scored','scored_with_deferred_benchmarks') and c['audit']=='pending']},indent=2))


if __name__=='__main__':main()
