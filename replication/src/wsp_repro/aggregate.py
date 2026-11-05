"""Across-independent-generation sample variance, separate from episode variance."""
import json
from collections import defaultdict
from pathlib import Path
import statistics
from .spec import make_spec,MODELS

ROOT=Path(__file__).resolve().parents[2]


def aggregate(root=ROOT):
    groups=defaultdict(list)
    for j in make_spec()['jobs']:
        if j["experiment"]!=5:continue
        folder=root/'runs'/j['id'];artifact=folder/'artifact.json';score=folder/'evaluation.json'
        a=json.loads(artifact.read_text()) if artifact.exists() else {}
        key=(j['model'],a.get('transport',MODELS[j['model']]['transport']),j['game'],j['kind'],j.get('width'))
        attempt={'repeat':j['repeat'],'job':j['id'],'status':'pending'}
        if artifact.exists() and score.exists():
            e=json.loads(score.read_text());attempt.update(status='scored',mean=e['mean'],episode_variance=e['episode_sample_variance'])
        elif (folder/'state.json').exists():
            s=json.loads((folder/'state.json').read_text());kind=s.get('error_type','');message=s.get('message','')
            terminal=(folder/'terminal-failure.json').exists() or kind=='IncompleteModelOutput' or message.startswith(
                ('DeepSeek response ended:','OpenAI response incomplete:'))
            invalid=kind in {'ValueError','SyntaxError'} and (folder/'call/answer.md').exists()
            if terminal or invalid:attempt.update(status='model_failure',error_type=kind)
            elif s.get('status')=='failed':attempt.update(status='infrastructure_error',error_type=kind)
        if (folder/'evaluation-error.json').exists() and attempt['status']!='scored':attempt['status']='infrastructure_error'
        groups[key].append(attempt)
    rows=[]
    for key,attempts in sorted(groups.items(),key=lambda x:str(x[0])):
        if len({x["repeat"] for x in attempts})!=len(attempts):raise ValueError("Duplicate repeats")
        values=[x for x in attempts if x['status']=='scored']
        means=[x["mean"] for x in values]
        rows.append(dict(zip(("model","transport","game","kind","width"),key),
            n=len(means),expected_repeats=3,complete=len(means)==3,mean=statistics.mean(means) if means else None,
            attempts_finished=len(attempts)==3 and all(x['status'] in ('scored','model_failure') for x in attempts),
            model_failures=sum(x['status']=='model_failure' for x in attempts),
            infrastructure_errors=sum(x['status']=='infrastructure_error' for x in attempts),
            pending=sum(x['status']=='pending' for x in attempts),
            variance_scope='three independent generations' if len(means)==3 else 'available scored outputs only; incomplete valid repetition group',
            run_sample_variance=statistics.variance(means) if len(means)>1 else None,
            run_sample_std=statistics.stdev(means) if len(means)>1 else None,runs=values,attempts=attempts))
    return rows


def main():
    rows=aggregate()
    folder=ROOT/"artifacts";folder.mkdir(exist_ok=True)
    (folder/"game-summary.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))


if __name__=="__main__":main()
