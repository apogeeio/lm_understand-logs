"""Bounded, restartable independent repeats and a separate high-budget game arm."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
CONFIG=ROOT/'replication/configs/supplementary.json'
CONTROL=ROOT/'replication/artifacts/control/supplementary'
RETRYABLE={'RateLimited','TransientProviderError','QuotaExhausted','URLError','TimeoutError','ConnectionResetError','RemoteDisconnected'}


def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def car_eval(job,out):
    artifact=json.loads((out/'artifact.json').read_text());actor=out/artifact['artifact']
    assert hashlib.sha256(actor.read_bytes()).hexdigest()==artifact['sha256']
    # A single CPU simulator per study avoids competing JAX compilations.
    with (CONTROL/'car-eval.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for name,n,seed in [('search',256,1400),('test',1024,1401)]:
            result=out/(name+'-evaluation.json')
            if result.exists():continue
            env=dict(os.environ,JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            with (out/(name+'-evaluation.log')).open('a') as log:
                subprocess.run([str(ROOT/'.venv-sim/bin/python'),'-m','wsp_repro.car','--actor',str(actor),
                    '--episodes',str(n),'--seed',str(seed),'--output',str(result)],cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
        save(out/'evaluation.json',{'job':job,'search':json.loads((out/'search-evaluation.json').read_text()),
            'test':json.loads((out/'test-evaluation.json').read_text()),'frozen_sha256':artifact['sha256'],
            'search_seed':1400,'test_seed':1401,'seed_stream':'fixed replication streams'})


def execute(job):
    out=ROOT/'replication/runs'/job['id'];out.mkdir(parents=True,exist_ok=True)
    status=out/'supplementary-status.json'
    if status.exists() and json.loads(status.read_text())['status'] in ('completed','model_failure','execution_failure'):return
    with (out/'supplementary.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        for attempt in range(100):
            save(status,{'status':'running','pid':os.getpid(),'updated':time.time(),'infrastructure_attempt':attempt})
            try:
                if job['experiment'] in (1,2,3):
                    from .run_reading import generate
                    generate(job)
                elif job['experiment']==4:
                    from .car import generate
                    generate(job);car_eval(job,out)
                elif job['experiment']==5:
                    from .run_games import generate
                    from .dispatch_eval import evaluate_frozen
                    _,result=generate(job)
                    if result in ('failed','retained_failure'):
                        state=json.loads((out/'state.json').read_text())
                        error=RuntimeError(state.get('message','Generation failed'))
                        error.provider_type=state.get('error_type','RuntimeError');raise error
                    if result not in ('frozen','already_frozen'):raise RuntimeError(result)
                    evaluate_frozen(job)
                else:raise ValueError('Only experiments 1..5 are enabled')
                save(status,{'status':'completed','updated':time.time(),'semantic_review':'pending'});return
            except Exception as error:
                kind=getattr(error,'provider_type',type(error).__name__)
                retry=kind in RETRYABLE
                terminal=kind=='IncompleteModelOutput' or str(error).startswith(('DeepSeek response ended:','OpenAI response incomplete:'))
                record={'status':'infrastructure_retry' if retry else ('model_failure' if terminal else 'execution_failure'),
                    'error_type':kind,'message':str(error),'updated':time.time(),'infrastructure_attempt':attempt}
                save(status,record)
                with (out/'supplementary-events.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
                if not retry:return
                time.sleep(min(1800,60*2**min(attempt,5))+random.uniform(0,30))


def launch_one(job,concurrency):
    env=dict(os.environ,PYTHONPATH=str(ROOT/'replication/src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for model,slots in concurrency.items():env['WSP_'+model.upper()+'_CONCURRENCY']=str(slots)
    with (CONTROL/(job['id']+'.log')).open('a') as log:
        result=subprocess.run([sys.executable,'-m','wsp_repro.supplementary','--job',job['id']],cwd=ROOT,env=env,stdout=log,stderr=log)
    print(json.dumps({'job':job['id'],'exit_code':result.returncode,'updated':time.time()}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job');args=ap.parse_args()
    config=json.loads(CONFIG.read_text());CONTROL.mkdir(parents=True,exist_ok=True)
    if args.job:
        execute(next(j for j in config['jobs'] if j['id']==args.job));return
    with (CONTROL/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(CONTROL/'controller.json',{'pid':os.getpid(),'started':time.time(),'concurrency':config['provider_concurrency'],
            'manifest_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest()})
        pools={m:ThreadPoolExecutor(max_workers=n) for m,n in config['provider_concurrency'].items()}
        futures=[pools[j['model']].submit(launch_one,j,config['provider_concurrency']) for j in config['jobs']]
        for future in futures:future.result()
        for pool in pools.values():pool.shutdown()
        save(CONTROL/'finished.json',{'updated':time.time(),'status':'attempts_finished_review_required'})

if __name__=='__main__':main()
