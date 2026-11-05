"""Reload a game controller after a frozen policy's evaluator finishes, without losing API replies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from .launch_math import ROOT, launch


def command(pid):
    try:
        raw=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().rstrip('\0')
        return raw.split('\0') if raw else []
    except (FileNotFoundError, ProcessLookupError):
        return []


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--old-pid',type=int,required=True)
    ap.add_argument('--model',choices=['astra','deepseek'],required=True)
    args=ap.parse_args()
    proc=Path('/proc')/str(args.old_pid)
    receipt=ROOT/f'replication/artifacts/control/games-upgrade-{args.old_pid}.json'
    if receipt.exists():return
    expected=['wsp_repro.overnight_games','--model',args.model]
    stopped=False
    def terminate(signum,frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,terminate)
    try:
        while True:
            cmd=command(args.old_pid)
            if not cmd:raise RuntimeError('Original controller exited; inspect before replacement')
            if '-m' not in cmd or cmd[cmd.index('-m')+1:]!=expected or proc.joinpath('cwd').resolve()!=ROOT:
                raise RuntimeError('PID does not identify this study game controller')
            # Inspect only after stopping the parent: it cannot race from child-wait
            # into a new model request between our check and the stop signal.
            os.kill(args.old_pid,signal.SIGSTOP);stopped=True
            state=json.loads((ROOT/f'replication/artifacts/queue-{args.model}.json').read_text())
            job=state.get('job','')
            children=proc.joinpath('task',str(args.old_pid),'children').read_text().split()
            evaluators=[]
            for child in children:
                cc=command(child)
                if any(m in cc for m in ('wsp_repro.evaluate_game','wsp_repro.dispatch_eval')) and any(
                    x==job or f'/{job}/' in x for x in cc) and job:
                    evaluators.append(child)
            if not evaluators:
                os.kill(args.old_pid,signal.SIGCONT);stopped=False
                time.sleep(5);continue
            folder=ROOT/'replication/runs'/job
            frozen=json.loads((folder/'artifact.json').read_text())
            policy=folder/frozen['artifact']
            assert hashlib.sha256(policy.read_bytes()).hexdigest()==frozen['sha256']
            checkpoint=receipt.with_suffix('.checkpoint.json')
            checkpoint.write_text(json.dumps({'status':'waiting_evaluator','old_pid':args.old_pid,
                'model':args.model,'job':job,'children':{child:command(child) for child in evaluators},
                'frozen_policy_sha256':frozen['sha256'],'updated':time.time()},indent=2))
            began=time.monotonic()
            while any(command(child) for child in evaluators):
                if time.monotonic()-began>3800:raise RuntimeError('Evaluator did not finish within its normal limit')
                time.sleep(1)
            result=folder/'evaluation.json'
            if not result.exists():raise RuntimeError('Evaluator failed; resume original controller for error handling')
            assert hashlib.sha256(policy.read_bytes()).hexdigest()==frozen['sha256']
            record={'status':'restarting','old_pid':args.old_pid,'model':args.model,'job':job,
                'no_active_provider_call':True,'evaluation_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),
                'frozen_policy_sha256':frozen['sha256'],'reason':'Load current game controller after a frozen evaluation boundary',
                'updated':time.time()}
            receipt.write_text(json.dumps(record,indent=2))
            os.kill(args.old_pid,signal.SIGTERM);os.kill(args.old_pid,signal.SIGCONT);stopped=False
            for _ in range(30):
                if not command(args.old_pid):break
                time.sleep(1)
            if command(args.old_pid):raise RuntimeError('Original controller did not terminate')
            launch('games-'+args.model,'wsp_repro.overnight_games',('--model',args.model),python_env='.venv-games')
            record.update(status='restarted',updated=time.time());receipt.write_text(json.dumps(record,indent=2))
            return
    finally:
        if stopped and command(args.old_pid):os.kill(args.old_pid,signal.SIGCONT)


if __name__=='__main__':main()
