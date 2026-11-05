import json
from wsp_repro import aggregate as module


def test_terminal_failure_is_counted_without_fabricating_a_zero_or_a_third_valid_repeat(tmp_path,monkeypatch):
    jobs=[dict(id=f'case-{r}',experiment=5,model='deepseek',game='seaquest',kind='network',width=4,repeat=r)
          for r in range(3)]
    monkeypatch.setattr(module,'make_spec',lambda:{'jobs':jobs})
    for job,mean in zip(jobs,[10.,20.]):
        p=tmp_path/'runs'/job['id'];p.mkdir(parents=True)
        (p/'artifact.json').write_text(json.dumps({'job':job,'transport':'deepseek_api'}))
        (p/'evaluation.json').write_text(json.dumps({'mean':mean,'episode_sample_variance':5.}))
    p=tmp_path/'runs'/jobs[2]['id'];p.mkdir(parents=True)
    (p/'state.json').write_text(json.dumps({'status':'failed','error_type':'RuntimeError','message':'DeepSeek response ended: length'}))
    row=module.aggregate(tmp_path)[0]
    assert row['attempts_finished'] and not row['complete']
    assert row['n']==2 and row['model_failures']==1 and row['pending']==0
    assert row['mean']==15. and row['run_sample_variance']==50.
    assert 'incomplete' in row['variance_scope']
