import json
import pytest
from wsp_repro.aggregate_supplementary import stats
from wsp_repro import run_reading
from wsp_repro.providers import IncompleteModelOutput


def test_missing_output_is_excluded_from_mean_and_ddof1_variance():
    result = stats([0.0, None, 1.0])
    assert result['n'] == 2 and result['failures'] == 1
    assert result['mean'] == .5 and result['sample_variance'] == .5
    assert stats([None, 1.0, None])['sample_variance'] is None
    assert stats([None])['mean'] is None


def test_empty_cached_response_is_not_scored_or_regenerated(tmp_path, monkeypatch):
    job = {'id': 'empty', 'experiment': 3, 'case': 'python', 'model': 'deepseek', 'effort': 'max'}
    folder = tmp_path / 'replication/runs/empty/turn-000'
    folder.mkdir(parents=True)
    (folder / 'answer.md').write_text('')
    (folder / 'call.json').write_text(json.dumps({'transport': 'deepseek_api'}))
    monkeypatch.setattr(run_reading, 'ROOT', tmp_path)
    monkeypatch.setattr(run_reading, 'prompt_for', lambda _: 'Prompt')
    monkeypatch.setattr(run_reading, 'call', lambda *a, **kw: pytest.fail('Must reuse cached response'))
    with pytest.raises(IncompleteModelOutput, match='Empty final content'):
        run_reading.generate(job)
    assert not (folder.parent / 'result.json').exists()
    assert (folder / 'answer.md').read_text() == ''
