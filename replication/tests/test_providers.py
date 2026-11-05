import json
import pytest
from wsp_repro import providers


def test_openai_has_only_explicit_context_and_no_native_tools(tmp_path,monkeypatch):
    monkeypatch.setattr(providers,"credentials",lambda:{"OPENAI_API_KEY":"dummy-not-a-key"})
    seen={}
    def fake_post(url,payload,key):
        seen.update(payload)
        return {"status":"completed","model":"gpt-6-astra","output":[
            {"type":"message","content":[{"type":"output_text","text":"answer"}]}],"usage":{}}
    monkeypatch.setattr(providers,"_post",fake_post)
    history=[{"role":"user","content":"experimental task"}]
    text,meta=providers.call("astra","xhigh",history,tmp_path,transport="openai_api")
    assert text=="answer" and seen["input"]==history
    assert seen["instructions"]=="Answer the user's request." and seen["tools"]==[]
    assert seen["store"] is False and "previous_response_id" not in seen
    assert not {"temperature","top_p","conversation"}&seen.keys()
    assert "dummy-not-a-key" not in (tmp_path/"request.json").read_text()


def test_deepseek_uses_max_and_does_not_replay_hidden_reasoning(tmp_path,monkeypatch):
    monkeypatch.setattr(providers,"credentials",lambda:{"DEEPSEEK_API_KEY":"dummy-not-a-key"})
    seen={}
    def fake_post(url,payload,key):
        seen.update(payload)
        return {"model":"deepseek-flash","choices":[{"finish_reason":"stop","message":{
            "content":"answer","reasoning_content":"not part of subsequent input"}}],"usage":{}}
    monkeypatch.setattr(providers,"_post",fake_post)
    text,meta=providers.call("deepseek","max",[{"role":"user","content":"task"}],tmp_path)
    assert seen["reasoning_effort"]=="max" and seen["thinking"]=={"type":"enabled"}
    assert [m["role"] for m in seen["messages"]]==["system","user"]
    assert "reasoning_content" not in json.dumps(seen) and "tools" not in seen


def test_native_plan_preserves_opaque_same_run_history_without_cli_envelope(tmp_path,monkeypatch):
    monkeypatch.setattr(providers,"credentials",lambda:{})
    seen={}
    output=[{"type":"reasoning","encrypted_content":"opaque-test"},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}]
    def fake_post(url,payload,key):
        seen.update(payload)
        assert url.endswith("/responses") and key=="coding-plan-local-bridge"
        return {"status":"completed","model":"gpt-6-astra","output":output,"usage":{}}
    monkeypatch.setattr(providers,"_post",fake_post)
    history=[{"role":"user","content":"isolated task"}]
    text,meta=providers.call("astra","max",history,tmp_path)
    assert seen["input"]==history and seen["instructions"]=="Answer the user's request."
    assert seen["tools"]==[] and seen["store"] is False
    assert meta["continuation_items"]==output and meta["transport"]=="coding_plan_responses_bridge"
    assert seen["include"]==["reasoning.encrypted_content"]


def test_truncated_reply_is_durable_and_cannot_trigger_another_sample(tmp_path,monkeypatch):
    monkeypatch.setattr(providers,"credentials",lambda:{"DEEPSEEK_API_KEY":"dummy-not-a-key"})
    calls=[]
    def fake_post(url,payload,key):
        calls.append(payload)
        return {"model":"deepseek-flash","choices":[{"finish_reason":"length","message":{
            "content":"partial policy","reasoning_content":"unfinished reasoning"}}],"usage":{"completion_tokens":128000}}
    monkeypatch.setattr(providers,"_post",fake_post)
    history=[{"role":"user","content":"task"}]
    for _ in range(2):
        with pytest.raises(providers.IncompleteModelOutput):providers.call("deepseek","max",history,tmp_path)
    assert len(calls)==1
    failure=json.loads((tmp_path/'call-failure.json').read_text())
    assert failure['terminal_model_output'] and failure['usage']['completion_tokens']==128000
    assert (tmp_path/'partial-answer.md').read_text()=='partial policy'
    assert not (tmp_path/'call.json').exists() and not (tmp_path/'answer.md').exists()
    assert 'dummy-not-a-key' not in (tmp_path/'request.json').read_text()


def test_game_restart_keeps_terminal_failure_without_calling_provider(tmp_path,monkeypatch):
    from wsp_repro import run_games
    monkeypatch.setattr(run_games,'ROOT',tmp_path)
    def forbidden_call(*args,**kwargs):raise AssertionError('Terminal failed case must not be resampled')
    monkeypatch.setattr(run_games,'call',forbidden_call)
    folder=tmp_path/'runs'/'failed-job';folder.mkdir(parents=True)
    original={'status':'failed','error_type':'RuntimeError','message':'DeepSeek response ended: length'}
    (folder/'state.json').write_text(json.dumps(original))
    assert run_games.generate({'id':'failed-job'})==('failed-job','retained_failure')
    assert json.loads((folder/'state.json').read_text())==original
    assert not (folder/'artifact.json').exists() and not (folder/'evaluation.json').exists()
