import json
import shutil

from conftest import ScriptedModel

from rsi_env.models.toy import ToyModel, action
from rsi_env.models.toy_policy import PC_EXT, SC_EXT
from rsi_env.runtime import ToolRuntime, sandbox_env
from rsi_env.worker import build_harness


def _rt(tmp_path, **kw):
    return ToolRuntime(tmp_path, env=sandbox_env(), timeout=30, **kw)


def test_minimal_harness_executes_one_action_per_step(tmp_path, template_dir):
    model = ScriptedModel([action("echo hello > out.txt && cat out.txt"), "Final answer: ok"])
    h = build_harness(template_dir, model, _rt(tmp_path), {})
    res = h.run("make a file")
    assert res["final"] == "Final answer: ok" and res["tool_calls"] == 1
    assert (tmp_path / "out.txt").read_text().strip() == "hello"
    assert "[tool:bash]" in model.seen[1][-1]["content"] and "hello" in model.seen[1][-1]["content"]


def test_full_tool_outputs_persist(tmp_path, template_dir):
    model = ScriptedModel([action("python -c \"print('x'*20000)\""), "Final answer: ok"])
    h = build_harness(template_dir, model, _rt(tmp_path), {})
    h.run("big output")
    files = list((tmp_path / ".tool_outputs").glob("*.txt"))
    assert files and len(files[0].read_text()) > 20000
    assert "truncated; full output in" in model.seen[1][-1]["content"]


def test_automatic_compaction_retains_original_task(tmp_path, template_dir):
    replies = [action("echo " + "y" * 50) for _ in range(4)] + ["summary", "Final answer: ok"]
    model = ScriptedModel(replies)
    h = build_harness(template_dir, model, _rt(tmp_path), {"compaction_threshold": 60})
    h.run("ORIGINAL TASK TEXT")
    compact_calls = [m for m in model.seen if m[-1]["content"].startswith("[compact]")]
    assert compact_calls, "compaction should trigger above the threshold"
    after = model.seen[-1]
    assert any(m["content"] == "ORIGINAL TASK TEXT" for m in after)


def test_bad_action_json_is_reported(tmp_path, template_dir):
    model = ScriptedModel(["<action>{not json}</action>", "Final answer: ok"])
    h = build_harness(template_dir, model, _rt(tmp_path), {})
    assert h.run("t")["final"] == "Final answer: ok"
    assert "invalid action JSON" in model.seen[1][-1]["content"]


def test_protected_path_guard_records_incident(tmp_path):
    incidents = []
    rt = _rt(tmp_path, protected=["controller_private"], on_incident=incidents.append)
    out = rt.bash("cat ../controller_private/registry.json")
    assert out.startswith("BLOCKED") and incidents


def _harness_with(tmp_path, template_dir, exts):
    d = tmp_path / "h"
    shutil.copytree(template_dir, d)
    cfg = json.loads((d / "harness.json").read_text())
    for name, code in exts:
        (d / f"{name}.py").write_text(code)
        cfg["extensions"].append(name)
    (d / "harness.json").write_text(json.dumps(cfg))
    return d


def _acc(harness_dir, tmp_path, n=60):
    from rsi_env.benchmarks.suites import get_suite, grade
    model = ToyModel({"backend": "toy", "seed": 3})
    h = build_harness(harness_dir, model, _rt(tmp_path, persist_outputs=False), {})
    items = get_suite("mini-aime").items * 2
    return sum(grade(it, h.solve(it.prompt, {"seed": i})) for i, it in enumerate(items[:n])) / n


def test_self_consistency_extension_improves_toy_math(tmp_path, template_dir):
    base = _acc(template_dir, tmp_path)
    voted = _acc(_harness_with(tmp_path, template_dir, [("self_consistency", SC_EXT)]), tmp_path)
    assert voted > base


def test_extensions_compose_in_either_order(tmp_path, template_dir):
    d = _harness_with(tmp_path, template_dir, [("self_consistency", SC_EXT), ("postcondition", PC_EXT)])
    base = _acc(template_dir, tmp_path)
    assert _acc(d, tmp_path) > base  # voting still active under the outer postcondition extension
