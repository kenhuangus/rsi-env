"""Isolated worker process that loads a (possibly Improver-written) harness + model.

The Host Controller never imports candidate harness code into its own process;
it launches this worker instead:

    python -m rsi_env.worker check --harness DIR --model SPEC.json --out OUT.json
    python -m rsi_env.worker eval  --harness DIR --model SPEC.json --items ITEMS.jsonl
                                   --trials 3 --settings SETTINGS.json --out OUT.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import traceback
from pathlib import Path

from .benchmarks.suites import Item, grade, terminal_verifier
from .models import load_model
from .runtime import ToolRuntime, sandbox_env


def load_harness_module(harness_dir: str | Path):
    path = Path(harness_dir) / "harness.py"
    spec = importlib.util.spec_from_file_location(f"rsi_harness_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_harness(harness_dir, model, runtime, settings):
    mod = load_harness_module(harness_dir)
    if not hasattr(mod, "build"):
        raise TypeError("harness.py must define build(model, runtime, settings, harness_dir)")
    h = mod.build(model, runtime, settings, str(harness_dir))
    for attr in ("run", "solve"):
        if not callable(getattr(h, attr, None)):
            raise TypeError(f"harness object must provide callable .{attr}()")
    return h


def _sampling(settings: dict, trial: int) -> dict:
    s = {k: settings[k] for k in ("temperature", "top_p", "top_k") if k in settings}
    s["max_tokens"] = settings.get("max_output_tokens", 81_920)
    s["seed"] = trial
    return s


def cmd_check(a) -> dict:
    """Load-and-sample check: establishes executability, not capability (paper B.1)."""
    errors = []
    sample = None
    try:
        model = load_model(json.loads(Path(a.model).read_text(encoding="utf-8")))
        settings = json.loads(Path(a.settings).read_text(encoding="utf-8")) if a.settings else {}
        with tempfile.TemporaryDirectory() as tmp:
            rt = ToolRuntime(Path(tmp), env=sandbox_env(), timeout=60, persist_outputs=False)
            h = build_harness(a.harness, model, rt, settings)
            sample = h.solve("Compute: (12*12 + 1) mod 50. Give the integer answer.", _sampling(settings, 0))
            if not isinstance(sample, str):
                errors.append(f"solve() returned {type(sample).__name__}, expected str")
            echo = rt.bash("echo rsi-env-ok")
            if "rsi-env-ok" not in echo:
                errors.append("bash tool execution failed")
    except Exception:  # noqa: BLE001 - report every failure as a validity error
        errors.append(traceback.format_exc(limit=4))
    return {"ok": not errors, "errors": errors, "sample": (sample or "")[:500]}


def _run_trial(a, it: dict, trial: int, model, settings: dict) -> dict:
    from contextlib import nullcontext

    from .domains import redteam, research, research_pro

    iid = str(it.get("id"))
    kind = it.get("kind", "qa")
    target = redteam.TargetServer(it["challenge"]) if kind == "redteam" else None
    with tempfile.TemporaryDirectory() as tmp, (target or nullcontext()):
        d = Path(tmp)
        for name, content in (it.get("files") or {}).items():
            (d / name).write_text(content, encoding="utf-8")
        if target is not None:
            target.install(d)
        if kind == "research":
            research.install_tools(d)
        if kind == "research_pro":
            research_pro.install_tools(d)
        if kind == "research_ext":
            from .benchmarks.published import context_corpus
            from .domains.research import FETCH_PY, SEARCH_PY
            (d / "corpus.json").write_text(json.dumps(context_corpus(it)), encoding="utf-8")
            (d / "search.py").write_text(SEARCH_PY, encoding="utf-8")
            (d / "fetch.py").write_text(FETCH_PY, encoding="utf-8")
        budget = settings.get("eval_tool_budget")
        import time as _time
        rt = ToolRuntime(d, env=sandbox_env(), timeout=120, persist_outputs=False,
                         max_tool_calls=int(budget) if budget else None,
                         deadline=_time.time() + float(settings.get("eval_trial_seconds", 900)))
        prompt = it.get("prompt") or (research.task_prompt(it["question"]) if kind == "research" else "")
        try:
            h = build_harness(a.harness, model, rt, settings)
            resp = h.solve(prompt, _sampling(settings, trial))
        except Exception as e:  # noqa: BLE001 - a crashing harness fails the item
            resp = f"[harness error] {type(e).__name__}: {e}"
        if kind == "redteam":
            ok = redteam.grade(it["challenge"], resp)
        elif kind == "research":
            ok = research.grade(it, resp)
        elif kind == "research_pro":
            ok = research_pro.grade(it, resp)
        elif kind == "research_ext":
            from .benchmarks.published import grade_squad
            ok = grade_squad(it, resp)
        elif kind == "terminal" or "check" in it:
            if "check" in it:
                ok = "[exit=0]" in ToolRuntime(d, env=sandbox_env(), timeout=60,
                                                persist_outputs=False).bash(it["check"])
            else:
                ok = bool(terminal_verifier(int(it.get("suite_seed", 7)), int(it["tb_index"]))(d))
        else:
            ok = grade(Item(iid, prompt, str(it.get("answer", "")), kind=kind), resp)
    if a.transcripts:
        out = Path(a.transcripts)
        out.mkdir(parents=True, exist_ok=True)
        lines = [f"# {iid} trial {trial}: {'PASS' if ok else 'FAIL'}", "", "## Task", prompt, "", "## Tool calls"]
        for e in rt.log:
            lines += [f"$ {e['cmd']}", e["out"], ""]
        if target is not None:
            lines += ["## Target conversation"] + [f"> {t['message']}" + (" [doc]" if t["doc"] else "")
                                                     + f"\n< {t['reply']}" for t in target.transcript] + [""]
        lines += ["## Final response", resp or ""]
        (out / f"{iid}_t{trial}.md").write_text("\n".join(lines), encoding="utf-8")
    return {"id": iid, "ok": bool(ok), "tool_calls": rt.tool_calls, "resp": (resp or "")[-300:]}


def cmd_eval(a) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    model = load_model(json.loads(Path(a.model).read_text(encoding="utf-8")))
    settings = json.loads(Path(a.settings).read_text(encoding="utf-8")) if a.settings else {}
    items = [json.loads(x) for x in Path(a.items).read_text(encoding="utf-8").splitlines() if x.strip()]
    jobs = [(it, trial) for it in items for trial in range(a.trials)]
    workers = max(1, int(settings.get("eval_concurrency", 1)))
    if workers == 1:
        results = [_run_trial(a, it, t, model, settings) for it, t in jobs]
    else:
        with ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(lambda j: _run_trial(a, j[0], j[1], model, settings), jobs))
    per_item: dict[str, list[bool]] = {}
    responses: dict[str, list[str]] = {}
    for r in results:
        per_item.setdefault(r["id"], []).append(r["ok"])
        responses.setdefault(r["id"], []).append(r["resp"])
    return {"per_item": per_item, "tool_calls": sum(r["tool_calls"] for r in results),
            "responses": responses if a.keep_responses else {},
            "usage": getattr(model, "usage", None)}


def cmd_improve(a) -> dict:
    """One improvement round: the Improver's harness+model act inside the RSI Environment."""
    import os
    import time

    from .envcli import Client

    model = load_model(json.loads(Path(a.model).read_text(encoding="utf-8")))
    settings = json.loads(Path(a.settings).read_text(encoding="utf-8")) if a.settings else {}
    task = Path(a.task).read_text(encoding="utf-8")
    ws = Path(a.workspace)
    client = Client.from_env()
    rt = ToolRuntime(
        ws, env=sandbox_env({"RSI_URL": os.environ["RSI_URL"], "RSI_TOKEN": os.environ["RSI_TOKEN"]},
                            bin_dir=ws / "bin"),
        timeout=a.command_timeout, protected=json.loads(a.protected),
        on_incident=lambda d: client.call("report_incident", detail=d),
        stop_check=lambda: client.call("round_state")["accepted"] is not None,
        deadline=a.deadline, sandbox_policy=True)
    t0 = time.time()
    h = build_harness(a.harness, model, rt, settings)
    res = h.run(task, _sampling(settings, 0))
    return {"final": res.get("final", ""), "steps": res.get("steps", 0), "tool_calls": rt.tool_calls,
            "seconds": round(time.time() - t0, 2), "messages": res.get("messages", [])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rsi_env.worker")
    ap.add_argument("mode", choices=["check", "eval", "improve"])
    ap.add_argument("--task")
    ap.add_argument("--workspace")
    ap.add_argument("--protected", default="[]")
    ap.add_argument("--deadline", type=float)
    ap.add_argument("--command-timeout", type=float, default=900.0)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--settings")
    ap.add_argument("--items")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--keep-responses", action="store_true")
    ap.add_argument("--transcripts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"check": cmd_check, "eval": cmd_eval, "improve": cmd_improve}[a.mode](a)
    Path(a.out).write_text(json.dumps(res), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
