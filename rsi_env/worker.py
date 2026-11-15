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


def cmd_eval(a) -> dict:
    model = load_model(json.loads(Path(a.model).read_text(encoding="utf-8")))
    settings = json.loads(Path(a.settings).read_text(encoding="utf-8")) if a.settings else {}
    items = [json.loads(x) for x in Path(a.items).read_text(encoding="utf-8").splitlines() if x.strip()]
    per_item: dict[str, list[bool]] = {}
    responses: dict[str, list[str]] = {}
    tool_calls = 0
    for it in items:
        iid = str(it.get("id"))
        for trial in range(a.trials):
            with tempfile.TemporaryDirectory() as tmp:
                d = Path(tmp)
                for name, content in (it.get("files") or {}).items():
                    (d / name).write_text(content, encoding="utf-8")
                rt = ToolRuntime(d, env=sandbox_env(), timeout=120, persist_outputs=False)
                try:
                    h = build_harness(a.harness, model, rt, settings)
                    resp = h.solve(it["prompt"], _sampling(settings, trial))
                except Exception as e:  # noqa: BLE001 - a crashing harness fails the item
                    resp = f"[harness error] {e}"
                tool_calls += rt.tool_calls
                if it.get("kind") == "terminal" or "check" in it:
                    if "check" in it:
                        ok = "[exit=0]" in ToolRuntime(d, env=sandbox_env(), timeout=60,
                                                        persist_outputs=False).bash(it["check"])
                    else:
                        ok = bool(terminal_verifier(int(it.get("suite_seed", 7)), int(it["tb_index"]))(d))
                else:
                    ok = grade(Item(iid, it["prompt"], str(it.get("answer", "")), kind=it.get("kind", "qa")), resp)
            per_item.setdefault(iid, []).append(bool(ok))
            responses.setdefault(iid, []).append((resp or "")[-300:])
    return {"per_item": per_item, "tool_calls": tool_calls,
            "responses": responses if a.keep_responses else {}}


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
        deadline=a.deadline)
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
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"check": cmd_check, "eval": cmd_eval, "improve": cmd_improve}[a.mode](a)
    Path(a.out).write_text(json.dumps(res), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
