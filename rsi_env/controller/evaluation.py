"""Independent capability evaluation (controller-side; the paper uses Harbor).

Evaluation is never used as the acceptance criterion for intermediate updates.
It runs the harness in an isolated worker process against private suites and
reports avg@k / pass@k.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from ..benchmarks.suites import Suite, all_k, avg_at_k, get_suite, pass_at_k
from ..runtime import sandbox_env


def run_worker(mode: str, harness_dir: Path, model_spec: dict, work_dir: Path, *, items: list[dict] | None = None,
               trials: int = 3, settings: dict | None = None, timeout: float | None = None,
               transcripts: Path | None = None) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        t = Path(tmp)
        (t / "model.json").write_text(json.dumps(model_spec), encoding="utf-8")
        (t / "settings.json").write_text(json.dumps(settings or {}), encoding="utf-8")
        args = [sys.executable, "-m", "rsi_env.worker", mode, "--harness", str(harness_dir),
                "--model", str(t / "model.json"), "--settings", str(t / "settings.json"),
                "--trials", str(trials), "--out", str(t / "out.json")]
        if items is not None:
            (t / "items.jsonl").write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
            args += ["--items", str(t / "items.jsonl")]
        if transcripts is not None:
            args += ["--transcripts", str(Path(transcripts).resolve())]
        env = sandbox_env()
        env.update({k: v for k, v in __import__("os").environ.items()
                    if k.endswith(("_API_KEY", "_BASE_URL")) or k.startswith("ANTHROPIC_")})  # controller-side only
        p = subprocess.run(args, cwd=t, env=env, capture_output=True, text=True, timeout=timeout)
        out = t / "out.json"
        if not out.exists():
            return {"ok": False, "errors": [p.stderr[-2000:] or "worker produced no output"]}
        return json.loads(out.read_text(encoding="utf-8"))


def suite_items(suite: Suite, seed: int = 7) -> list[dict]:
    rows = []
    for it in suite.items:
        if "row" in it.meta:          # real-world domain items carry their full spec
            rows.append(dict(it.meta["row"]))
            continue
        row = {"id": it.id, "prompt": it.prompt, "answer": it.answer, "kind": it.kind}
        if it.kind == "terminal":
            row.update(files=it.meta["files"], tb_index=it.meta["index"], suite_seed=seed)
        rows.append(row)
    return rows


def score(metric: str, per_item: dict[str, list[bool]]) -> dict:
    trials = list(per_item.values())
    main = pass_at_k(trials) if metric == "pass@k" else avg_at_k(trials)
    return {"score": round(main, 1), "all_k": round(all_k(trials), 1), "n": len(trials)}


class Evaluator:
    def __init__(self, work_dir: Path, settings: dict, trials: int = 3, seed: int = 7, salt: str = ""):
        self.work_dir, self.settings, self.trials, self.seed = Path(work_dir), settings, trials, seed
        self.salt = salt

    def evaluate(self, harness_dir: Path, model_spec: dict, benchmarks: list[str], tag: str = "") -> dict:
        out = {}
        for name in benchmarks:
            suite = get_suite(name, self.seed, self.salt)
            tdir = self.work_dir / "transcripts" / (tag.replace("/", "_") or "eval") / name
            res = run_worker("eval", harness_dir, model_spec, self.work_dir,
                             items=suite_items(suite, self.seed), trials=self.trials, settings=self.settings,
                             transcripts=tdir)
            if "per_item" not in res:
                out[name] = {"error": res.get("errors"), "score": None, "metric": suite.metric}
                continue
            out[name] = {**score(suite.metric, res["per_item"]), "metric": suite.metric.replace("k", str(self.trials)),
                         "tool_calls": res.get("tool_calls", 0), "per_item": res["per_item"]}
        return out
