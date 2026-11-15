"""Experiment Controller: runs the RSI loop S_t -> S_{t+1} (paper Eq. 1) for one task.

Per round t:
  1. choose the Improver: Self -> the active (H_t, M_t); Teacher -> a fixed external system;
  2. run it inside the RSI Environment until it submits an accepted replacement, returns
     TASK_COMPLETE, or exhausts the round budget;
  3. on acceptance the successor S_{t+1} becomes active (and, under Self, the next Improver).
Afterwards every accepted version is evaluated independently on private benchmarks.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import RunConfig
from .controller.host import TEMPLATE, HostController
from .controller.server import ControllerServer
from .tasks import get_task


class Experiment:
    def __init__(self, run_dir: str | Path, cfg: RunConfig, log=print):
        self.run_dir = Path(run_dir).resolve()
        self.cfg = cfg
        self.task = get_task(cfg.task)
        self.log = log

    def _improver(self, host: HostController) -> tuple[Path, dict]:
        if self.cfg.improver == "self":
            h, m = host.registry.split(host.registry.active)
            return host.registry.harness_dir(h), host.registry.model_spec(m)
        spec = self.cfg.teacher_model or {"backend": "toy", "name": "toy-teacher", "profile": "teacher"}
        return Path(self.cfg.teacher_harness or TEMPLATE), spec

    def run(self) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(self.cfg.to_dict(), indent=2), encoding="utf-8")
        host = HostController(self.run_dir, self.task, self.cfg)
        prompt = self.task.prompt(rsi_doc="See docs/INFRA.md. Controller services are available via the `rsi` command.")
        traj_dir = host.private / "trajectories"
        traj_dir.mkdir(exist_ok=True)
        rounds: list[dict] = []
        total_calls, t_start = 0, time.time()
        with ControllerServer(host) as srv:
            for r in range(1, self.cfg.budget.rounds + 1):
                host.begin_round(r)
                hdir, spec = self._improver(host)
                improver_id = host.registry.active if self.cfg.improver == "self" else spec.get("name", "teacher")
                self.log(f"[round {r}] active={host.registry.active} improver={improver_id}")
                res = self._run_round(host, srv, hdir, spec, prompt, r, traj_dir)
                total_calls += res.get("tool_calls", 0)
                if host.accepted_this_round:
                    res["final"] = f"submitted; accepted as {host.accepted_this_round}"
                info = {"round": r, "improver": improver_id, "accepted": host.accepted_this_round,
                        "tool_calls": res.get("tool_calls", 0), "seconds": res.get("seconds", 0),
                        "final": (res.get("final") or "")[-200:], "error": res.get("error")}
                rounds.append(info)
                self.log(f"[round {r}] accepted={info['accepted']} tool_calls={info['tool_calls']} "
                         f"final={info['final'][:80]!r}")
                if host.accepted_this_round:
                    continue
                if (res.get("final") or "").strip().endswith("TASK_COMPLETE"):
                    self.log("[run] Improver returned TASK_COMPLETE")
                    break
        elapsed = time.time() - t_start
        report = self._evaluate(host, rounds, total_calls, elapsed)
        (self.run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (self.run_dir / "report.md").write_text(render_report(report), encoding="utf-8")
        return report

    def _run_round(self, host, srv, hdir, spec, prompt, r, traj_dir) -> dict:
        tmp = host.private / "round_io"
        tmp.mkdir(exist_ok=True)
        (tmp / "task.txt").write_text(prompt, encoding="utf-8")
        (tmp / "model.json").write_text(json.dumps(spec), encoding="utf-8")
        (tmp / "settings.json").write_text(json.dumps(host.settings), encoding="utf-8")
        out = traj_dir / f"round_{r:02d}.json"
        env = dict(os.environ, RSI_URL=srv.url, RSI_TOKEN=srv.token,
                   PYTHONPATH=str(Path(__file__).resolve().parent.parent))
        deadline = time.time() + self.cfg.budget.round_seconds
        args = [sys.executable, "-m", "rsi_env.worker", "improve", "--harness", str(hdir),
                "--model", str(tmp / "model.json"), "--settings", str(tmp / "settings.json"),
                "--task", str(tmp / "task.txt"), "--workspace", str(host.env_root),
                "--protected", json.dumps(host.protected_markers), "--deadline", str(deadline),
                "--command-timeout", str(self.cfg.budget.command_timeout), "--out", str(out)]
        try:
            p = subprocess.run(args, env=env, capture_output=True, text=True,
                               timeout=self.cfg.budget.round_seconds + 120)
        except subprocess.TimeoutExpired:
            return {"error": "round timed out"}
        if not out.exists():
            return {"error": (p.stderr or "improver produced no output")[-1500:]}
        return json.loads(out.read_text(encoding="utf-8"))

    def _evaluate(self, host: HostController, rounds, total_calls, elapsed) -> dict:
        versions = [v["id"] for v in host.registry.lineage()]
        final = host.registry.active
        to_eval = versions if self.cfg.eval_every_version else sorted({versions[0], final}, key=versions.index)
        evals = {}
        for vid in to_eval:
            self.log(f"[eval] {vid} on {', '.join(self.cfg.benchmarks)}")
            evals[vid] = host.evaluate_version(vid)
        return {"task": self.task.name, "setting": self.task.setting, "improver": self.cfg.improver,
                "target_benchmarks": list(self.task.target_benchmarks), "base": versions[0], "final": final,
                "lineage": host.registry.lineage(), "rounds": rounds, "tool_calls": total_calls,
                "hours": round(elapsed / 3600, 4), "seconds": round(elapsed, 1), "evals": evals,
                "incidents": host.incidents(), "trials": self.cfg.eval.trials}


def _fmt(s, base=None) -> str:
    if s is None:
        return "err"
    if base is None:
        return f"{s:.1f}"
    d = s - base
    arrow = "↑" if d > 0 else "↓" if d < 0 else "="
    return f"{s:.1f} ({arrow}{abs(d):.1f})"


def render_report(rep: dict) -> str:
    bms = list(next(iter(rep["evals"].values())).keys())
    base = rep["evals"][rep["base"]]
    lines = [f"# RSI-Env run: task={rep['task']} ({rep['setting']}), improver={rep['improver']}", "",
             f"Final version **{rep['final']}** · tool calls {rep['tool_calls']} · time {rep['seconds']} s · "
             f"trials per item {rep['trials']}", "",
             "| version | " + " | ".join(f"{b} ({base[b].get('metric', '')})" + (" *" if b in rep['target_benchmarks'] else "")
                                       for b in bms) + " |",
             "|---|" + "---|" * len(bms)]
    for vid, ev in rep["evals"].items():
        cells = [_fmt(ev[b].get("score"), None if vid == rep["base"] else base[b].get("score")) for b in bms]
        lines.append(f"| {vid} | " + " | ".join(cells) + " |")
    lines += ["", "\\* target-aligned benchmark", "", "## Rounds", "",
              "| round | improver | accepted | tool calls | seconds | final |", "|---|---|---|---|---|---|"]
    for r in rep["rounds"]:
        lines.append(f"| {r['round']} | {r['improver']} | {r['accepted'] or '-'} | {r['tool_calls']} | "
                     f"{r['seconds']} | {(r['final'] or r.get('error') or '').strip()[:60].replace('|', '/')} |")
    lines += ["", "## Lineage", ""]
    for v in rep["lineage"]:
        lines.append(f"- {v['id']} ← {v['parent'] or 'seed'} (round {v['round']}, by {v['improver']})")
    if rep["incidents"]:
        lines += ["", "## Incidents", ""] + [f"- round {i['round']}: {i['detail'][:160]}" for i in rep["incidents"]]
    return "\n".join(lines) + "\n"
