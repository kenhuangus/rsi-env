"""Command-line entry point: ``rsi-env`` (or ``python -m rsi_env``)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import Budget, EvalSettings, RunConfig
from .runner import Experiment, _fmt
from .tasks import TASKS, get_task


def _spec(arg: str | None, default: dict | None) -> dict | None:
    if not arg:
        return default
    p = Path(arg)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else json.loads(arg)


def make_config(a) -> RunConfig:
    return RunConfig(
        task=a.task, improver=a.improver,
        target_model=_spec(a.target_model, {"backend": "toy", "name": "toy-base", "seed": a.seed}),
        teacher_model=_spec(a.teacher_model, None), teacher_harness=a.teacher_harness,
        benchmarks=(a.benchmarks or DOMAIN_BENCHMARKS.get(a.task, DEFAULT_BENCHMARKS)).split(","),
        budget=Budget(rounds=a.rounds, round_seconds=a.round_hours * 3600, command_timeout=a.command_timeout),
        eval=EvalSettings(trials=a.trials, concurrency=a.concurrency, context_window=a.context_window,
                          compaction_threshold=int(a.context_window * 0.9)),
        seed=a.seed, eval_every_version=not a.final_only)


DEFAULT_BENCHMARKS = "mini-gpqa,mini-aime,mini-terminal,mini-simpleqa"
DOMAIN_BENCHMARKS = {"redteam": "redteam-ctf", "research": "research-qa", "research-pro": "research-pro"}


def cmd_run(a) -> None:
    cfg = make_config(a)
    out = Path(a.out or f"runs/{a.task}-{a.improver}-{time.strftime('%Y%m%d-%H%M%S')}")
    rep = Experiment(out, cfg).run()
    print((out / "report.md").read_text(encoding="utf-8"))
    print(f"run directory: {out.resolve()}")
    return rep


def cmd_matrix(a) -> None:
    """Run every task x improver (Table 1 + Figure 3 analogue) and write a combined table."""
    out = Path(a.out or f"runs/matrix-{time.strftime('%Y%m%d-%H%M%S')}")
    rows = []
    for task in a.tasks.split(","):
        for imp in a.improvers.split(","):
            a.task, a.improver = task, imp
            cfg = make_config(a)
            cfg.eval_every_version = False
            rep = Experiment(out / f"{task}-{imp}", cfg, log=lambda *_: None).run()
            rows.append(rep)
            print(f"done {task}/{imp}: final {rep['final']}")
    bms = rows[0]["evals"][rows[0]["base"]].keys()
    lines = ["| task | setting | improver | " + " | ".join(bms) + " | final (H/M) | tool calls | seconds |",
             "|---|---|---|" + "---|" * len(bms) + "---|---|---|"]
    base = rows[0]["evals"][rows[0]["base"]]
    lines.append("| base | – | seed | " + " | ".join(_fmt(base[b]["score"]) for b in bms) + " | H0/M0 | n/a | n/a |")
    for r in rows:
        ev = r["evals"][r["final"]]
        b0 = r["evals"][r["base"]]
        cells = [_fmt(ev[b]["score"], b0[b]["score"]) + (" *" if b in r["target_benchmarks"] else "") for b in bms]
        lines.append(f"| {r['task']} | {r['setting']} | {r['improver']} | " + " | ".join(cells)
                     + f" | {r['final']} | {r['tool_calls']} | {r['seconds']} |")
    table = "\n".join(lines) + "\n\n\\* target-aligned benchmark\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrix.md").write_text(table, encoding="utf-8")
    print(table)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rsi-env", description="RSI-Env: open-ended recursive self-improvement")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--rounds", type=int, default=10)
        p.add_argument("--round-hours", type=float, default=3.0)
        p.add_argument("--trials", type=int, default=3)
        p.add_argument("--benchmarks", help="comma list; default depends on the task")
        p.add_argument("--concurrency", type=int, default=1, help="parallel evaluation trials")
        p.add_argument("--context-window", type=int, default=262_144)
        p.add_argument("--command-timeout", type=float, default=1800.0, help="per bash command (s)")
        p.add_argument("--target-model", help="model spec JSON (file or inline); default: toy")
        p.add_argument("--teacher-model", help="Teacher model spec JSON (file or inline)")
        p.add_argument("--teacher-harness", help="Teacher harness directory (default: minimal harness)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--final-only", action="store_true", help="evaluate only base and final versions")
        p.add_argument("--out")

    p = sub.add_parser("run", help="run one task with one Improver configuration")
    p.add_argument("--task", default="knowledge", choices=sorted(TASKS))
    p.add_argument("--improver", default="self", choices=["self", "teacher"])
    common(p)
    p = sub.add_parser("matrix", help="run several tasks x improvers and tabulate")
    p.add_argument("--tasks", default=",".join(t for t in TASKS if t not in ("redteam", "research", "research-pro")))
    p.add_argument("--improvers", default="self,teacher")
    common(p)
    sub.add_parser("tasks", help="list the seven tasks")
    p = sub.add_parser("prompt", help="print a task prompt (Appendix B)")
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    a = ap.parse_args(argv)

    if a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "matrix":
        cmd_matrix(a)
    elif a.cmd == "tasks":
        for t in TASKS.values():
            print(f"{t.name:<12} {t.setting:<20} capability={t.capability:<10} feedback={list(t.feedback_benchmarks)}")
    elif a.cmd == "prompt":
        print(get_task(a.task).prompt())
    return 0
