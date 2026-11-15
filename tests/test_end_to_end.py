"""End-to-end RSI runs with the offline toy backend."""
from conftest import make_cfg

from rsi_env.runner import Experiment


def test_knowledge_self_recursive_continuation(tmp_path):
    """Mirrors the paper's case study: H0/M0 -> H1/M0 (harness) -> H1/M1 (SFT by the updated system)."""
    rep = Experiment(tmp_path / "k", make_cfg("knowledge", "self", rounds=5), log=lambda *_: None).run()
    ids = [v["id"] for v in rep["lineage"]]
    assert ids == ["H0/M0", "H1/M0", "H1/M1"]
    # under Self the improver of round t+1 is the system accepted in round t
    assert [r["improver"] for r in rep["rounds"][:3]] == ["H0/M0", "H1/M0", "H1/M1"]
    assert rep["rounds"][-1]["final"].strip() == "TASK_COMPLETE"
    sqa = {v: rep["evals"][v]["mini-simpleqa"]["score"] for v in ids}
    assert sqa["H1/M1"] > sqa["H0/M0"]
    journal = (tmp_path / "k" / "rsi_environment" / "journal.md").read_text()
    assert journal.count("[op-done]") == 2 and "Round 3 - conclusion" in journal
    assert not rep["incidents"]


def test_teacher_is_fixed_and_never_trains(tmp_path):
    rep = Experiment(tmp_path / "t", make_cfg("knowledge", "teacher", rounds=4), log=lambda *_: None).run()
    assert rep["final"].endswith("/M0")
    assert {r["improver"] for r in rep["rounds"]} == {"toy-teacher"}


def test_benchmark_guided_uses_feedback(tmp_path):
    rep = Experiment(tmp_path / "bg", make_cfg("bg-terminal", "self", rounds=3,
                                               benchmarks=("mini-terminal",)), log=lambda *_: None).run()
    audit = (tmp_path / "bg" / "controller_private" / "audit.jsonl").read_text()
    assert '"event": "feedback"' in audit
    assert rep["final"] == "H1/M0"
