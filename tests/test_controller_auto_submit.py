"""Tests for the controller-side round-end backstop (runner.Experiment._auto_submit_round).

This runs in the controller after a round, reading the round's local-eval audit and submitting
the best validated candidate when the in-worker safeguard was lost (e.g. the worker was killed at
the deadline mid-eval).
"""
import json
from pathlib import Path
from types import SimpleNamespace

from rsi_env.runner import Experiment
from rsi_env.worker import AUTO_SUBMIT_MARGIN


class FakeHost:
    def __init__(self, private, accept="candidate/harness"):
        self.private = Path(private)
        self.accept = accept            # which --harness submit accepts (None = accept any)
        self.accepted_this_round = None
        self.submits = []

    def submit(self, harness="active", model="active", improver=None):
        self.submits.append(harness)
        if self.accept is None or harness == self.accept:
            self.accepted_this_round = "H1/M0"
            return {"accepted": True, "version": "H1/M0"}
        raise RuntimeError("validity check failed")


def _audit(d, rows):
    (Path(d) / "audit.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _ev(round_, harness, score):
    return {"t": 0.0, "round": round_, "event": "local-eval", "harness": harness, "score": score}


def _run(host, r):
    Experiment._auto_submit_round(SimpleNamespace(log=lambda *a, **k: None), host, r)


def test_promotes_best_winner_from_audit(tmp_path):
    host = FakeHost(tmp_path, accept="candidate/harness")
    _audit(tmp_path, [_ev(2, "active", 57.5), _ev(2, "candidate/harness", 80.0),
                      _ev(2, "candidate/harness", 100.0)])
    _run(host, 2)
    assert host.accepted_this_round == "H1/M0"
    assert host.submits == ["candidate/harness"]


def test_only_considers_the_given_round(tmp_path):
    host = FakeHost(tmp_path, accept=None)
    # The 100.0 winner belongs to round 1; round 2 has no beating candidate -> no submit.
    _audit(tmp_path, [_ev(1, "active", 50.0), _ev(1, "candidate/harness", 100.0),
                      _ev(2, "active", 70.0)])
    _run(host, 2)
    assert host.submits == [] and host.accepted_this_round is None


def test_requires_beating_baseline_by_margin(tmp_path):
    host = FakeHost(tmp_path, accept=None)
    _audit(tmp_path, [_ev(2, "active", 70.0),
                      _ev(2, "candidate/harness", 70.0 + AUTO_SUBMIT_MARGIN - 0.5)])
    _run(host, 2)
    assert host.submits == [] and host.accepted_this_round is None


def test_falls_back_to_next_best_on_rejection(tmp_path):
    host = FakeHost(tmp_path, accept="candidate/s2")  # best candidate is broken; second accepted
    _audit(tmp_path, [_ev(3, "active", 70.0), _ev(3, "candidate/broken", 99.0),
                      _ev(3, "candidate/s2", 88.0)])
    _run(host, 3)
    assert host.accepted_this_round == "H1/M0"
    assert host.submits == ["candidate/broken", "candidate/s2"]


def test_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_NO_AUTO_SUBMIT", "1")
    host = FakeHost(tmp_path, accept=None)
    _audit(tmp_path, [_ev(2, "candidate/harness", 100.0)])
    _run(host, 2)
    assert host.submits == [] and host.accepted_this_round is None


def test_no_audit_file_is_safe(tmp_path):
    host = FakeHost(tmp_path, accept=None)
    _run(host, 2)  # no audit.jsonl written
    assert host.submits == []
