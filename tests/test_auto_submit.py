"""Tests for the round-end auto-submit safeguard (worker.auto_submit_best)."""
from rsi_env.worker import AUTO_SUBMIT_MARGIN, _best_local_eval_scores, auto_submit_best


def _log(*pairs):
    return [{"cmd": c, "out": o} for c, o in pairs]


class FakeClient:
    """Minimal stand-in for envcli.Client used by the safeguard."""

    def __init__(self, accepted=None, accept_harness=None, reject_error="validity check failed"):
        self.accepted = accepted          # value returned by round_state
        self.accept_harness = accept_harness  # which --harness submit accepts
        self.reject_error = reject_error
        self.submits = []

    def call(self, service, **payload):
        if service == "round_state":
            return {"round": 1, "accepted": self.accepted}
        if service == "submit":
            self.submits.append(payload["harness"])
            if self.accept_harness is None or payload["harness"] == self.accept_harness:
                return {"accepted": True, "version": "H1/M0"}
            raise RuntimeError(self.reject_error)
        raise AssertionError(f"unexpected service {service}")


def test_best_scores_parses_per_harness_max():
    log = _log(
        ('rsi local-eval --data dev.jsonl --harness active --trials 2', '{"score": 65.0, "n": 40}'),
        ('rsi local-eval --data dev.jsonl --harness active --trials 2', '{"score": 70.0, "n": 40}'),
        ('rsi local-eval --data dev.jsonl --harness candidate/harness', '{"score": 82.5, "n": 40}'),
        ('rsi local-eval --data dev.jsonl --harness candidate/harness', '{"score": 85.0, "n": 40}'),
        ('echo not an eval', 'noise'),
    )
    s = _best_local_eval_scores(log)
    assert s == {"active": 70.0, "candidate/harness": 85.0}


def test_auto_submit_promotes_best_winner():
    client = FakeClient(accepted=None, accept_harness="candidate/harness")
    log = _log(
        ('rsi local-eval --harness active', '{"score": 70.0}'),
        ('rsi local-eval --harness candidate/harness', '{"score": 85.0}'),
    )
    r = auto_submit_best(client, log)
    assert r["triggered"] and r["accepted"] == "H1/M0" and r["harness"] == "candidate/harness"
    assert client.submits == ["candidate/harness"]


def test_auto_submit_skips_when_already_accepted():
    client = FakeClient(accepted="H1/M0")
    log = _log(('rsi local-eval --harness candidate/harness', '{"score": 99.0}'))
    r = auto_submit_best(client, log)
    assert r["triggered"] is False and client.submits == []


def test_auto_submit_requires_beating_baseline_by_margin():
    client = FakeClient(accepted=None)
    # candidate only 1.0 above baseline (< margin) -> not a winner
    log = _log(
        ('rsi local-eval --harness active', '{"score": 70.0}'),
        ('rsi local-eval --harness candidate/harness', '{"score": %.1f}' % (70.0 + AUTO_SUBMIT_MARGIN - 1)),
    )
    r = auto_submit_best(client, log)
    assert r["triggered"] is False and client.submits == []


def test_auto_submit_falls_back_to_next_best_on_rejection():
    # best candidate is broken (rejected); the safeguard then tries the next-best, which is accepted
    client = FakeClient(accepted=None, accept_harness="candidate/s2")
    log = _log(
        ('rsi local-eval --harness active', '{"score": 70.0}'),
        ('rsi local-eval --harness candidate/broken', '{"score": 95.0}'),
        ('rsi local-eval --harness candidate/s2', '{"score": 88.0}'),
    )
    r = auto_submit_best(client, log)
    assert r["triggered"] and r["accepted"] == "H1/M0" and r["harness"] == "candidate/s2"
    assert client.submits == ["candidate/broken", "candidate/s2"]  # tried best first, then fell back


def test_auto_submit_disabled_by_env(monkeypatch):
    monkeypatch.setenv("RSI_NO_AUTO_SUBMIT", "1")
    client = FakeClient(accepted=None, accept_harness="candidate/harness")
    log = _log(('rsi local-eval --harness candidate/harness', '{"score": 99.0}'))
    r = auto_submit_best(client, log)
    assert r["triggered"] is False and "disabled" in r["reason"] and client.submits == []
