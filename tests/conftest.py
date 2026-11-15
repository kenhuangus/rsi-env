import shutil
from pathlib import Path

import pytest

from rsi_env.config import Budget, EvalSettings, RunConfig
from rsi_env.controller.host import TEMPLATE, HostController
from rsi_env.tasks import get_task


def make_cfg(task="knowledge", improver="self", rounds=4, trials=1, benchmarks=("mini-simpleqa",)):
    return RunConfig(task=task, improver=improver, benchmarks=list(benchmarks),
                     budget=Budget(rounds=rounds, round_seconds=600), eval=EvalSettings(trials=trials))


@pytest.fixture
def host_factory(tmp_path):
    def make(task="knowledge", **kw):
        cfg = make_cfg(task, **kw)
        h = HostController(tmp_path / f"run-{task}", get_task(task), cfg)
        h.begin_round(1)
        return h
    return make


@pytest.fixture
def candidate(tmp_path):
    """Copy the minimal harness into a host's workspace as candidate/harness."""
    def make(host, name="candidate/harness"):
        dst = host.env_root / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(TEMPLATE, dst)
        return dst
    return make


class ScriptedModel:
    """A tiny fake model that replays a list of replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.spec = {}

    def generate(self, messages, sampling=None):
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0) if self.replies else "Final answer: done"


@pytest.fixture
def template_dir() -> Path:
    return TEMPLATE
