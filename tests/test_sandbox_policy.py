import pytest

from rsi_env.runtime import ToolRuntime, sandbox_env


@pytest.fixture
def rt(tmp_path):
    incidents = []
    r = ToolRuntime(tmp_path, env=sandbox_env(), timeout=30, persist_outputs=False,
                    on_incident=incidents.append, sandbox_policy=True)
    r.incidents = incidents
    return r


@pytest.mark.parametrize("cmd", [
    'find / -name "*.py"', "find /c -name send.py 2>/dev/null", "ls -laR /", "find ~ -type f",
    'python3 -c "import rsi_env; print(1)"; find / -name "*.py" -path "*rsi_env*" | head',
])
def test_filesystem_scans_blocked(rt, cmd):
    assert rt.bash(cmd).startswith("BLOCKED: whole-filesystem")


@pytest.mark.parametrize("cmd", [
    "cat /c/Users/x/rsi-env/rsi_env/domains/research.py",
    'python -c "import inspect, rsi_env.domains.research as r; print(inspect.getsource(r))"',
])
def test_evaluator_source_is_protected(rt, cmd):
    assert rt.bash(cmd).startswith("BLOCKED") and rt.incidents


@pytest.mark.parametrize("cmd", ["find . -name '*.md'", "ls -la experiments/", "echo ok", "find ./data -type f"])
def test_normal_commands_allowed(rt, cmd):
    assert not rt.bash(cmd).startswith("BLOCKED")


def test_policy_off_by_default(tmp_path):
    rt = ToolRuntime(tmp_path, env=sandbox_env(), timeout=5, persist_outputs=False)
    assert not rt.bash("find /nonexistent-dir-xyz 2>/dev/null; echo done").startswith("BLOCKED")
