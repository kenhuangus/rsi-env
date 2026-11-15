"""Tool runtime: executes the harness's ``bash`` tool inside a sandbox directory.

* Commands run with the sandbox as working directory and a scrubbed environment
  (no controller paths, no credentials except the controller-issued session token).
* Full tool outputs are persisted under ``.tool_outputs/``; the model sees a
  truncated view with a pointer to the full file (paper A.1: full tool outputs persist).
* A guard refuses commands that reference protected controller paths and reports
  them as incidents. This is a policy guard, not OS-level isolation; for strong
  isolation run the RSI Environment in a container (see README).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

MAX_VIEW = 12_000


def find_bash() -> str:
    env = os.environ.get("RSI_BASH")
    if env:
        return env
    if os.name == "nt":  # prefer Git for Windows bash over the WSL launcher in System32
        for c in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe",
                  r"C:\Program Files (x86)\Git\bin\bash.exe"):
            if Path(c).exists():
                return c
    found = shutil.which("bash")
    if not found:
        raise RuntimeError("bash not found; install Git for Windows or set RSI_BASH")
    return found


def posix(p: str | Path) -> str:
    return str(p).replace("\\", "/")


class ToolRuntime:
    def __init__(self, cwd: Path, env: dict | None = None, timeout: float = 900.0,
                 protected: list[str] | None = None,
                 on_incident: Callable[[str], None] | None = None,
                 stop_check: Callable[[], bool] | None = None,
                 deadline: float | None = None,
                 persist_outputs: bool = True):
        self.cwd = Path(cwd)
        self.env = env if env is not None else sandbox_env()
        self.timeout = timeout
        self.protected = [p.lower() for p in (protected or []) if p]
        self.on_incident = on_incident
        self.stop_check = stop_check
        self.deadline = deadline
        self.persist = persist_outputs
        self.tool_calls = 0
        self.bash_path = find_bash()

    def should_stop(self) -> bool:
        if self.deadline is not None and time.time() >= self.deadline:
            return True
        return bool(self.stop_check and self.stop_check())

    def _guard(self, command: str) -> str | None:
        low = command.lower().replace("\\", "/")
        for p in self.protected:
            if p in low:
                return p
        return None

    def bash(self, command: str) -> str:
        self.tool_calls += 1
        hit = self._guard(command)
        if hit:
            msg = f"BLOCKED: command references a protected controller path ({hit}). Incident recorded."
            if self.on_incident:
                self.on_incident(f"protected-path access attempt: {command[:300]}")
            return msg
        timeout = self.timeout
        if self.deadline is not None:
            timeout = max(1.0, min(timeout, self.deadline - time.time()))
        try:
            p = subprocess.run([self.bash_path, "-c", command], cwd=self.cwd, env=self.env,
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=timeout)
            out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
            out = f"[exit={p.returncode}]\n{out}"
        except subprocess.TimeoutExpired:
            out = f"[timeout after {timeout:.0f}s]"
        if self.persist:
            d = self.cwd / ".tool_outputs"
            d.mkdir(exist_ok=True)
            f = d / f"{int(time.time() * 1000)}_{self.tool_calls:05d}.txt"
            f.write_text(f"$ {command}\n{out}", encoding="utf-8")
            if len(out) > MAX_VIEW:
                out = out[:MAX_VIEW // 2] + f"\n...[truncated; full output in {posix(f.relative_to(self.cwd))}]...\n" \
                      + out[-MAX_VIEW // 2:]
        elif len(out) > MAX_VIEW:
            out = out[:MAX_VIEW // 2] + "\n...[truncated]...\n" + out[-MAX_VIEW // 2:]
        return out


SAFE_ENV = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE",
            "LANG", "LC_ALL", "COMSPEC", "PATHEXT", "PROGRAMFILES", "APPDATA", "LOCALAPPDATA")


def sandbox_env(extra: dict | None = None, bin_dir: Path | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k.upper() in SAFE_ENV}
    project_root = Path(__file__).resolve().parent.parent
    env["PYTHONPATH"] = str(project_root)
    env["PYTHONIOENCODING"] = "utf-8"
    env["RSI_PYTHON"] = posix(sys.executable)
    if bin_dir is not None:
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    env.update(extra or {})
    return env
