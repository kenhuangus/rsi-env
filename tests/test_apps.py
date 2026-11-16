"""Tests for the practical-use app layer (rsi_env/apps.py). Offline plumbing only."""
import json

import pytest

from rsi_env import apps


def test_read_docs_builds_corpus(tmp_path):
    (tmp_path / "a.md").write_text("# Alpha\nThe Zephyr engine was built in Marport.", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("x" * 3200, encoding="utf-8")  # forces chunking
    (tmp_path / "c.jsonl").write_text(
        json.dumps({"title": "Rec1", "text": "hello world"}) + "\n" +
        json.dumps({"id": "r2", "content": "second record"}) + "\n", encoding="utf-8")
    docs = apps._read_docs(tmp_path)
    ids = [d["id"] for d in docs]
    assert len(ids) == len(set(ids))                       # ids unique
    assert any("hello world" in d["text"] for d in docs)   # jsonl field used
    assert sum(1 for d in docs if d["id"].startswith("sub-b")) >= 3  # 3200 chars -> >=3 chunks
    assert all({"id", "title", "text"} <= set(d) for d in docs)


def test_read_docs_errors_when_empty(tmp_path):
    (tmp_path / "pic.png").write_bytes(b"\x89PNG")
    with pytest.raises(SystemExit):
        apps._read_docs(tmp_path)


def test_resolve_harness_default_is_seed():
    h = apps.resolve_harness(None, None)
    assert (h / "harness.py").exists() and (h / "harness.json").exists()


def test_resolve_harness_from_run(tmp_path):
    active = tmp_path / "rsi_environment" / "active" / "harness"
    active.mkdir(parents=True)
    (active / "harness.py").write_text("def build(*a): ...", encoding="utf-8")
    (active / "harness.json").write_text("{}", encoding="utf-8")
    (tmp_path / "rsi_environment" / "active" / "VERSION").write_text("h3/m0\n", encoding="utf-8")
    h = apps.resolve_harness(None, str(tmp_path))
    assert h == active.resolve()
    assert "h3/m0" in apps.harness_label(h, str(tmp_path))


def test_resolve_harness_missing_raises(tmp_path):
    with pytest.raises(SystemExit):
        apps.resolve_harness(str(tmp_path), None)


def test_redteam_send_py_is_valid_python_and_targets_endpoint():
    src = apps.SEND_PY_REAL.format(url="https://app.example/chat", msg_field="prompt", doc_field="document",
                                   reply_path=["choices", "0", "text"], extra={"session": "x"}, headers={"X-Auth": "t"})
    compile(src, "send.py", "exec")                  # generated tool is syntactically valid
    assert "https://app.example/chat" in src and "'prompt'" in src and "X-Auth" in src


def test_redteam_requires_authorization():
    a = apps.main
    with pytest.raises(SystemExit):
        apps.main(["redteam", "--endpoint", "http://localhost:9/x", "--goal", "test", "--model", "{}"])


def test_research_prompt_mentions_tools_and_final_answer():
    p = apps.RESEARCH_PROMPT.format(budget=25, q="Q?")
    assert "search.py" in p and "fetch.py" in p and "Final answer:" in p
