import json
import shutil

from rsi_env.envcli import main


def test_harness_helpers_avoid_escaping(tmp_path, template_dir, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(template_dir, tmp_path / "active" / "harness")
    (tmp_path / "p.txt").write_text('﻿Use "quotes" and <action> tags; it\'s plain text.', encoding="utf-8")
    assert main(["harness-copy", "--dst", "candidate/x"]) == 0
    assert main(["harness-prompt", "--harness", "candidate/x", "--file", "p.txt", "--append"]) == 0
    assert main(["harness-set", "--harness", "candidate/x", "--key", "max_steps", "--value", "200"]) == 0
    cfg = json.loads((tmp_path / "candidate/x/harness.json").read_text())
    assert cfg["max_steps"] == 200
    assert cfg["system_prompt"].endswith('Use "quotes" and <action> tags; it\'s plain text.')
    assert "﻿" not in cfg["system_prompt"]
