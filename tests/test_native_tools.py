from types import SimpleNamespace

from rsi_env.models.openai_compat import from_native, to_native
from rsi_env.models.toy import action


def test_text_protocol_maps_to_native_tool_messages():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
            {"role": "assistant", "content": "Looking.\n" + action("ls")},
            {"role": "user", "content": "[tool:bash]\n[exit=0]\na.txt"},
            {"role": "assistant", "content": action("cat a.txt")},
            {"role": "user", "content": "[harness] something"}]
    out = to_native(msgs)
    assert out[0]["content"].startswith("sys") and "native tool calls" in out[0]["content"]
    assert out[2]["tool_calls"][0]["function"]["name"] == "bash" and out[2]["content"] == "Looking."
    assert out[3] == {"role": "tool", "tool_call_id": out[2]["tool_calls"][0]["id"], "content": "[exit=0]\na.txt"}
    assert out[5]["role"] == "tool" and out[6]["role"] == "user"  # dangling call closed before user text


def test_native_response_maps_back_to_action_text():
    call = SimpleNamespace(function=SimpleNamespace(name="bash", arguments='{"command": "ls -la"}'))
    text = from_native(SimpleNamespace(content="", tool_calls=[call]))
    assert text == '<action>{"tool": "bash", "command": "ls -la"}</action>'
    assert from_native(SimpleNamespace(content="Final answer: 4", tool_calls=None)) == "Final answer: 4"
