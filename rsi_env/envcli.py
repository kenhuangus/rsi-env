"""``rsi`` - the Improver-side client for the controlled interfaces.

Inside the RSI Environment the session URL and token are provided through the
``RSI_URL`` / ``RSI_TOKEN`` environment variables. Usage::

    rsi status
    rsi lineage
    rsi infer --prompt "What is ...?" [--model active|cand-0001] [--seed 1]
    rsi local-eval --data data/local_val.jsonl [--harness candidate/harness] [--model active] [--trials 2]
    rsi train-sft --data data/sft.jsonl [--epochs 2] [--lr 1e-4]
    rsi submit [--harness candidate/harness] [--model cand-0001]
    rsi feedback --benchmark NAME [--harness DIR] [--model ID]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


class RSIError(RuntimeError):
    pass


class Client:
    def __init__(self, url: str, token: str):
        self.url, self.token = url.rstrip("/"), token

    @classmethod
    def from_env(cls) -> "Client":
        try:
            return cls(os.environ["RSI_URL"], os.environ["RSI_TOKEN"])
        except KeyError:
            raise RSIError("RSI_URL / RSI_TOKEN are not set (are you inside the RSI Environment?)") from None

    def call(self, service: str, **payload) -> dict:
        req = urllib.request.Request(f"{self.url}/{service}", data=json.dumps(payload).encode(),
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RSIError(json.loads(e.read() or b"{}").get("error", str(e))) from None

    def infer(self, prompt: str, model: str = "active", seed: int = 0) -> str:
        return self.call("infer", prompt=prompt, model=model, seed=seed)["text"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rsi")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("lineage")
    sub.add_parser("docs")
    p = sub.add_parser("infer"); p.add_argument("--prompt", required=True)  # noqa: E702
    p.add_argument("--model", default="active"); p.add_argument("--seed", type=int, default=0)  # noqa: E702
    p = sub.add_parser("local-eval"); p.add_argument("--data", required=True)  # noqa: E702
    p.add_argument("--harness", default="active"); p.add_argument("--model", default="active")  # noqa: E702
    p.add_argument("--trials", type=int, default=2)
    p = sub.add_parser("train-sft"); p.add_argument("--data", required=True)  # noqa: E702
    p.add_argument("--epochs", type=int, default=2); p.add_argument("--lr", type=float, default=1e-4)  # noqa: E702
    p = sub.add_parser("submit")
    p.add_argument("--harness", default="active"); p.add_argument("--model", default="active")  # noqa: E702
    p = sub.add_parser("feedback"); p.add_argument("--benchmark", required=True)  # noqa: E702
    p.add_argument("--harness", default="active"); p.add_argument("--model", default="active")  # noqa: E702
    a = ap.parse_args(argv)
    c = Client.from_env()
    try:
        if a.cmd == "status":
            print(json.dumps(c.call("status")))
        elif a.cmd == "lineage":
            lin = c.call("lineage")
            print(f"active: {lin['active']}")
            for v in lin["versions"]:
                print(f"  {v['id']:<10} parent={v['parent']!s:<10} round={v['round']:<3} by={v['improver']}")
            for k, v in lin["candidates"].items():
                print(f"  candidate {k}: examples={v['examples']} losses={v['losses']}")
        elif a.cmd == "docs":
            print(c.call("docs")["doc"])
        elif a.cmd == "infer":
            print(c.infer(a.prompt, a.model, a.seed))
        elif a.cmd == "local-eval":
            print(json.dumps(c.call("local_eval", data=a.data, harness=a.harness, model=a.model, trials=a.trials)))
        elif a.cmd == "train-sft":
            print(json.dumps(c.call("train_sft", data=a.data, epochs=a.epochs, lr=a.lr)))
        elif a.cmd == "submit":
            print(json.dumps(c.call("submit", harness=a.harness, model=a.model)))
        elif a.cmd == "feedback":
            print(json.dumps(c.call("feedback", benchmark=a.benchmark, harness=a.harness, model=a.model)))
    except RSIError as e:
        print(json.dumps({"error": str(e)}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
