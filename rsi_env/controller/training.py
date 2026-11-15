"""Training service behind the controlled ``train-sft`` interface (paper Appendix A.4).

``train-sft`` accepts JSONL prompt-completion pairs plus training settings and
returns the losses and a registered *model candidate*. The paper relays these
calls to the Tinker SDK; here the backend is chosen from the base model spec:

* ``toy``    - :class:`ToySFTTrainer`, a real (if tiny) learner: completions are
               written into the checkpoint and used at inference.
* ``tinker`` - :class:`TinkerTrainer`, an adapter for the Tinker SDK
               (``pip install tinker``); experimental, see README.

Every call is screened for contamination: training prompts that match a private
benchmark prompt are recorded as an incident and the candidate is marked so the
validity check rejects it.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from ..benchmarks.suites import extract_final, normalize
from ..benchmarks.world import parse_fact_question


def read_pairs(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not {"prompt", "completion"} <= set(r):
                raise ValueError("each JSONL row needs 'prompt' and 'completion'")
            rows.append(r)
    return rows


def norm_prompt(p: str) -> str:
    return re.sub(r"\s+", " ", p.lower()).strip()


class ToySFTTrainer:
    def train(self, rows: list[dict], base_spec: dict, epochs: int, lr: float, out_path: Path) -> tuple[dict, list[float]]:
        learned: dict[str, str] = {}
        if base_spec.get("checkpoint"):
            learned.update(json.loads(Path(base_spec["checkpoint"]).read_text(encoding="utf-8"))["learned"])
        new = 0
        for r in rows:
            key = parse_fact_question(r["prompt"])
            if key:
                val = extract_final(r["completion"])
                new += key not in learned
                learned[key] = val
        frac = new / max(1, len(rows))
        l0 = 2.0 + 4.0 * frac          # loss starts higher when more of the data is novel
        losses = [round(l0 * math.exp(-0.15 * e * max(1.0, lr * 1e4)), 4) for e in range(int(epochs) + 1)]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"base": base_spec.get("name"), "learned": learned}, indent=1), encoding="utf-8")
        spec = {**{k: v for k, v in base_spec.items() if k != "id"}, "checkpoint": str(out_path),
                "name": f"{base_spec.get('name', 'toy')}+sft{len(learned)}"}
        return spec, losses


class TinkerTrainer:  # pragma: no cover - requires the Tinker service and credentials
    """Experimental relay to the Tinker SDK (LoRA SFT). Verify against current Tinker docs."""

    def train(self, rows, base_spec, epochs, lr, out_path):
        import tinker
        from tinker import types

        sc = tinker.ServiceClient()
        tc = sc.create_lora_training_client(base_model=base_spec["tinker_base_model"])
        tok = tc.get_tokenizer()
        data = []
        for r in rows:
            p, c = tok.encode(r["prompt"]), tok.encode(r["completion"])
            toks, w = p + c, [0] * len(p) + [1] * len(c)
            data.append(types.Datum(model_input=types.ModelInput.from_ints(toks[:-1]),
                                    loss_fn_inputs={"target_tokens": toks[1:], "weights": w[1:]}))
        losses = []
        for _ in range(int(epochs)):
            fb = tc.forward_backward(data, "cross_entropy").result()
            tc.optim_step(types.AdamParams(learning_rate=lr)).result()
            losses.append(float(getattr(fb, "metrics", {}).get("loss:sum", float("nan"))))
        path = tc.save_weights_for_sampler(name=out_path.stem).result().path
        spec = {**base_spec, "tinker_path": path, "name": f"{base_spec.get('name')}+{out_path.stem}"}
        return spec, losses


class TrainingService:
    def __init__(self, out_dir: Path, private_prompts: set[str]):
        self.out_dir = Path(out_dir)
        self.private = private_prompts
        self.count = 0

    def train_sft(self, data_path: Path, base_spec: dict, epochs: int = 2, lr: float = 1e-4) -> dict:
        rows = read_pairs(data_path)
        contaminated = sum(norm_prompt(r["prompt"]) in self.private for r in rows)
        backend = base_spec.get("backend")
        trainer = ToySFTTrainer() if backend == "toy" else TinkerTrainer() if base_spec.get(
            "tinker_base_model") else None
        if trainer is None:
            raise RuntimeError(f"no training backend for model backend {backend!r} "
                               "(set 'tinker_base_model' in the model spec to use Tinker)")
        self.count += 1
        cand_id = f"cand-{self.count:04d}"
        spec, losses = trainer.train(rows, base_spec, epochs, lr, self.out_dir / f"{cand_id}.json")
        return {"candidate_model": cand_id, "spec": spec, "losses": losses, "examples": len(rows),
                "epochs": epochs, "lr": lr, "contaminated_rows": contaminated}
