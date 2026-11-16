"""Run budgets and evaluation settings (paper Table 2 / Appendix A.5)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Budget:
    """Default improvement budget: ten rounds of three hours (30 h total).

    The Improver may finish a round early; reported runtime is time actually used.
    """

    rounds: int = 10
    round_seconds: float = 3 * 3600
    max_steps_per_round: int = 400
    command_timeout: float = 900.0
    feedback_calls_per_round: int = 5  # Benchmark-Guided only


@dataclass
class EvalSettings:
    trials: int = 3                       # each item evaluated three times
    context_window: int = 262_144
    compaction_threshold: int = 235_929   # ~90% of the context window
    max_output_tokens: int = 81_920
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    simpleqa_subset: int = 500            # paper: fixed 500-question random subset
    simpleqa_judge: str = "normalized-match"  # paper: gpt-5.5 judge
    concurrency: int = 1                  # parallel trials per evaluation (match server slots)

    def sampling(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_output_tokens,
        }


@dataclass
class RunConfig:
    task: str = "knowledge"               # one of rsi_env.tasks.TASKS
    improver: str = "self"                # "self" or "teacher"
    target_model: dict = field(default_factory=lambda: {"backend": "toy", "name": "toy-base"})
    teacher_model: dict | None = None
    teacher_harness: str | None = None    # directory; defaults to the minimal harness
    benchmarks: list[str] = field(default_factory=lambda: [
        "mini-gpqa", "mini-aime", "mini-terminal", "mini-simpleqa"])
    budget: Budget = field(default_factory=Budget)
    eval: EvalSettings = field(default_factory=EvalSettings)
    seed: int = 0
    eval_every_version: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
