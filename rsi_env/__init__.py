"""RSI-Env: an open-ended environment for recursive self-improvement of agents.

Implementation of the system described in arXiv:2611.01736.
"""
from .config import Budget, EvalSettings, RunConfig  # noqa: F401
from .tasks import TASKS, get_task  # noqa: F401

__version__ = "0.1.0"
