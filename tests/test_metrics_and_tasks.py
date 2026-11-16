from rsi_env.benchmarks.suites import (Item, all_k, avg_at_k, extract_final, get_suite, grade, pass_at_k,
                                      public_corpus)
from rsi_env.tasks import (BENCHMARK_ACCESS_GUIDED, BENCHMARK_ACCESS_NONE, IMPROVER_MARKER,
                           OBJECTIVE_OPEN_GENERAL, TASKS)


def test_avg_and_pass_at_k():
    trials = [[True, False, False], [False, False, False], [True, True, True]]
    assert round(avg_at_k(trials), 2) == round(100 * 4 / 9, 2)
    assert round(pass_at_k(trials), 2) == round(100 * 2 / 3, 2)
    assert round(all_k(trials), 2) == round(100 / 3, 2)


def test_grading_variants():
    assert grade(Item("1", "q", "1234 km"), "Final answer: 1,234 kilometres")
    assert grade(Item("2", "q", "b", kind="mc"), "reasoning...\nFinal answer: (B)")
    assert grade(Item("3", "q", "42"), "Final answer: 42.")
    assert not grade(Item("4", "q", "Ostrin"), "Final answer: Velmor")
    assert extract_final("a\nFinal answer: x\nFinal answer: y") == "y"


def test_seven_paper_tasks_three_settings_plus_domains():
    paper = {k: t for k, t in TASKS.items() if t.capability not in ("redteam", "research", "research_pro")}
    assert len(paper) == 7
    assert {t.setting for t in paper.values()} == {"open-general", "capability-directed", "benchmark-guided"}
    assert sum(t.setting == "capability-directed" for t in paper.values()) == 3
    assert sum(t.setting == "benchmark-guided" for t in paper.values()) == 3
    for name in ("redteam", "research", "research-pro"):  # real-world domains: broad goal, no benchmark feedback
        assert TASKS[name].setting == "capability-directed" and not TASKS[name].feedback_benchmarks


def test_prompt_block_replacement_rules():
    og = TASKS["general"].prompt()
    assert IMPROVER_MARKER in og and OBJECTIVE_OPEN_GENERAL in og and BENCHMARK_ACCESS_NONE in og
    for name in ("reasoning", "knowledge", "agentic"):  # only the Objective block is replaced
        p = TASKS[name].prompt()
        assert OBJECTIVE_OPEN_GENERAL not in p and BENCHMARK_ACCESS_NONE in p
        assert not TASKS[name].feedback_benchmarks
    for name in ("bg-terminal", "bg-simpleqa", "bg-aime"):  # Objective and Benchmark access replaced
        p = TASKS[name].prompt()
        assert BENCHMARK_ACCESS_GUIDED in p and BENCHMARK_ACCESS_NONE not in p
        assert TASKS[name].feedback_benchmarks
    assert "TASK_COMPLETE" in og


def test_private_prompts_never_in_public_corpus():
    corpus = " ".join(d["text"] for d in public_corpus())
    for it in get_suite("mini-simpleqa").items:
        assert it.prompt not in corpus
        assert it.answer in corpus  # the knowledge itself is learnable from public resources
