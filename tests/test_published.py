"""Tests for published-benchmark loaders and SQuAD grading (offline fixtures)."""
import json

from rsi_env.benchmarks import published as pub


def test_squad_f1_and_grade():
    assert pub.squad_f1("the Eiffel Tower", "Eiffel Tower") == 1.0   # articles ignored
    assert pub.squad_f1("Paris", "London") == 0.0
    it = {"answer": "Marie Curie", "aliases": ["Maria Sklodowska-Curie"]}
    assert pub.grade_squad(it, "reasoning...\nFinal answer: Marie Curie.")
    assert pub.grade_squad(it, "Final answer: Maria Sklodowska-Curie")  # alias
    assert not pub.grade_squad(it, "Final answer: Albert Einstein")
    assert not pub.grade_squad(it, "I could not determine the answer")  # no 'Final answer:'


def test_load_hotpotqa(tmp_path):
    rows = [{"_id": "x1", "question": "Who directed it?", "answer": "Jane Doe",
             "context": [["Film A", ["It was directed by Jane Doe.", "It premiered in 1999."]],
                         ["Distractor", ["Unrelated sentence."]]]}]
    f = tmp_path / "hp.json"
    f.write_text(json.dumps(rows), encoding="utf-8")
    items = pub.load_hotpotqa(f)
    assert len(items) == 1
    it = items[0]
    assert it["kind"] == "research_ext" and it["benchmark"] == "hotpotqa" and it["answer"] == "Jane Doe"
    corpus = pub.context_corpus(it)
    assert len(corpus) == 2 and "directed by Jane Doe" in corpus[0]["text"]
    assert "Final answer" in it["prompt"] and "search.py" in it["prompt"]


def test_load_2wiki(tmp_path):
    rows = [{"_id": "w1", "question": "Which country?", "answer": "Freedonia",
             "context": [["Capital", ["The capital is Ville.", "Ville is in Freedonia."]]]}]
    f = tmp_path / "2w.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")  # jsonl form
    items = pub.load_2wiki(f)
    assert items[0]["answer"] == "Freedonia" and items[0]["benchmark"] == "2wiki"
    assert pub.context_corpus(items[0])[0]["title"] == "Capital"


def test_load_musique(tmp_path):
    rows = [{"id": "m1", "question": "Combined?", "answer": "42", "answer_aliases": ["forty-two"],
             "paragraphs": [{"title": "P1", "paragraph_text": "a=10.", "is_supporting": True},
                            {"title": "P2", "paragraph_text": "b=32.", "is_supporting": True}]}]
    f = tmp_path / "mu.jsonl"
    f.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    items = pub.load_musique(f)
    assert items[0]["answer"] == "42" and "forty-two" in items[0]["aliases"]
    assert len(pub.context_corpus(items[0])) == 2


def test_load_benchmark_samples_deterministically(tmp_path):
    rows = [{"_id": str(i), "question": f"q{i}", "answer": f"a{i}",
             "context": [["T", [f"text {i}"]]]} for i in range(20)]
    f = tmp_path / "hp.json"
    f.write_text(json.dumps(rows), encoding="utf-8")
    a = pub.load_benchmark("hotpotqa", f, n=5, seed=1)
    b = pub.load_benchmark("hotpotqa", f, n=5, seed=1)
    assert len(a) == 5 and [x["ext_id"] for x in a] == [x["ext_id"] for x in b]
    assert [x["id"] for x in a] == [f"hotpotqa-{j:04d}" for j in range(5)]


def test_worker_grades_research_ext_end_to_end(tmp_path):
    # the seed harness over a context corpus; toy model won't answer, but the pipeline must run + grade
    from pathlib import Path

    from rsi_env.controller.evaluation import run_worker
    rows = pub.load_benchmark("hotpotqa", _write_hp(tmp_path), n=None)
    harness = Path("rsi_env/harness_template").resolve()
    spec = {"backend": "toy", "name": "toy-base", "seed": 0}
    settings = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_output_tokens": 512,
                "eval_tool_budget": 5, "eval_trial_seconds": 30, "eval_concurrency": 1}
    res = run_worker("eval", harness, spec, tmp_path / "e", items=rows, trials=1, settings=settings, timeout=120)
    assert "per_item" in res and set(res["per_item"]) == {r["id"] for r in rows}


def _write_hp(tmp_path):
    rows = [{"_id": "x1", "question": "Who?", "answer": "Jane",
             "context": [["A", ["Jane did it."]]]}]
    f = tmp_path / "hp2.json"
    f.write_text(json.dumps(rows), encoding="utf-8")
    return f
