"""Tests for the harder research-pro benchmark (multi-axis, no-single-fix)."""
from collections import Counter

import pytest

from rsi_env.benchmarks.suites import DOMAIN_SUITES, get_suite
from rsi_env.domains import research_pro as rp


def test_pool_has_all_eight_axes():
    by = Counter(q["type"] for q in rp.question_pool())
    assert set(by) == set(rp.TYPES)
    assert all(v >= 5 for v in by.values())


def test_dev_and_private_are_disjoint():
    dev, rest = rp.split()
    dq = {q["question"] for q in dev}
    rq = {q["question"] for q in rest}
    assert dq and rq and not (dq & rq)


def test_private_set_is_salted_and_stratified():
    a = rp.items("eval", salt="alpha", n=40)
    b = rp.items("eval", salt="beta", n=40)
    # different secret salts -> different sampled questions (private set is secret)
    assert [x["question"] for x in a] != [x["question"] for x in b]
    # every axis is represented so no single harness fix can saturate the score
    assert set(Counter(x["qtype"] for x in a)) == set(rp.TYPES)


def test_gold_answers_grade_true():
    it = rp.items("eval", salt="s", n=40)
    for x in it:
        gold = x["answer"].replace("|", ", ")
        assert rp.grade(x, f"work...\nFinal answer: {gold}"), x["qtype"]


def test_list_question_needs_exact_set():
    p5 = next(x for x in rp.items("eval", salt="s", n=40) if x["mode"] == "set")
    names = p5["answer"].split("|")
    assert rp.grade(p5, "Final answer: " + ", ".join(reversed(names)))      # order-insensitive
    assert not rp.grade(p5, "Final answer: " + names[0])                    # missing one -> wrong
    assert not rp.grade(p5, "Final answer: " + ", ".join(names + ["Extra gadget"]))  # extra -> wrong


def test_latest_vs_specific_correction_never_collide():
    w = rp.build_world()
    for i in w["corrected"]:
        o = w["orgs"][i]
        assert o["corrections"][-1][1] == o["founded"]            # P2 answer = latest correction
        if len(o["corrections"]) >= 2:                            # P7 asks the earliest correction
            assert o["corrections"][0][1] != o["corrections"][-1][1]


def test_corpus_contains_resolution_documents():
    docs = rp.build_corpus()
    kinds = Counter(d["id"].split("-")[0] for d in docs)
    assert kinds["org"] >= 40 and kinds["device"] >= 50 and kinds["news"] >= 20


def test_registered_as_private_domain_suite():
    assert "research-pro" in DOMAIN_SUITES
    suite = get_suite("research-pro", salt="secret")
    assert suite.kind == "research_pro" and len(suite.items) == 40
