"""Deep-research PRO domain: a deliberately harder multi-hop QA benchmark.

The ordinary ``research`` domain (``research.py``) saturates: a single, well-written
system-prompt protocol takes the live agent from ~82% to 100% in one accepted
harness update, after which the Improver correctly declares ``TASK_COMPLETE``.
That is a *single* recursive step.

This domain is built so that **no single harness edit saturates it**. It has
several *independent* difficulty axes, each defeated by a different, separable
harness skill, and some that are in tension (so one prompt instruction cannot
serve both). A seed harness scores low; each accepted version typically unlocks
one axis and leaves measurable headroom, so under a Self Improver the loop keeps
finding justified updates round after round. Nothing here trains the model -
every gain is in the harness (M0 throughout).

Independent axes (question types)
=====  =========================================================  ===========================
type   what it needs                                              axis defeated by
=====  =========================================================  ===========================
P1     5-hop chain (device -> org -> hq -> country -> capital     disciplined decomposition /
       -> river of the capital)                                   a running fact ledger
P2     value from the MOST RECENT of several dated corrections    recency selection across
       (the encyclopedia and older corrections are stale)         multiple conflicting sources
P3     filtered count (devices after YEAR in FIELD)               writing a script over the
                                                                  corpus instead of fetching
P4     arithmetic combining two separate aggregations             multi-step quantitative plan
P5     EXHAUSTIVE list, graded by exact set equality              completeness / enumeration,
       (all devices developed at ORG)                            not first-answer-and-stop
P6     earliest-founded org in a country, using corrected years   superlative + correction
P7     value AS OF a SPECIFIC dated correction (not the latest)   reading a named source, which
                                                                  CONTRADICTS the P2 instinct
P8     disambiguate two researchers sharing a first name, then    entity disambiguation before
       answer a fact about the right one                         answering
=====  =========================================================  ===========================

Tools are the same as ``research`` (``search.py`` BM25 + ``fetch.py``) over a
larger, noisier ``corpus.json`` with distractor documents. Facts are fictional,
so nothing can be answered from memory. No real systems or people are involved.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from . import TOOL_BUDGET
from .research import FETCH_PY, SEARCH_PY  # identical BM25 search + fetch tools, over our corpus.json

SEED = 20261116
SYL = ["bar", "cel", "dor", "fen", "gal", "hor", "ist", "jun", "kel", "lom", "mar", "nev", "orb", "pel",
       "quin", "ros", "sel", "tav", "ur", "vin", "wes", "xan", "yor", "zel", "an", "bri", "cor", "del",
       "esk", "fro", "gud", "hesp", "ilv", "oko", "pryn", "stav", "thal", "umb"]
# A small pool of shared FIRST names forces disambiguation (P8): several people share one.
FIRST_NAMES = ["Adre", "Boril", "Caden", "Doralt", "Emberin"]
FIELDS = ["materials science", "acoustics", "hydrology", "robotics", "cryptography", "optics",
          "agronomy", "metallurgy", "seismology"]
CURRENCIES = ["crown", "mark", "dinar", "florin", "real", "peso", "lira", "rand", "taler", "shilling"]
DEV_KINDS = ["engine", "lens", "loom", "valve", "array", "pump", "rotor", "coil", "mast", "sieve"]


def _word(rng, parts, used):
    while True:
        n = "".join(rng.choice(SYL) for _ in range(parts)).capitalize()
        if n not in used:
            used.add(n)
            return n


@lru_cache(maxsize=2)
def build_world(seed: int = SEED) -> dict:
    rng = random.Random(seed)
    used: set[str] = set()
    countries = [{"name": _word(rng, 3, used) + "ia", "currency": c} for c in CURRENCIES]
    rivers = [{"name": _word(rng, 2, used), "length_km": rng.randint(120, 3900)} for _ in range(24)]
    cities = []
    for i in range(50):
        cities.append({"name": _word(rng, 2, used) + rng.choice(["", "ton", "burg", "port", "ford", "mouth"]),
                       "country": countries[i % len(countries)]["name"], "river": rng.choice(rivers)["name"],
                       "pop_old": rng.randint(40, 900) * 1000})
    for c in countries:
        c["capital"] = next(x["name"] for x in cities if x["country"] == c["name"])
    # People: some share a first name (P8). Give each a unique surname so entity identity is well-defined.
    people = []
    for _ in range(70):
        first = rng.choice(FIRST_NAMES) if rng.random() < 0.55 else _word(rng, 2, used)
        people.append({"name": f"{first} {_word(rng, 3, used)}", "first": first.split()[0],
                       "born": rng.choice(cities)["name"], "birth_year": rng.randint(1890, 1985),
                       "field": rng.choice(FIELDS)})
    orgs = []
    for i in range(50):
        kind = rng.choice(["Institute", "Laboratories", "Works", "Foundation", "Company", "Collective"])
        orgs.append({"name": f"{_word(rng, 2, used)} {kind}", "founder": people[i % len(people)]["name"],
                     "founded": rng.randint(1900, 2005), "hq": rng.choice(cities)["name"]})
    for p in people:
        p["employer"] = rng.choice(orgs)["name"]
    devices = []
    for i in range(60):
        inv = people[rng.randrange(len(people))]
        devices.append({"name": f"{_word(rng, 2, used)} {rng.choice(DEV_KINDS)}", "inventor": inv["name"],
                        "year": rng.randint(1920, 2015), "org": rng.choice(orgs)["name"]})
    # Multi-correction orgs (P2/P7): a chain of dated corrections; the LATEST year holds the true value,
    # earlier corrections and the encyclopedia are stale (different) values.
    corrected = rng.sample(range(len(orgs)), 22)
    for i in corrected:
        o = orgs[i]
        true = o["founded"]
        yrs = sorted(rng.sample([2013, 2015, 2017, 2019, 2021, 2023], rng.choice([2, 3])))
        vals = [true + rng.choice([-9, -6, -4, 3, 7, 11]) for _ in yrs[:-1]] + [true]
        o["enc_founded"] = true + rng.choice([-13, -10, 10, 13])  # stale encyclopedia value, distinct again
        o["corrections"] = list(zip(yrs, vals))                    # [(year, stated_value), ...], last == true
    return {"countries": countries, "rivers": rivers, "cities": cities, "people": people, "orgs": orgs,
            "devices": devices, "corrected": set(corrected)}


@lru_cache(maxsize=2)
def build_corpus(seed: int = SEED) -> list[dict]:
    w = build_world(seed)
    docs: list[dict] = []

    def add(kind, title, text):
        docs.append({"id": f"{kind}-{len(docs):04d}", "title": title, "text": text})

    for c in w["countries"]:
        add("country", c["name"], f"{c['name']} is a country. The capital of {c['name']} is {c['capital']}. "
                                  f"The official currency of {c['name']} is the {c['currency']}.")
    for r in w["rivers"]:
        add("river", f"{r['name']} River", f"The {r['name']} River has a length of {r['length_km']} km.")
    for c in w["cities"]:
        add("city", c["name"], f"{c['name']} is a city in {c['country']}. The {c['river']} River flows "
                               f"through {c['name']}. At the 2011 census {c['name']} had {c['pop_old']:,} people.")
    for p in w["people"]:
        add("person", p["name"], f"{p['name']} (born {p['birth_year']}) is a researcher in {p['field']}. "
                                 f"{p['name']} was born in {p['born']} and works at {p['employer']}.")
    for i, o in enumerate(w["orgs"]):
        if i in w["corrected"]:
            shown = o["enc_founded"]
        else:
            shown = o["founded"]
        add("org", o["name"], f"{o['name']} was established by {o['founder']}. According to this "
                              f"encyclopedia entry it was founded in {shown}. Its headquarters are in {o['hq']}.")
        if i in w["corrected"]:
            for yr, val in o["corrections"]:
                add("news", f"Correction ({yr}): {o['name']}", f"Gazette, {yr}. Correction: the founding year of "
                    f"{o['name']} is now confirmed to be {val}. This supersedes earlier statements.")
    for d in w["devices"]:
        add("device", d["name"], f"The {d['name']} was invented by {d['inventor']} in {d['year']}. "
                                 f"It was developed at {d['org']}.")
    # Distractor news documents (noise): plausible-looking but irrelevant to any question.
    rng = random.Random(seed + 9)
    for _ in range(30):
        c = rng.choice(w["cities"])
        add("news", f"Festival in {c['name']}", f"Herald. The annual festival in {c['name']} drew large crowds "
            f"along the {c['river']} River this year; no records were broken.")
    return docs


def _by(seq, key, val):
    return next(x for x in seq if x[key] == val)


def _true_founded(o: dict, corrected: set, idx: int) -> int:
    return o["corrections"][-1][1] if idx in corrected else o["founded"]


@lru_cache(maxsize=2)
def question_pool(seed: int = SEED) -> list[dict]:
    w = build_world(seed)
    rng = random.Random(seed + 1)
    cities, orgs, people, devices = w["cities"], w["orgs"], w["people"], w["devices"]
    corrected = w["corrected"]
    cap_of = {c["name"]: c["capital"] for c in w["countries"]}
    river_of = {c["name"]: c["river"] for c in cities}
    city_country = {c["name"]: c["country"] for c in cities}
    qs: list[dict] = []

    # P1: 5-hop chain device -> org -> hq city -> country -> capital -> river of the capital
    for d in devices:
        org = _by(orgs, "name", d["org"])
        country = city_country[org["hq"]]
        cap = cap_of[country]
        qs.append(("P1", f"Consider the device '{d['name']}'. It was developed at an organization; that "
                   f"organization is headquartered in a city, which lies in a country. Which river flows "
                   f"through the capital of that country?", river_of[cap], ""))

    # P2: value from the MOST RECENT correction (latest year wins)
    for i, o in enumerate(orgs):
        if i in corrected:
            qs.append(("P2", f"Taking the most recent correction into account, in what year was {o['name']} "
                       f"founded?", str(o["corrections"][-1][1]), ""))

    # P3: filtered count - devices invented strictly after YEAR by a researcher in FIELD
    for field in FIELDS:
        for year in (1950, 1970, 1990):
            n = sum(1 for d in devices
                    if d["year"] > year and _by(people, "name", d["inventor"])["field"] == field)
            if 1 <= n <= 6:
                qs.append(("P3", f"How many devices were invented strictly after {year} by a researcher whose "
                           f"field is {field}?", str(n), ""))

    # P4: arithmetic combining two aggregations
    for _ in range(40):
        country = rng.choice(w["countries"])["name"]
        yr = rng.choice([1940, 1960, 1980])
        a = sum(1 for i, o in enumerate(orgs)
                if city_country[o["hq"]] == country and _true_founded(o, corrected, i) < yr)
        org = rng.choice(orgs)
        b = sum(1 for d in devices if d["org"] == org["name"])
        if 0 <= a <= 6 and 0 <= b <= 6 and (a + b) >= 1:
            qs.append(("P4", f"Add together these two numbers: (1) the number of organizations headquartered in "
                       f"{country} that were founded before {yr} (use the corrected founding years); and (2) the "
                       f"number of devices developed at {org['name']}. What is the total?", str(a + b), ""))

    # P5: exhaustive list - all devices developed at ORG (graded by exact set equality)
    by_org: dict[str, list] = {}
    for d in devices:
        by_org.setdefault(d["org"], []).append(d["name"])
    for org, names in by_org.items():
        if 2 <= len(names) <= 5:
            qs.append(("P5", f"List the names of ALL devices developed at {org}. Give them as a comma-separated "
                       f"list.", "|".join(sorted(names)), "set"))

    # P6: earliest-founded org in a country, using CORRECTED founding years
    by_country: dict[str, list] = {}
    for i, o in enumerate(orgs):
        by_country.setdefault(city_country[o["hq"]], []).append((i, o))
    for country, lst in by_country.items():
        if len(lst) >= 3:
            yrs = [(_true_founded(o, corrected, i), o["name"]) for i, o in lst]
            if len({y for y, _ in yrs}) == len(yrs):  # unique minimum
                qs.append(("P6", f"Among all organizations headquartered in {country}, which one was founded "
                           f"earliest? Use the corrected founding years.", min(yrs)[1], ""))

    # P7: value AS OF a SPECIFIC dated correction (NOT the latest) - tension with P2
    for i, o in enumerate(orgs):
        if i in corrected and len(o["corrections"]) >= 2:
            yr, val = o["corrections"][0]  # the EARLIEST correction, deliberately superseded later
            qs.append(("P7", f"According to the Gazette correction published in {yr} (not any later correction), "
                       f"what founding year did it state for {o['name']}?", str(val), ""))

    # P8: disambiguation - two researchers share a first name; identify by field, then give birth city
    first_counts = Counter(p["first"] for p in people)
    for p in people:
        if first_counts[p["first"]] >= 2:
            # unique by (first name, field)?
            same = [q for q in people if q["first"] == p["first"] and q["field"] == p["field"]]
            if len(same) == 1:
                qs.append(("P8", f"Several researchers share the first name {p['first']}. In which city was the "
                           f"one whose field is {p['field']} born?", p["born"], ""))

    # de-duplicate by question text
    out, seen = [], set()
    for t, q, a, mode in qs:
        if q not in seen:
            seen.add(q)
            out.append({"type": t, "question": q, "answer": a, "mode": mode})
    rng.shuffle(out)
    return out


def split(seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Public dev questions (for the Improver) and the disjoint private pool."""
    pool = question_pool(seed)
    dev = [q for i, q in enumerate(pool) if i % 4 == 0]
    rest = [q for i, q in enumerate(pool) if i % 4 != 0]
    return dev, rest


TYPES = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8")


def _stratified(qs: list[dict], n: int, rng: random.Random) -> list[dict]:
    by: dict[str, list] = {}
    for q in qs:
        by.setdefault(q["type"], []).append(q)
    for v in by.values():
        rng.shuffle(v)
    out, types = [], sorted(by)
    i = 0
    while len(out) < n and any(by.values()):
        t = types[i % len(types)]
        if by[t]:
            out.append(by[t].pop())
        i += 1
    return out


def task_prompt(question: str) -> str:
    return ("[research-pro-task] Answer the research question using the local document collection (the facts are "
            "fictional; do not rely on memory). Tools in the current directory: python search.py \"query\" "
            f"(ranked results) and python fetch.py DOC_ID (full document). You may use at most {TOOL_BUDGET} "
            "tool calls. Some questions require counting, listing, comparison, arithmetic, or choosing a "
            "specific dated source. "
            f"Question: {question}\nEnd your reply with 'Final answer: <answer>'.")


def items(which: str, salt: str = "", n: int = 40) -> list[dict]:
    dev, rest = split()
    if which == "dev":
        chosen = _stratified(dev, n, random.Random(11))
        tag = "devpro"
    else:
        rng = random.Random(int(hashlib.sha256(f"research-pro|{salt}".encode()).hexdigest()[:12], 16))
        chosen = _stratified(rest, n, rng)
        tag = "rqpro"
    return [{"id": f"{tag}-{i:02d}-{q['type']}", "kind": "research_pro", "prompt": task_prompt(q["question"]),
             "question": q["question"], "answer": q["answer"], "qtype": q["type"], "mode": q.get("mode", "")}
            for i, q in enumerate(chosen)]


def install_tools(task_dir: Path) -> None:
    import json
    task_dir = Path(task_dir)
    (task_dir / "corpus.json").write_text(json.dumps(build_corpus()), encoding="utf-8")
    (task_dir / "search.py").write_text(SEARCH_PY, encoding="utf-8")
    (task_dir / "fetch.py").write_text(FETCH_PY, encoding="utf-8")


# ----------------------------------------------------------------------------- grading
def _norm(s: str) -> str:
    s = s.lower().replace(",", " ")
    s = re.sub(r"\b(the|river|inhabitants|residents|people|devices|device|year|named?|total|is|was)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _as_set(s: str) -> set:
    s = re.sub(r"\b(and|the|device|named?)\b", " ", (s or "").lower())
    parts = re.split(r"[,\n;|]+|\s{2,}", s)
    return {re.sub(r"[^a-z0-9]+", " ", p).strip() for p in parts if re.sub(r"[^a-z0-9]+", "", p)}


def grade(item: dict, response: str) -> bool:
    m = re.findall(r"final answer\s*[:\-]\s*(.+)", response or "", re.I | re.S)
    if not m:
        return False
    got_raw = m[-1].strip()
    if item.get("mode") == "set":
        want = {_norm(x) for x in item["answer"].split("|")}
        got = {_norm(x) for x in _as_set(got_raw)}
        got.discard("")
        return got == want
    got, want = _norm(got_raw), _norm(item["answer"])
    if want.isdigit():
        nums = re.findall(r"\d+", got)
        return bool(nums) and nums[0] == want
    return got == want or (want in got and len(got) <= len(want) + 12)
