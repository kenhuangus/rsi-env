"""Deep-research domain: multi-hop question answering over a document collection.

A seeded generator builds a self-contained "web" of ~270 documents (encyclopedia
pages for people, organizations, cities, countries, rivers and devices, plus news
articles and dated corrections). Questions need research rather than recall:

* T1  device → inventor → birth city                        (2 hops)
* T2  organization → founder → birth city → river            (3 hops)
* T3  device → developing org → HQ city → country → currency (4 hops)
* T4  which of two organizations was founded earlier         (comparison; may need a correction)
* T5  population "according to the most recent census"       (news supersedes encyclopedia)
* T6  how many devices a person invented                     (aggregation across pages)
* T7  founding year of an organization with a published correction

Agents use two tools inside the task directory: ``python search.py "query"`` (BM25,
ranked titles + snippets) and ``python fetch.py DOC_ID`` (full text). The facts are
fictional, so models cannot answer from memory.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from . import TOOL_BUDGET

SEED = 20261115
SYL = ["bar", "cel", "dor", "fen", "gal", "hor", "ist", "jun", "kel", "lom", "mar", "nev", "orb", "pel",
       "quin", "ros", "sel", "tav", "ur", "vin", "wes", "xan", "yor", "zel", "an", "bri", "cor", "del"]
FIELDS = ["materials science", "acoustics", "hydrology", "robotics", "cryptography", "optics", "agronomy"]
CURRENCIES = ["crown", "mark", "dinar", "florin", "real", "peso", "lira", "rand", "taler", "shilling"]


def _name(rng, parts, used):
    while True:
        n = "".join(rng.choice(SYL) for _ in range(parts)).capitalize()
        if n not in used:
            used.add(n)
            return n


@lru_cache(maxsize=2)
def build_world(seed: int = SEED) -> dict:
    rng = random.Random(seed)
    used: set[str] = set()
    countries = [{"name": _name(rng, 3, used) + "ia", "currency": c} for c in CURRENCIES]
    rivers = [{"name": _name(rng, 2, used), "length_km": rng.randint(120, 3900)} for _ in range(16)]
    cities = []
    for i in range(30):
        cities.append({"name": _name(rng, 2, used) + rng.choice(["", "ton", "burg", "port", "ford"]),
                       "country": countries[i % len(countries)]["name"], "river": rng.choice(rivers)["name"],
                       "pop_old": rng.randint(40, 900) * 1000})
        cities[-1]["pop_new"] = int(cities[-1]["pop_old"] * rng.uniform(1.05, 1.4)) // 100 * 100
    for c in countries:
        c["capital"] = next(x["name"] for x in cities if x["country"] == c["name"])
    people = []
    for _ in range(40):
        people.append({"name": f"{_name(rng, 2, used)} {_name(rng, 3, used)}", "born": rng.choice(cities)["name"],
                       "birth_year": rng.randint(1890, 1985), "field": rng.choice(FIELDS)})
    orgs = []
    for i in range(30):
        kind = rng.choice(["Institute", "Laboratories", "Works", "Foundation", "Company"])
        orgs.append({"name": f"{_name(rng, 2, used)} {kind}", "founder": people[i]["name"],
                     "founded": rng.randint(1900, 2005), "hq": rng.choice(cities)["name"]})
    for p in people:
        p["employer"] = rng.choice(orgs)["name"]
    devices = []
    inventors = people[:]
    rng.shuffle(inventors)
    multi = inventors[:10]
    for i in range(36):
        inv = multi[i % 10] if i < 24 else rng.choice(inventors[10:])
        devices.append({"name": f"{_name(rng, 2, used)} {rng.choice(['engine', 'lens', 'loom', 'valve', 'array', 'pump'])}",
                        "inventor": inv["name"], "year": rng.randint(1920, 2015), "org": rng.choice(orgs)["name"]})
    corrected = rng.sample(range(len(orgs)), 10)
    for i in corrected:
        orgs[i]["wrong_founded"] = orgs[i]["founded"] + rng.choice([-7, -4, -2, 3, 5, 9])
    return {"countries": countries, "rivers": rivers, "cities": cities, "people": people, "orgs": orgs,
            "devices": devices, "corrected": corrected}


@lru_cache(maxsize=2)
def build_corpus(seed: int = SEED) -> list[dict]:
    w = build_world(seed)
    docs: list[dict] = []

    def add(kind, title, text):
        docs.append({"id": f"{kind}-{len(docs):04d}", "title": title, "text": text})

    for c in w["countries"]:
        add("country", c["name"], f"{c['name']} is a country whose capital is {c['capital']}. "
                                  f"The official currency of {c['name']} is the {c['currency']}.")
    for r in w["rivers"]:
        add("river", f"{r['name']} River", f"The {r['name']} River is {r['length_km']} km long.")
    for c in w["cities"]:
        add("city", c["name"], f"{c['name']} is a city in {c['country']}. The {c['river']} River flows through "
                               f"{c['name']}. At the 2011 census {c['name']} had {c['pop_old']:,} inhabitants.")
    for p in w["people"]:
        add("person", p["name"], f"{p['name']} (born {p['birth_year']}) is a researcher in {p['field']}. "
                                 f"{p['name'].split()[0]} was born in {p['born']} and later worked at {p['employer']}.")
    corrected = set(w["corrected"])
    for i, o in enumerate(w["orgs"]):
        # corrected organizations keep a stale encyclopedia entry; only the 2021 correction is right
        founded = f" It was founded in {o['wrong_founded'] if i in corrected else o['founded']}."
        add("org", o["name"], f"{o['name']} was established by {o['founder']}.{founded} "
                              f"Its headquarters are located in {o['hq']}.")
        if i in corrected:
            add("news", f"Profile: {o['name']} (2019)", f"Daily Ledger, 2019. A profile of {o['name']}, which the "
                f"paper reported was founded in {o['wrong_founded']}, describes its growth since then.")
            add("news", f"Correction: {o['name']} (2021)", f"Daily Ledger, 2021. Correction: an earlier article "
                f"misstated the founding year of {o['name']}. It was founded in {o['founded']}, not {o['wrong_founded']}.")
    for d in w["devices"]:
        add("device", d["name"], f"The {d['name']} was invented by {d['inventor']} in {d['year']}. "
                                 f"It was developed at {d['org']}.")
    for c in w["cities"][::2]:
        add("news", f"Census results: {c['name']} (2024)", f"Statistics Office, 2024. The most recent census "
            f"counted {c['pop_new']:,} residents in {c['name']}, up from the 2011 figure.")
    return docs


def _by(seq, key, val):
    return next(x for x in seq if x[key] == val)


@lru_cache(maxsize=2)
def question_pool(seed: int = SEED) -> list[dict]:
    w = build_world(seed)
    rng = random.Random(seed + 1)
    qs = []
    for d in w["devices"]:
        inv = _by(w["people"], "name", d["inventor"])
        qs.append(("T1", f"In which city was the inventor of the {d['name']} born?", inv["born"]))
        hq = _by(w["cities"], "name", _by(w["orgs"], "name", d["org"])["hq"])
        cur = _by(w["countries"], "name", hq["country"])["currency"]
        qs.append(("T3", f"What currency is used in the country where the organization that developed the "
                         f"{d['name']} is headquartered?", cur))
    for i, o in enumerate(w["orgs"]):
        city = _by(w["cities"], "name", _by(w["people"], "name", o["founder"])["born"])
        qs.append(("T2", f"Which river flows through the city where the founder of {o['name']} was born?",
                   city["river"]))
        if i in w["corrected"]:
            qs.append(("T7", f"In what year was {o['name']} founded?", str(o["founded"])))
    orgs = w["orgs"]
    for _ in range(40):
        a, b = rng.sample(range(len(orgs)), 2)
        if orgs[a]["founded"] == orgs[b]["founded"]:
            continue
        if a not in w["corrected"] and b not in w["corrected"]:
            continue
        win = orgs[a] if orgs[a]["founded"] < orgs[b]["founded"] else orgs[b]
        qs.append(("T4", f"Which was founded earlier: {orgs[a]['name']} or {orgs[b]['name']}?", win["name"]))
    for c in w["cities"][::2]:
        qs.append(("T5", f"According to the most recent census, what is the population of {c['name']}?",
                   f"{c['pop_new']}"))
    counts = Counter(d["inventor"] for d in w["devices"])
    for name, n in counts.items():
        if n >= 2:
            qs.append(("T6", f"How many devices did {name} invent?", str(n)))
    city_country = {c["name"]: c["country"] for c in w["cities"]}
    # T8 set aggregation with corrected values: count orgs in a country founded before a year
    for c in w["countries"]:
        for year in (1940, 1960, 1980):
            n = sum(1 for o in orgs if city_country[o["hq"]] == c["name"] and o["founded"] < year)
            if 1 <= n <= 4:
                qs.append(("T8", f"How many organizations headquartered in {c['name']} were founded before {year}?",
                           str(n)))
    # T9 superlative over a reverse lookup: earliest device developed at an organization
    by_org: dict[str, list] = {}
    for d in w["devices"]:
        by_org.setdefault(d["org"], []).append(d)
    for org, ds in by_org.items():
        if len(ds) >= 2 and len({d["year"] for d in ds}) == len(ds):
            qs.append(("T9", f"Among the devices developed at {org}, which one was invented earliest?",
                       min(ds, key=lambda d: d["year"])["name"]))
    # T10 multi-constraint intersection: field x river of birth city
    river_of = {c["name"]: c["river"] for c in w["cities"]}
    for f in FIELDS:
        for r in w["rivers"]:
            hits = [p for p in w["people"] if p["field"] == f and river_of[p["born"]] == r["name"]]
            if len(hits) == 1:
                qs.append(("T10", f"Which researcher in {f} was born in a city on the {r['name']} River?",
                           hits[0]["name"]))
    # T11 arithmetic over two 3-hop chains
    length = {r["name"]: r["length_km"] for r in w["rivers"]}
    born = {p["name"]: p["born"] for p in w["people"]}
    devs = w["devices"]
    for _ in range(30):
        a, b = rng.sample(devs, 2)
        ra, rb = river_of[born[a["inventor"]]], river_of[born[b["inventor"]]]
        if ra != rb:
            qs.append(("T11", f"What is the combined length in km of the river flowing through the birth city of "
                              f"the inventor of the {a['name']} and the river flowing through the birth city of "
                              f"the inventor of the {b['name']}?", str(length[ra] + length[rb])))
    out = []
    seen = set()
    for t, q, a in qs:
        if q not in seen:
            seen.add(q)
            out.append({"type": t, "question": q, "answer": a})
    rng.shuffle(out)
    return out


def split(seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Public dev questions (for the Improver) and the remaining private pool."""
    pool = question_pool(seed)
    dev = [q for i, q in enumerate(pool) if i % 5 == 0]
    rest = [q for i, q in enumerate(pool) if i % 5 != 0]
    return dev, rest


def task_prompt(question: str) -> str:
    return ("[research-task] Answer the research question using the local document collection (the facts are "
            "fictional; do not rely on memory). Tools in the current directory: python search.py \"query\" "
            f"(ranked results) and python fetch.py DOC_ID (full document). You may use at most {TOOL_BUDGET} "
            "tool calls. "
            f"Question: {question}\nEnd your reply with 'Final answer: <answer>'.")


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


HARD_TYPES = ("T2", "T3", "T4", "T7", "T8", "T9", "T10", "T11")  # private set focuses on these


def items(which: str, salt: str = "", n: int = 28) -> list[dict]:
    dev, rest = split()
    if which == "dev":
        chosen = _stratified([q for q in dev if q["type"] in HARD_TYPES], n, random.Random(7))
        tag = "dev"
    else:
        rng = random.Random(int(hashlib.sha256(f"research|{salt}".encode()).hexdigest()[:12], 16))
        chosen = _stratified([q for q in rest if q["type"] in HARD_TYPES], n, rng)
        tag = "rq"
    return [{"id": f"{tag}-{i:02d}-{q['type']}", "kind": "research", "prompt": task_prompt(q["question"]),
             "question": q["question"], "answer": q["answer"], "qtype": q["type"]} for i, q in enumerate(chosen)]


# ----------------------------------------------------------------------------- tools
SEARCH_PY = r'''"""Search the document collection. Usage: python search.py "query" [-k 5]"""
import json, math, re, sys
from collections import Counter
args = sys.argv[1:]
k = 5
if "-k" in args:
    i = args.index("-k"); k = int(args[i + 1]); del args[i:i + 2]
if not args:
    print(__doc__); sys.exit(0)
docs = json.load(open("corpus.json", encoding="utf-8"))
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
toks = [tok(d["title"] + " " + d["title"] + " " + d["text"]) for d in docs]
N = len(docs); avg = sum(map(len, toks)) / N
df = Counter(t for ts in toks for t in set(ts))
q = tok(" ".join(args))
def bm25(ts):
    tf = Counter(ts); s = 0.0
    for t in q:
        if t in tf:
            idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(ts) / avg))
    return s
ranked = sorted(range(N), key=lambda i: -bm25(toks[i]))[:k]
for i in ranked:
    d = docs[i]
    print(f"[{d['id']}] {d['title']} :: {d['text'][:110]}...")
'''

FETCH_PY = r'''"""Fetch a full document. Usage: python fetch.py DOC_ID"""
import json, sys
if len(sys.argv) < 2:
    print(__doc__); sys.exit(0)
docs = {d["id"]: d for d in json.load(open("corpus.json", encoding="utf-8"))}
for doc_id in sys.argv[1:]:
    d = docs.get(doc_id.strip("[]"))
    print(f"# {d['title']} ({d['id']})\n{d['text']}\n" if d else f"no document {doc_id}")
'''


def install_tools(task_dir: Path) -> None:
    task_dir = Path(task_dir)
    (task_dir / "corpus.json").write_text(json.dumps(build_corpus()), encoding="utf-8")
    (task_dir / "search.py").write_text(SEARCH_PY, encoding="utf-8")
    (task_dir / "fetch.py").write_text(FETCH_PY, encoding="utf-8")


def _norm(s: str) -> str:
    s = s.lower().replace(",", "")
    s = re.sub(r"\b(the|river|inhabitants|residents|people|devices|km)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def grade(item: dict, response: str) -> bool:
    m = re.findall(r"final answer\s*[:\-]\s*(.+)", response or "", re.I)
    if not m:
        return False
    got, want = _norm(m[-1]), _norm(item["answer"])
    if want.isdigit():
        nums = re.findall(r"\d+", got)
        return bool(nums) and nums[0] == want
    return got == want or (want in got and len(got) <= len(want) + 12)
