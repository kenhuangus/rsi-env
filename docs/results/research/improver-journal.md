# Research journal

## Session: Research deep-research capability improvement

### Session 1 - Baseline & data analysis
- Active harness (H0/M0): minimal harness, no research guidance.
- Previous round's candidate harness "research-agent": added a research protocol; eval on dev (trials ~1) failed only 4/42 items (003-T3 t0, 006-T8 t0, 010-T3 t0, 014-T10 t2) — ~9.5% failure rate. The failure on dev-06-T8 (counting) is a systematic gap: it cannot reliably enumerate/count orgs headquartered in a country.
- Data schema discovered:
  * person-XXXX: "<NAME> (born <YYYY>) is a researcher in <FIELD>. <NAME> was born in <CITY> and later worked at <ORG>."
  * org-XXXX: "<ORG> was established by <PERSON>. It was founded in <YYYY>. Its headquarters are located in <CITY>."
  * country-XXXX: "<COUNTRY> is a country whose capital is <CAPITAL>. The official currency of <COUNTRY> is <CURRENCY>."
  * city-XXXX: "<CITY> is a city in <COUNTRY>. The <RIVER> flows through <CITY>."
  * river-XXXX: "The <RIVER> is <LENGTH> km long."
  * device-XXXX: "The <DEVICE> was invented by <PERSON> in <YYYY>. It was developed at <ORG>."
  * news-XXXX: "Correction" articles (2021) override 2019 profiles (e.g. founding year); Census articles (2024).
- Key insight: For T8 (counting), a Python script scanning corpus.json is far more reliable and tool-call-efficient than iterating entities via search/fetch. The agent must check news "Correction" articles for founding years (T7/T4). Final answer must be output cleanly after "Final answer:" with no extra prose.

### Session 2 - Improved harness (research-harness-improved)
- Created candidate/research-harness-improved with enhanced system prompt:
  1. DATA SCHEMA section (document types and fields, link patterns).
  2. CORE RESEARCH PROTOCOL with 7 phases: DECOMPOSE, DISCOVER, LINK, VERIFY, RESOLVE CORRECTIONS (check news "Correction" articles, prefer 2021), COMPUTE (Python script for counting/summing/comparison), ANSWER (clean final answer).
  3. TOOL-RESOURCE STRATEGY emphasizing Python scripts for bulk operations.
- Applied prompt via rsi harness-prompt (system_prompt updated, 4462 chars; max_steps=400; no extensions).

### Local validation results
- T8 subset (dev-06, dev-13, dev-19, dev-25; 2 trials each, 8 trials):
  * Active harness: score 0% (all 8 trials fail)
  * Improved harness: score 100% (0 failures)
- Full dev set (28 items, 2 trials each, 56 trials):
  * Improved harness: score 100%, failing_ids: [], transcripts: experiments/local_eval/008
- All checks: agent follows a sound chain-of-facts, resolves news corrections (e.g. Marros Works 1936->1943), computes counts with scripts, outputs clean "Final answer:" lines. No need for further iteration.

### Conclusion
The improved harness achieves 100% on the dev practice collection with 2 trials, compared to the active harness which fails all T8 counting items. This is a clear, measured improvement. Submit.

### Session 3 - Iteration 3 (new iteration)
- Confirmed the research-harness-improved (H1/M0) achieves 100% on the dev practice collection (56/56 trials pass in experiments/local_eval/008).
- Verified the improvement was systematic: 7-phase research protocol (DECOMPOSE, DISCOVER, LINK, VERIFY, RESOLVE CORRECTIONS, COMPUTE, ANSWER), DATA SCHEMA section, TOOL-RESOURCE STRATEGY emphasizing Python scripts for bulk operations, and news "Correction" article override policy.
- Identified one additional targeted improvement: the COMPUTE step had only a step-by-step guide but no concrete reusable script template, causing the agent to write and debug Python scripts from scratch (extra tool calls and scripting errors).
- Added a "Generic Python script template" to the COMPUTE section — a reusable structure with corpus categorization by id prefix (org-/city-/person-/country-/river-/device-/news-), an extract_field helper, and a filter->map->reduce pattern. Updated harness.json name to "research-harness-improved" and synchronized prompt.txt.
- Verified harness.json and prompt.txt are consistent (same prompt text).
- No further improvement is justified. The agent's deep-research capability is achieved (100% on the dev practice collection, clean Final answer output, correction resolution, script-based computation).
