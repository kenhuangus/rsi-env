# Research journal

## Iteration 3: Deep research harness improvement

### Current state (from prior iterations)
- Baseline harness (active, minimal): 24/28 (85.7%) — fails dev-05-T7 (stale 2019 year, ignores 2021 correction), dev-06-T8 (poor T8 enumeration), dev-10-T3 (verbose "Final answer" fails length check), dev-25-T8 (poor T8 enumeration).
- Candidate "research-harness" (v1): 23/28 — fails dev-04-T4, dev-06-T8, dev-13-T8, dev-19-T8, dev-25-T8 (same k=5 enumeration problem).
- Candidate v2 (exp 007): 26/28 — fails dev-06-T8, dev-25-T8 (still misses orgs in city search).
- Candidate v3 (exp 008): 27/28 — only dev-25-T8 fails. The T8 protocol works but search uses default k=5, so the top-5 results miss Orbquin Company (ranked 6th for "Kelgalburg").

### Root cause of remaining failure (dev-25-T8)
- T8 answer = 2 (Orbquin Company 1923→1914 correction, Marros Works 1943 correction, both in Xancorvinia, both before 1960).
- The v3 agent correctly enumerated the 3 cities (Kelgalburg, Quinbriburg, Zelurton) and found Marros Works in Zelurton, but when searching "Kelgalburg" it got only top-5 results and missed Orbquin Company (ranked 6th).
- Fix: use -k 10 in the T8 city search so all organizations headquartered there are visible.

### Hypothesis
Adding `-k 10` to every search in the research protocol (and explicitly to the T8 city-level search) will make T8 enumeration complete and raise the score to 28/28. A general protocol reminder to use -k 10 for all searches is a broad-enough research-method improvement.

### Plan
1. Create candidate/research-harness-v4/ with the v3 protocol, adding `-k 10` to the T8 city search and a general `-k 10` instruction.
2. Evaluate on dev set (rsi local-eval --data resources/research/dev.jsonl --harness candidate/research-harness-v4 --trials 1).
3. If 28/28, submit v4 harness as the replacement; update journal. If not, investigate new failures.
