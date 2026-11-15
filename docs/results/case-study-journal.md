# Research journal

## Round 1 - harness:self_consistency (active H0/M0)
- Hypothesis: harness:self_consistency improves knowledge without regressions.
- Evidence: baseline=51.7 candidate=62.5 (data/local_val.jsonl)
- Changed: candidate/harness/self_consistency.py (+harness.json)
- Decision: submit candidate
[op-done] harness:self_consistency
## Round 2 - sft:knowledge (active H1/M0)
- Hypothesis: sft:knowledge improves knowledge without regressions.
- Evidence: baseline=62.5 candidate=75.0 (data/local_val.jsonl)
- Changed: model weights (SFT, see data/sft_provenance.json)
- Decision: submit candidate
[op-done] sft:knowledge
## Round 3 - conclusion (active H1/M1)
- All planned operations tried; no further justified improvement identified.
