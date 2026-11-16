# Using top published benchmarks with the RSI-Env agents

The agents can be measured on, and improved against, the standard benchmarks the field
reports — not only the synthetic `research` / `research-pro` / `redteam` domains. The
loaders in `rsi_env/benchmarks/published.py` convert each dataset's **official released
format** into the agent's item schema, and the agent uses its own `search.py` / `fetch.py`
tools over the question's supporting documents, so evaluation is offline and fair.

Datasets are read from a **local file you download**. Nothing is fetched or uploaded by
RSI-Env; no personal data leaves your machine.

## Deep research (multi-hop QA with supporting documents)

| Benchmark | Reference | License | Loader name |
|---|---|---|---|
| HotpotQA (distractor) | Yang et al., EMNLP 2018 | CC BY-SA 4.0 | `hotpotqa` |
| 2WikiMultiHopQA | Ho et al., COLING 2020 | Apache-2.0 | `2wiki` |
| MuSiQue-Answerable | Trivedi et al., TACL 2022 | CC BY 4.0 | `musique` |

Download the dev/validation split from the official source (examples):

```powershell
# HotpotQA distractor dev (official mirror)
Invoke-WebRequest "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json" `
  -OutFile data\hotpot_dev_distractor_v1.json

# 2WikiMultiHopQA and MuSiQue: get dev.json / musique_ans_v1.0_dev.jsonl from their
# official GitHub releases (Apache-2.0 / CC BY 4.0) and place the files under data\.
```

### Score the seed agent vs. a self-improved agent

```powershell
$env:CLOVER_BASE_URL = "http://<your-server>:8000/v1"

# seed harness (H0) on 100 HotpotQA questions
python -m rsi_env.apps research-bench --benchmark hotpotqa `
  --data data\hotpot_dev_distractor_v1.json --n 100 --trials 1 `
  --model examples\clover-macmini.json

# the harness a run produced (same command, add --from-run)
python -m rsi_env.apps research-bench --benchmark hotpotqa `
  --data data\hotpot_dev_distractor_v1.json --n 100 --trials 1 `
  --from-run runs\research-pro-self --model examples\clover-macmini.json
```

Answers are graded with the standard SQuAD exact-match / token-F1 metric
(`grade_squad`), the same metric these benchmarks report.

## Red teaming (prompt injection / secret extraction)

These published benchmarks match RSI-Env's authorized, secret-extraction framing
(recovering a guarded secret, or evaluating indirect-injection defenses) rather than
harmful-content elicitation:

| Benchmark | Reference | What it measures |
|---|---|---|
| Gandalf (Lakera) | Lakera, 2023 | single-turn secret-word extraction across defense levels |
| TensorTrust | Toyer et al., ICLR 2024 | prompt extraction / hijacking around a secret access code |
| AgentDojo | Debenedetti et al., NeurIPS 2024 | indirect prompt injection in agentic tool use (utility + security) |
| InjecAgent | Zhan et al., 2024 | indirect prompt injection across tool-using agents |

These are wired as **local, LLM-backed guarded targets** (the defender model runs on your
own server, the agent tries to recover the secret through `send.py`), reusing the target
pattern from `rsi_env/domains/redteam.py`. Only run them against the local simulated
targets, or against endpoints you are explicitly authorized to test
(`python -m rsi_env.apps redteam ... --authorized`). The adapter lives alongside the
research loaders; see `rsi_env/benchmarks/published.py`.
