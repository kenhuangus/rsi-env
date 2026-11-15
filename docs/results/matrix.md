| task | setting | improver | mini-gpqa | mini-aime | mini-terminal | mini-simpleqa | final (H/M) | tool calls | seconds |
|---|---|---|---|---|---|---|---|---|---|
| base | – | seed | 61.7 | 63.3 | 75.0 | 54.4 | H0/M0 | n/a | n/a |
| general | open-general | self | 73.3 (↑11.6) | 64.4 (↑1.1) | 91.7 (↑16.7) | 58.3 (↑3.9) | H2/M0 | 17 | 19.9 |
| general | open-general | teacher | 79.2 (↑17.5) | 68.9 (↑5.6) | 75.0 (=0.0) | 56.7 (↑2.3) | H1/M0 | 10 | 7.0 |
| reasoning | capability-directed | self | 79.2 (↑17.5) * | 68.9 (↑5.6) * | 75.0 (=0.0) | 56.7 (↑2.3) | H1/M0 | 10 | 7.7 |
| reasoning | capability-directed | teacher | 79.2 (↑17.5) * | 68.9 (↑5.6) * | 75.0 (=0.0) | 56.7 (↑2.3) | H1/M0 | 10 | 7.4 |
| knowledge | capability-directed | self | 72.5 (↑10.8) | 68.9 (↑5.6) | 75.0 (=0.0) | 75.0 (↑20.6) * | H1/M1 | 17 | 15.4 |
| knowledge | capability-directed | teacher | 79.2 (↑17.5) | 68.9 (↑5.6) | 75.0 (=0.0) | 56.7 (↑2.3) * | H1/M0 | 10 | 7.2 |
| agentic | capability-directed | self | 61.7 (=0.0) | 63.3 (=0.0) | 91.7 (↑16.7) * | 54.4 (=0.0) | H1/M0 | 10 | 20.1 |
| agentic | capability-directed | teacher | 61.7 (=0.0) | 63.3 (=0.0) | 91.7 (↑16.7) * | 54.4 (=0.0) | H1/M0 | 10 | 20.9 |
| bg-terminal | benchmark-guided | self | 61.7 (=0.0) | 63.3 (=0.0) | 91.7 (↑16.7) * | 54.4 (=0.0) | H1/M0 | 10 | 10.5 |
| bg-terminal | benchmark-guided | teacher | 61.7 (=0.0) | 63.3 (=0.0) | 91.7 (↑16.7) * | 54.4 (=0.0) | H1/M0 | 10 | 10.5 |
| bg-simpleqa | benchmark-guided | self | 50.8 (↓10.9) | 63.3 (=0.0) | 75.0 (=0.0) | 65.0 (↑10.6) * | H0/M1 | 17 | 10.6 |
| bg-simpleqa | benchmark-guided | teacher | 61.7 (=0.0) | 63.3 (=0.0) | 75.0 (=0.0) | 54.4 (=0.0) * | H0/M0 | 10 | 6.1 |
| bg-aime | benchmark-guided | self | 61.7 (=0.0) | 63.3 (=0.0) * | 75.0 (=0.0) | 54.4 (=0.0) | H0/M0 | 10 | 5.8 |
| bg-aime | benchmark-guided | teacher | 61.7 (=0.0) | 63.3 (=0.0) * | 75.0 (=0.0) | 54.4 (=0.0) | H0/M0 | 10 | 5.5 |

\* target-aligned benchmark
