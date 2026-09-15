# R1 pilot: legacy Chengdu (1,901 nodes)

Fixed exploratory protocol: 20,000 training OD groups, 5,000 validation, 5,000 test; both directions.
30 epochs, hidden=128, seeds 42/99/1234. Validation selects checkpoints; test is evaluated after selection.
These results do not reproduce historical Harbin/Beijing runs and are not a full benchmark.

| Mode | Mean test MRE (%) | Sample std (percentage points) |
|---|---:|---:|
| shared_l1 | 13.384782 | 0.990156 |
| shared_tilde_63_1 | 13.305578 | 0.968955 |
| shared_tilde_62_2 | 13.601149 | 1.074158 |
| shared_linf | 6.185984 | 0.142200 |
| cross_scalar | 10.704023 | 0.156989 |
| cross_l1 | 13.779506 | 0.581717 |
| cross_tilde_62_2 | 12.384106 | 0.414573 |
| cross_tilde_2_62 | 7.222617 | 0.361123 |

Signed heads are not clipped. Negative-prediction fractions and per-seed/bucket scores are recorded in pilot.json.
Shared and cross families have different architectures; use within-family comparisons to isolate decoder effects.
Checkpoints and per-query predictions remain on the server under results/audit-r1-20260913/checkpoints/.
