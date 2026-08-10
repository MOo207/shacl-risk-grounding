# Inherent Risk Register (NFCRM-1:2025 §6.9 demo)

_Computed by `scripts/compute_inherent_risk.py` using `pipeline/nfcrm`._

## §6.9 Per-Attack-Type Defaults

These are the default likelihood and impact values used when no scenario-level override is provided. They reflect the public threat-intelligence consensus for each CIC-IDS2018 attack class. Per-scenario overrides (asset criticality, EPSS score, organisational incidence data) take priority via the `*_override` parameters of `compute_risk_score`.

| Attack | L (time/exp) | Impact (C/I/A) | Risk = L×I | Level (en) | Level (ar) |
|---|---|---|---|---|---|
| Benign | 0 (0/0) | 0 (0/0/0) | N/A | N/A | غير قابل للتطبيق |
| BruteForce | 5 (5/5) | 4 (4/3/2) | 20 | Catastrophic | كارثي |
| DDoS | 5 (4/5) | 5 (1/1/5) | 25 | Catastrophic | كارثي |
| DoS | 5 (4/5) | 4 (1/1/4) | 20 | Catastrophic | كارثي |
| Infiltration | 3 (3/3) | 5 (5/5/3) | 15 | High | مرتفع |
| WebAttack | 5 (4/5) | 4 (4/4/2) | 20 | Catastrophic | كارثي |

## Demo enrichment (`results/intersymbolic_explanations.json`)

| Flow | True | Predicted | L | I | Score | Level |
|---|---|---|---|---|---|---|
| 1 | DDoS | DDoS | 5 | 5 | 25 | Catastrophic |
| 2 | BruteForce | BruteForce | 5 | 4 | 20 | Catastrophic |
| 3 | DoS | DoS | 5 | 4 | 20 | Catastrophic |
| 4 | WebAttack | WebAttack | 5 | 4 | 20 | Catastrophic |
| 5 | Infiltration | Infiltration | 3 | 5 | 15 | High |

## SLM run risk-level distribution (`results/slm_nl_classification.json`, N=60)

| Risk level | Total predictions | By attack type |
|---|---|---|
| N/A (non-attack) | 16 | Benign=16 |
| High | 12 | Infiltration=12 |
| Catastrophic | 29 | BruteForce=4, DDoS=8, DoS=8, WebAttack=9 |
