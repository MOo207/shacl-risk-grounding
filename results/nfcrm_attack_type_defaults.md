# NFCRM-1:2025 §6.9 Default Likelihood/Impact Values per CIC-IDS2018 Attack Type

_Source: `pipeline/nfcrm/likelihood.py`, `pipeline/nfcrm/impact.py`. See `audit/nfcrm_compliance_matrix.md` §6.9 for the full clause text._

| Attack type | Likelihood (max of time/exp) | Impact (max of C/I/A) | Inherent risk = L×I | Risk level |
|---|---|---|---|---|
| Benign | N/A (non-attack) | N/A (non-attack) | N/A | N/A |
| BruteForce | 5 (tp=5, ex=5) | 4 (C=4, I=3, A=2) | 20 | Catastrophic (كارثي) |
| DDoS | 5 (tp=4, ex=5) | 5 (C=1, I=1, A=5) | 25 | Catastrophic (كارثي) |
| DoS | 5 (tp=4, ex=5) | 4 (C=1, I=1, A=4) | 20 | Catastrophic (كارثي) |
| Infiltration | 3 (tp=3, ex=3) | 5 (C=5, I=5, A=3) | 15 | High (مرتفع) |
| WebAttack | 5 (tp=4, ex=5) | 4 (C=4, I=4, A=2) | 20 | Catastrophic (كارثي) |

Per-scenario overrides (asset criticality, EPSS, organisational incidence) take priority over these defaults via the `*_override` parameters of `compute_risk_score`.