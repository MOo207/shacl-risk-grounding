# NFCRM-1:2025 Full-Cycle Demo

Identification: §6.3, §6.4, §6.5; Assessment: §6.9, §6.10; Treatment: §6.8, §6.11, §6.12, §6.13, §6.14, §6.15, §6.16; Monitoring: §6.17, §6.18, §6.20

## Identification (§6.3, §6.4, §6.5)
- §6.3 Assets: 50
- §6.4 Vulnerabilities: 32
- §6.5 Threats: 6

## §6.8 Acceptable Risk Level
- Threshold: **Low**
- Rationale: Default thesis-demo threshold. A real organisation would set this per its risk appetite and the §6.5 sensitivity of the asset.

## Assessment + Treatment (§6.9, §6.10, §6.11–§6.16)

| Priority | Entry ID | Attack | Inherent | Treatment | Residual | Status |
|---|---|---|---|---|---|---|
| 0 | RR-CLASS-DDoS | DDoS | 25 (Catastrophic) | Mitigate: L-5 I-1 | 5 (Low) | planned |
| 0 | RR-CLASS-DoS | DoS | 20 (Catastrophic) | Mitigate: L-5 I-1 | 5 (Low) | planned |
| 0 | RR-CLASS-BruteForce | BruteForce | 20 (Catastrophic) | Mitigate: L-5 I-1 | 5 (Low) | planned |
| 0 | RR-CLASS-WebAttack | WebAttack | 20 (Catastrophic) | Mitigate: L-5 I-1 | 5 (Low) | planned |
| 0 | RR-CLASS-Infiltration | Infiltration | 15 (High) | Mitigate: L-1 I-5 | 5 (Low) | planned |

## Monitoring Report (§6.17, §6.18)
- Total entries: 5
- Above acceptable level: 0

**Inherent risk distribution**
  - Catastrophic: 4
  - High: 1

**Residual risk distribution**
  - Low: 5

**Treatment-decision mix**
  - Mitigate: 5

## §6.18 Statistical-Data Criteria
- **STAT-01 Risk-level distribution** — register entries grouped by inherent_risk_level_en (count per level)
- **STAT-02 Treatment-status distribution** — register entries grouped by treatment_status (count per status)
- **STAT-03 Residual-risk-level distribution** — register entries grouped by residual_risk_level_en (post-treatment) (count per level)
- **STAT-04 Treatment-decision mix** — register entries grouped by treatment_decision (count per decision)
- **STAT-05 Above-acceptable count** — register entries where residual_risk exceeds acceptable_level (count)

## §6.20 Triggers
- any_triggered: True
- treatment_plan_changes: 5
  - RR-CLASS-BruteForce: plan/decision/status changed
  - RR-CLASS-DDoS: plan/decision/status changed
  - RR-CLASS-DoS: plan/decision/status changed
  - RR-CLASS-Infiltration: plan/decision/status changed
  - RR-CLASS-WebAttack: plan/decision/status changed
- scenario_changes: 5
  - removed entry: RR-DEMO-001-DDoS
  - removed entry: RR-DEMO-002-BruteForce
  - removed entry: RR-DEMO-003-DoS
  - removed entry: RR-DEMO-004-WebAttack
  - removed entry: RR-DEMO-005-Infiltration
- control_changes: 5
  - RR-CLASS-BruteForce: controls changed (2 -> 2)
  - RR-CLASS-DDoS: controls changed (2 -> 2)
  - RR-CLASS-DoS: controls changed (2 -> 2)
  - RR-CLASS-Infiltration: controls changed (2 -> 2)
  - RR-CLASS-WebAttack: controls changed (2 -> 2)