# Asset-Specific Risk Register (NFCRM-1:2025 §6.9)

_Computed by `scripts/compute_asset_risk.py` from `results/multisource_arg.json`._

Each row is one (asset, attack-scenario) pair. **Inherent** risk uses the asset's worst exposed-CVE CVSS band (CISA-KEV forces band 5) for exploitability and scales the attack-class CIA prior by §6.3 criticality. **Residual** risk applies §6.12 reduction for §6.7 currently-applied controls.

Scenarios scored: **30** across **29** assets.

## Per-scenario risk

| Asset | Host | Crit | Attack | CVSS | KEV | Expl | L | I | Inherent | Residual |
|---|---|---|---|---|---|---|---|---|---|---|
| SRV-001 | ftp-srv-01 | critical | BruteForce | 10.0 | · | 4 | 5 | 5 | 25 (Catastrophic) | 20 (Catastrophic) |
| SRV-002 | ssh-srv-01 | critical | BruteForce | 7.8 | · | 3 | 5 | 5 | 25 (Catastrophic) | 20 (Catastrophic) |
| SRV-007 | php-srv-01 | critical | WebAttack | 7.5 | Y | 5 | 5 | 5 | 25 (Catastrophic) | 20 (Catastrophic) |
| SRV-008 | dvwa-srv-01 | critical | WebAttack | — | · | 5 | 5 | 5 | 25 (Catastrophic) | 20 (Catastrophic) |
| SRV-010 | phpmyadmin-srv-01 | high | Infiltration | 7.5 | Y | 5 | 5 | 5 | 25 (Catastrophic) | 25 (Catastrophic) |
| SRV-029 | heartbleed-srv-01 | critical | Infiltration | 10.0 | Y | 5 | 5 | 5 | 25 (Catastrophic) | 20 (Catastrophic) |
| SRV-030 | infiltration-victim-01 | high | Infiltration | 9.3 | Y | 5 | 5 | 5 | 25 (Catastrophic) | 20 (Catastrophic) |
| SRV-032 | win-srv-01 | critical | Infiltration | 9.3 | Y | 5 | 5 | 5 | 25 (Catastrophic) | 20 (Catastrophic) |
| SRV-033 | win-srv-02 | critical | Infiltration | 9.3 | Y | 5 | 5 | 5 | 25 (Catastrophic) | 20 (Catastrophic) |
| SRV-003 | telnet-srv-01 | high | Infiltration | 10.0 | · | 4 | 4 | 5 | 20 (Catastrophic) | 20 (Catastrophic) |
| SRV-011 | twiki-srv-01 | high | Infiltration | 10.0 | · | 4 | 4 | 5 | 20 (Catastrophic) | 20 (Catastrophic) |
| SRV-014 | smb-srv-01 | critical | Infiltration | 10.0 | · | 4 | 4 | 5 | 20 (Catastrophic) | 20 (Catastrophic) |
| SRV-015 | rservices-srv-01 | critical | Infiltration | 10.0 | · | 4 | 4 | 5 | 20 (Catastrophic) | 20 (Catastrophic) |
| SRV-018 | nfs-srv-01 | critical | Infiltration | 10.0 | · | 4 | 4 | 5 | 20 (Catastrophic) | 20 (Catastrophic) |
| SRV-019 | ftp2-srv-01 | high | Infiltration | 10.0 | · | 4 | 4 | 5 | 20 (Catastrophic) | 20 (Catastrophic) |
| SRV-021 | distcc-srv-01 | critical | Infiltration | 10.0 | · | 4 | 4 | 5 | 20 (Catastrophic) | 20 (Catastrophic) |
| SRV-025 | irc-srv-01 | critical | Infiltration | 10.0 | · | 4 | 4 | 5 | 20 (Catastrophic) | 20 (Catastrophic) |
| SRV-027 | drb-srv-01 | critical | Infiltration | 10.0 | · | 4 | 4 | 5 | 20 (Catastrophic) | 20 (Catastrophic) |
| SRV-028 | web-dos-target-01 | critical | DoS | 7.5 | · | 3 | 4 | 5 | 20 (Catastrophic) | 15 (High) |
| SRV-028 | web-dos-target-01 | critical | DDoS | 7.5 | · | 3 | 4 | 5 | 20 (Catastrophic) | 15 (High) |
| SRV-005 | dns-srv-01 | high | Infiltration | 7.1 | · | 3 | 3 | 5 | 15 (High) | 15 (High) |
| SRV-006 | http-srv-01 | critical | Infiltration | 7.8 | · | 3 | 3 | 5 | 15 (High) | 15 (High) |
| SRV-012 | tikiwiki-srv-01 | high | Infiltration | 7.5 | · | 3 | 3 | 5 | 15 (High) | 15 (High) |
| SRV-016 | rmi-srv-01 | high | Infiltration | 7.5 | · | 3 | 3 | 5 | 15 (High) | 15 (High) |
| SRV-020 | mysql-srv-01 | critical | Infiltration | 7.5 | · | 3 | 3 | 5 | 15 (High) | 15 (High) |
| SRV-022 | pgsql-srv-01 | high | Infiltration | 6.5 | · | 2 | 3 | 5 | 15 (High) | 15 (High) |
| SRV-023 | vnc-srv-01 | critical | Infiltration | 7.5 | · | 3 | 3 | 5 | 15 (High) | 15 (High) |
| SRV-026 | tomcat-srv-01 | high | Infiltration | 5.0 | · | 2 | 3 | 5 | 15 (High) | 15 (High) |
| SRV-024 | x11-srv-01 | medium | Infiltration | 7.5 | · | 3 | 3 | 4 | 12 (Medium) | 12 (Medium) |
| SRV-031 | infiltration-victim-02 | medium | Infiltration | — | · | 3 | 3 | 4 | 12 (Medium) | 8 (Medium) |

## Discrimination: inherent-score spread within each attack class

The previous attack-type-constant model assigns one identical score to every scenario in a class (stdev = 0 by construction). Asset-specific scoring produces non-zero spread wherever the asset population is heterogeneous.

| Attack class | n | min | max | mean | stdev | distinct scores |
|---|---|---|---|---|---|---|
| BruteForce | 2 | 25 | 25 | 25 | 0.0 | [25] |
| DDoS | 1 | 20 | 20 | 20 | 0.0 | [20] |
| DoS | 1 | 20 | 20 | 20 | 0.0 | [20] |
| Infiltration | 24 | 12 | 25 | 18.71 | 4.15 | [12, 15, 20, 25] |
| WebAttack | 2 | 25 | 25 | 25 | 0.0 | [25] |

## Inherent → residual effect of currently-applied controls (§6.7/§6.12)

- Scenarios where residual < inherent: **11/30**
- Mean inherent score: **19.63**; mean residual score: **17.83** (mean reduction **1.8**).
