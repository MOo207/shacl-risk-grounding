"""Compute NFCRM-1:2025 §6.9 inherent risk for the framework's outputs.

Reads:
    results/intersymbolic_explanations.json   - the 5-flow demo
    results/slm_nl_classification.json         - the larger SLM run
    (optional) overrides per attack type from a JSON file

Writes:
    results/inherent_risk_register.json        - per-flow risk register
    results/inherent_risk_register.md          - human-readable summary
    results/nfcrm_attack_type_defaults.md      - reference table

Stdlib only. No API calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.nfcrm import (  # noqa: E402
    ATTACK_IMPACT_DEFAULTS,
    ATTACK_LIKELIHOOD_DEFAULTS,
    compute_risk_score,
    risk_level_for_score,
)

RESULTS = REPO / "results"
DEMO_PATH = RESULTS / "intersymbolic_explanations.json"
SLM_PATH = RESULTS / "slm_nl_classification.json"
OUT_JSON = RESULTS / "inherent_risk_register.json"
OUT_MD = RESULTS / "inherent_risk_register.md"
DEFAULTS_MD = RESULTS / "nfcrm_attack_type_defaults.md"


def attack_type_summary() -> list[dict]:
    """Per-attack-type defaults (rows of the §6.9 reference table)."""
    rows = []
    for atk in sorted(ATTACK_LIKELIHOOD_DEFAULTS.keys()):
        r = compute_risk_score(atk)
        rows.append({
            "attack_type": atk,
            "likelihood": r.likelihood.value,
            "L_time_period": r.likelihood.time_period,
            "L_exploitability": r.likelihood.exploitability,
            "impact": r.impact.value,
            "C": r.impact.confidentiality,
            "I": r.impact.integrity,
            "A": r.impact.availability,
            "risk_score": r.score,
            "risk_level_en": r.level_en,
            "risk_level_ar": r.level_ar,
        })
    return rows


def enrich_demo_flows() -> list[dict]:
    """Attach §6.9 inherent-risk fields to each flow in the 5-flow demo."""
    if not DEMO_PATH.exists():
        return []
    flows = json.loads(DEMO_PATH.read_text(encoding="utf-8"))
    enriched = []
    for flow in flows:
        atk = flow.get("predicted_label") or flow.get("true_label")
        if atk not in ATTACK_LIKELIHOOD_DEFAULTS:
            continue
        r = compute_risk_score(atk)
        enriched.append({
            "flow_id": flow.get("flow_id"),
            "true_label": flow.get("true_label"),
            "predicted_label": flow.get("predicted_label"),
            "nfcrm_6_9": {
                "clause": "NFCRM-1:2025 §6.9",
                "likelihood": {
                    "value": r.likelihood.value,
                    "time_period": r.likelihood.time_period,
                    "exploitability": r.likelihood.exploitability,
                    "rationale": r.likelihood.rationale,
                },
                "impact": {
                    "value": r.impact.value,
                    "confidentiality": r.impact.confidentiality,
                    "integrity": r.impact.integrity,
                    "availability": r.impact.availability,
                    "rationale": r.impact.rationale,
                },
                "risk_score": r.score,
                "risk_level_en": r.level_en,
                "risk_level_ar": r.level_ar,
            },
        })
    return enriched


def aggregate_slm_run() -> dict | None:
    """Aggregate the SLM run's risk-level distribution per predicted class."""
    if not SLM_PATH.exists():
        return None
    data = json.loads(SLM_PATH.read_text(encoding="utf-8"))
    results = data.get("results", [])
    counts: dict[str, dict[str, int]] = {}
    for r in results:
        pred = r.get("predicted") or "<empty>"
        if pred not in ATTACK_LIKELIHOOD_DEFAULTS:
            continue
        rs = compute_risk_score(pred)
        counts.setdefault(rs.level_en, {"total": 0, "by_attack": {}})
        counts[rs.level_en]["total"] += 1
        counts[rs.level_en]["by_attack"][pred] = (
            counts[rs.level_en]["by_attack"].get(pred, 0) + 1
        )
    return {
        "source": str(SLM_PATH.relative_to(REPO).as_posix()),
        "n_predictions": len(results),
        "by_risk_level": counts,
    }


def render_md(report: dict) -> str:
    lines = [
        "# Inherent Risk Register (NFCRM-1:2025 §6.9 demo)",
        "",
        "_Computed by `scripts/compute_inherent_risk.py` using `pipeline/nfcrm`._",
        "",
        "## §6.9 Per-Attack-Type Defaults",
        "",
        "These are the default likelihood and impact values used when no scenario-level "
        "override is provided. They reflect the public threat-intelligence consensus for "
        "each CIC-IDS2018 attack class. Per-scenario overrides (asset criticality, EPSS "
        "score, organisational incidence data) take priority via the `*_override` "
        "parameters of `compute_risk_score`.",
        "",
        "| Attack | L (time/exp) | Impact (C/I/A) | Risk = L×I | Level (en) | Level (ar) |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["attack_type_defaults"]:
        if row["risk_score"] == 0:
            score_cell = "N/A"
            level_en = "N/A"
            level_ar = "غير قابل للتطبيق"
        else:
            score_cell = str(row["risk_score"])
            level_en = row["risk_level_en"]
            level_ar = row["risk_level_ar"]
        lines.append(
            f"| {row['attack_type']} | "
            f"{row['likelihood']} ({row['L_time_period']}/{row['L_exploitability']}) | "
            f"{row['impact']} ({row['C']}/{row['I']}/{row['A']}) | "
            f"{score_cell} | {level_en} | {level_ar} |"
        )
    lines.append("")
    lines.append("## Demo enrichment (`results/intersymbolic_explanations.json`)")
    lines.append("")
    if report["demo_enriched"]:
        lines.append("| Flow | True | Predicted | L | I | Score | Level |")
        lines.append("|---|---|---|---|---|---|---|")
        for f in report["demo_enriched"]:
            d = f["nfcrm_6_9"]
            lines.append(
                f"| {f['flow_id']} | {f['true_label']} | {f['predicted_label']} | "
                f"{d['likelihood']['value']} | {d['impact']['value']} | "
                f"{d['risk_score']} | {d['risk_level_en']} |"
            )
    else:
        lines.append("_demo file not present_")
    lines.append("")
    if report.get("slm_aggregate"):
        agg = report["slm_aggregate"]
        lines.append(f"## SLM run risk-level distribution (`{agg['source']}`, "
                     f"N={agg['n_predictions']})")
        lines.append("")
        lines.append("| Risk level | Total predictions | By attack type |")
        lines.append("|---|---|---|")
        for level, data in agg["by_risk_level"].items():
            atk_str = ", ".join(f"{k}={v}" for k, v in sorted(data["by_attack"].items()))
            lines.append(f"| {level} | {data['total']} | {atk_str} |")
        lines.append("")
    return "\n".join(lines)


def render_defaults_md() -> str:
    """A standalone reference table for inclusion in the thesis."""
    lines = [
        "# NFCRM-1:2025 §6.9 Default Likelihood/Impact Values per CIC-IDS2018 Attack Type",
        "",
        "_Source: `pipeline/nfcrm/likelihood.py`, `pipeline/nfcrm/impact.py`. "
        "See `audit/nfcrm_compliance_matrix.md` §6.9 for the full clause text._",
        "",
        "| Attack type | Likelihood (max of time/exp) | Impact (max of C/I/A) | "
        "Inherent risk = L×I | Risk level |",
        "|---|---|---|---|---|",
    ]
    for atk in sorted(ATTACK_LIKELIHOOD_DEFAULTS.keys()):
        r = compute_risk_score(atk)
        if r.score == 0:
            lines.append(
                f"| {atk} | N/A (non-attack) | N/A (non-attack) | N/A | N/A |"
            )
        else:
            lines.append(
                f"| {atk} | {r.likelihood.value} (tp={r.likelihood.time_period}, "
                f"ex={r.likelihood.exploitability}) | "
                f"{r.impact.value} (C={r.impact.confidentiality}, "
                f"I={r.impact.integrity}, A={r.impact.availability}) | "
                f"{r.score} | {r.level_en} ({r.level_ar}) |"
            )
    lines.append("")
    lines.append(
        "Per-scenario overrides (asset criticality, EPSS, organisational incidence) "
        "take priority over these defaults via the `*_override` parameters of "
        "`compute_risk_score`."
    )
    return "\n".join(lines)


def main() -> int:
    report = {
        "source_module": "pipeline.nfcrm",
        "clause": "NFCRM-1:2025 §6.9",
        "attack_type_defaults": attack_type_summary(),
        "demo_enriched": enrich_demo_flows(),
        "slm_aggregate": aggregate_slm_run(),
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")
    DEFAULTS_MD.write_text(render_defaults_md(), encoding="utf-8")

    print("section 6.9 attack-type defaults (Likelihood x Impact = Score -> Level):")
    for row in report["attack_type_defaults"]:
        if row["risk_score"] == 0:
            print(f"  {row['attack_type']:13s}  N/A")
        else:
            print(f"  {row['attack_type']:13s}  "
                  f"L={row['likelihood']}  I={row['impact']}  "
                  f"score={row['risk_score']:2d}  {row['risk_level_en']}")
    print()
    if report["demo_enriched"]:
        print(f"Demo flows enriched: {len(report['demo_enriched'])}")
    if report["slm_aggregate"]:
        agg = report["slm_aggregate"]
        print(f"SLM run risk distribution (N={agg['n_predictions']}):")
        for level, data in agg["by_risk_level"].items():
            print(f"  {level:15s}  {data['total']:3d} predictions")
    print(f"\n  -> {OUT_JSON.relative_to(REPO)}")
    print(f"  -> {OUT_MD.relative_to(REPO)}")
    print(f"  -> {DEFAULTS_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
