"""Compute NFCRM-1:2025 §6.9 ASSET-SPECIFIC inherent + residual risk from the ARG.

Unlike scripts/compute_inherent_risk.py (which scores every flow of a given
attack type with the same attack-class constant), this driver scores each
(asset, attack-scenario) pair using the asset's own exposed-CVE profile, §6.3
criticality, and §6.7 currently-applied controls. The result is a per-asset
risk register where risk varies across assets within the same attack class.

Offline: reads results/multisource_arg.json only. No network.

Writes:
    results/asset_risk_register.json   - per-scenario inherent/residual rows + summary
    results/asset_risk_register.md     - human-readable register + discrimination table
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.nfcrm import load_arg, score_arg  # noqa: E402

ARG_PATH = REPO / "results" / "multisource_arg.json"
OUT_JSON = REPO / "results" / "asset_risk_register.json"
OUT_MD = REPO / "results" / "asset_risk_register.md"


def discrimination_summary(rows: list[dict]) -> dict:
    """Per-attack-class spread of inherent scores (the anti-flat-lookup evidence)."""
    byc: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        byc[r["attack_type"]].append(r["inherent_risk_score"])
    summary = {}
    for atk, scores in byc.items():
        summary[atk] = {
            "n": len(scores),
            "min": min(scores),
            "max": max(scores),
            "mean": round(st.mean(scores), 2),
            "stdev": round(st.pstdev(scores), 2) if len(scores) > 1 else 0.0,
            "distinct_scores": sorted(set(scores)),
        }
    return summary


def render_md(report: dict) -> str:
    rows = report["scenarios"]
    L = [
        "# Asset-Specific Risk Register (NFCRM-1:2025 §6.9)",
        "",
        "_Computed by `scripts/compute_asset_risk.py` from `results/multisource_arg.json`._",
        "",
        "Each row is one (asset, attack-scenario) pair. **Inherent** risk uses the "
        "asset's worst exposed-CVE CVSS band (CISA-KEV forces band 5) for "
        "exploitability and scales the attack-class CIA prior by §6.3 criticality. "
        "**Residual** risk applies §6.12 reduction for §6.7 currently-applied controls.",
        "",
        f"Scenarios scored: **{len(rows)}** across **{report['n_assets_scored']}** assets.",
        "",
        "## Per-scenario risk",
        "",
        "| Asset | Host | Crit | Attack | CVSS | KEV | Expl | L | I | Inherent | Residual |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: -x["inherent_risk_score"]):
        L.append(
            f"| {r['asset_id']} | {r['hostname']} | {r['criticality']} | "
            f"{r['attack_type']} | {r['max_cvss'] if r['max_cvss'] is not None else '—'} | "
            f"{'Y' if r['in_cisa_kev'] else '·'} | {r['exploitability']} | "
            f"{r['likelihood']} | {r['impact']} | "
            f"{r['inherent_risk_score']} ({r['inherent_risk_level']}) | "
            f"{r['residual_risk_score']} ({r['residual_risk_level']}) |"
        )
    L += [
        "",
        "## Discrimination: inherent-score spread within each attack class",
        "",
        "The previous attack-type-constant model assigns one identical score to every "
        "scenario in a class (stdev = 0 by construction). Asset-specific scoring "
        "produces non-zero spread wherever the asset population is heterogeneous.",
        "",
        "| Attack class | n | min | max | mean | stdev | distinct scores |",
        "|---|---|---|---|---|---|---|",
    ]
    for atk, s in sorted(report["discrimination"].items()):
        L.append(
            f"| {atk} | {s['n']} | {s['min']} | {s['max']} | {s['mean']} | "
            f"{s['stdev']} | {s['distinct_scores']} |"
        )
    L += [
        "",
        "## Inherent → residual effect of currently-applied controls (§6.7/§6.12)",
        "",
        f"- Scenarios where residual < inherent: **{report['n_residual_reduced']}/{len(rows)}**",
        f"- Mean inherent score: **{report['mean_inherent']}**; "
        f"mean residual score: **{report['mean_residual']}** "
        f"(mean reduction **{report['mean_reduction']}**).",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    arg = load_arg(ARG_PATH)
    scenario_objs = score_arg(arg)
    rows = [r.to_dict() for r in scenario_objs]

    n_assets = len({r["asset_id"] for r in rows})
    n_reduced = sum(1 for r in rows if r["residual_risk_score"] < r["inherent_risk_score"])
    mean_inh = round(st.mean(r["inherent_risk_score"] for r in rows), 2) if rows else 0
    mean_res = round(st.mean(r["residual_risk_score"] for r in rows), 2) if rows else 0

    report = {
        "source": "results/multisource_arg.json",
        "clause": "NFCRM-1:2025 §6.9 (inherent) + §6.12 (residual)",
        "method": (
            "exploitability = max exposed-CVE CVSS band (CISA-KEV -> 5); "
            "impact = attack-class CIA prior scaled by §6.3 criticality; "
            "residual = §6.12 likelihood reduction per §6.7 applied control. "
            "EPSS deliberately excluded (held out as external validation GT)."
        ),
        "n_scenarios": len(rows),
        "n_assets_scored": n_assets,
        "n_residual_reduced": n_reduced,
        "mean_inherent": mean_inh,
        "mean_residual": mean_res,
        "mean_reduction": round(mean_inh - mean_res, 2),
        "discrimination": discrimination_summary(rows),
        "scenarios": rows,
    }

    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")

    print(f"Scored {len(rows)} (asset, attack) scenarios across {n_assets} assets.")
    print(f"Residual < inherent on {n_reduced} scenarios "
          f"(mean {mean_inh} -> {mean_res}).")
    print("Discrimination (inherent-score spread within attack class):")
    for atk, s in sorted(report["discrimination"].items()):
        flat = "FLAT" if s["stdev"] == 0 else f"stdev={s['stdev']}"
        print(f"  {atk:13s} n={s['n']:2d}  range {s['min']}-{s['max']}  {flat}")
    print(f"\n  -> {OUT_JSON.relative_to(REPO)}")
    print(f"  -> {OUT_MD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
