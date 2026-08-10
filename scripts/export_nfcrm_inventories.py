"""Export §6.3, §6.4, §6.5 inventories from the ARG.

Reads:
    results/multisource_arg.json

Writes:
    results/nfcrm_inventories.json
    results/nfcrm_inventories.md  - human-readable
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.nfcrm.inventories import export_inventories  # noqa: E402

RESULTS = REPO / "results"
ARG = RESULTS / "multisource_arg.json"


def render_md(payload: dict) -> str:
    lines = [
        "# NFCRM-1:2025 Inventories (§6.3, §6.4, §6.5)",
        "",
        f"_Source: `{payload['source_arg']}`_  ",
        f"_Clause references: {payload['clause_reference']}_",
        "",
        f"## §6.3 Asset Inventory ({len(payload['assets_6_3'])} entries)",
        "",
        "| Asset ID | Asset type | NFCRM classification | Software linked |",
        "|---|---|---|---|",
    ]
    for a in payload["assets_6_3"]:
        sw = ", ".join(a["has_software"]) or "—"
        lines.append(f"| `{a['asset_id']}` | {a['asset_type']} | {a['classification']} | {sw} |")
    lines.append("")
    lines.append(f"## §6.4 Vulnerability Inventory ({len(payload['vulnerabilities_6_4'])} entries)")
    lines.append("")
    lines.append("| CVE ID | Severity | KEV? | Affects assets |")
    lines.append("|---|---|---|---|")
    for v in payload["vulnerabilities_6_4"]:
        kev = "yes" if v["in_kev"] else "no"
        affects = ", ".join(v["affects_assets"][:3]) + (
            f" (+{len(v['affects_assets']) - 3})" if len(v["affects_assets"]) > 3 else ""
        )
        lines.append(f"| {v['cve_id']} | {v['severity'] or '—'} | {kev} | {affects or '—'} |")
    lines.append("")
    lines.append(f"## §6.5 Threat Inventory ({len(payload['threats_6_5'])} entries)")
    lines.append("")
    lines.append("| Threat ID | Framework | Description | Targets |")
    lines.append("|---|---|---|---|")
    for t in payload["threats_6_5"]:
        targets = ", ".join(t["targets_assets"][:3]) + (
            f" (+{len(t['targets_assets']) - 3})" if len(t["targets_assets"]) > 3 else ""
        )
        lines.append(f"| {t['threat_id']} | {t['framework']} | {t['description']} | {targets or '—'} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    payload = export_inventories(ARG, RESULTS)
    md_path = RESULTS / "nfcrm_inventories.md"
    md_path.write_text(render_md(payload), encoding="utf-8")
    n_a = len(payload["assets_6_3"])
    n_v = len(payload["vulnerabilities_6_4"])
    n_t = len(payload["threats_6_5"])
    print(f"NFCRM inventories built: {n_a} assets (§6.3), {n_v} vulns (§6.4), {n_t} threats (§6.5)")
    print(f"  -> results/nfcrm_inventories.json")
    print(f"  -> results/nfcrm_inventories.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
