"""Narrative quality scorer for LLM risk-artifact results.

Metrics per case:
- clause_count: number of distinct NFCRM §X.Y clause references
- clauses_cited: list of unique clause strings found
- cve_mentioned: True if the paired CVE ID appears in the narrative
- asset_mentioned: True if the asset ID appears in the narrative
- narrative_words: word count
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

CLAUSE_RE = re.compile(r"§\d+(?:\.\d+)+")


def score_narrative(case: dict) -> dict:
    narrative = case.get("narrative", "") or ""
    clauses = list({m.group() for m in CLAUSE_RE.finditer(narrative)})

    paired = case.get("paired_cve") or {}
    cve_id: Optional[str] = paired.get("cve_id")
    cve_mentioned = bool(cve_id and cve_id in narrative)

    asset_id: Optional[str] = (case.get("asset") or {}).get("id")
    asset_mentioned = bool(asset_id and asset_id in narrative)

    return {
        "case_id": case["case_id"],
        "clause_count": len(clauses),
        "clauses_cited": sorted(clauses),
        "cve_mentioned": cve_mentioned,
        "asset_mentioned": asset_mentioned,
        "narrative_words": len(narrative.split()),
    }


def evaluate_arm(jsonl_path: Path, test_set_path: Path) -> dict:
    """Score all cases in a JSONL result file against the test set."""
    test_set = json.loads(test_set_path.read_text(encoding="utf-8"))
    case_lookup = {c["case_id"]: c for c in test_set["cases"]}

    scores = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            case_id = result.get("case_id", "")
            merged = {**case_lookup.get(case_id, {}), "narrative": result.get("narrative", "")}
            scores.append(score_narrative(merged))

    if not scores:
        return {
            "n": 0,
            "mean_clause_count": 0.0,
            "pct_cve_mentioned": 0.0,
            "pct_asset_mentioned": 0.0,
            "mean_narrative_words": 0.0,
        }

    n = len(scores)
    return {
        "n": n,
        "mean_clause_count": sum(s["clause_count"] for s in scores) / n,
        "pct_cve_mentioned": 100 * sum(s["cve_mentioned"] for s in scores) / n,
        "pct_asset_mentioned": 100 * sum(s["asset_mentioned"] for s in scores) / n,
        "mean_narrative_words": sum(s["narrative_words"] for s in scores) / n,
        "per_case": scores,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--test-set", default="results/llm_experiment/test_set.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = evaluate_arm(Path(args.jsonl), Path(args.test_set))
    out = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
    else:
        print(out)


if __name__ == "__main__":
    main()
