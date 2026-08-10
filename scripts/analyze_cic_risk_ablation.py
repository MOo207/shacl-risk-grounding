"""Compute metrics from CIC risk ablation JSONL files -> summary.json + console table.

Usage:
  python -m scripts.analyze_cic_risk_ablation
"""
from __future__ import annotations
import json, statistics
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "results/cic_risk_ablation"
OUT    = IN_DIR / "summary.json"

ARM_ORDER = ["ml", "shacl", "llm", "tristage_ml_shacl", "tristage_llm_shacl"]
ARM_LABELS = {
    "ml":                 "ML",
    "shacl":              "SHACL",
    "llm":                "LLM",
    "tristage_ml_shacl":  "Tri-ML+SHACL",
    "tristage_llm_shacl": "Tri-LLM+SHACL",
}


def load_arm(arm: str) -> list[dict]:
    path = IN_DIR / f"{arm}.jsonl"
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def arm_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n   = len(rows)
    acc = sum(r["correct"] for r in rows) / n

    shacl_vals = [r["shacl_conform"] for r in rows
                  if r.get("shacl_conform") is not None]
    shacl_rate = (sum(shacl_vals) / len(shacl_vals)) if shacl_vals else None

    grc   = sum(bool(r.get("grc_complete")) for r in rows) / n

    depth_vals = [r["grounding_depth"] for r in rows
                  if r.get("grounding_depth") is not None]
    depth = statistics.mean(depth_vals) if depth_vals else None

    parse_errs = sum(r.get("parse_error", False) for r in rows)

    by_class: dict[str, list[bool]] = {}
    for r in rows:
        cls = r.get("true_class", "?")
        by_class.setdefault(cls, []).append(r["correct"])
    per_class = {cls: round(sum(v)/len(v), 4) for cls, v in sorted(by_class.items())}

    return {
        "n":                   n,
        "risk_level_accuracy": round(acc, 4),
        "shacl_conform_rate":  round(shacl_rate, 4) if shacl_rate is not None else None,
        "grc_completeness":    round(grc, 4),
        "grounding_depth":     round(depth, 2) if depth is not None else None,
        "parse_errors":        parse_errs,
        "per_class_accuracy":  per_class,
    }


def main() -> None:
    summary: dict = {"arms": {}}
    for arm in ARM_ORDER:
        rows = load_arm(arm)
        if rows:
            label = ARM_LABELS[arm]
            summary["arms"][label] = arm_metrics(rows)
            print(f"Loaded {arm}: {len(rows)} records")
        else:
            print(f"SKIP {arm}: no file found")

    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {OUT}\n")

    METRICS = ["risk_level_accuracy", "shacl_conform_rate",
               "grc_completeness", "grounding_depth"]
    col_w = 22
    header = f"{'Arm':<20}" + "".join(f"{m:<{col_w}}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for arm in ARM_ORDER:
        label = ARM_LABELS[arm]
        m_dict = summary["arms"].get(label, {})
        row = f"{label:<20}"
        for m in METRICS:
            v = m_dict.get(m)
            cell = f"{v:.4f}" if isinstance(v, float) else ("-" if v is None else str(v))
            row += f"{cell:<{col_w}}"
        print(row)


if __name__ == "__main__":
    main()
