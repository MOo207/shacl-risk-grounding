"""Score XGBoost (pre-computed §6.9 risk-level predictions) on the exact
case-id pools actually answered by the two tri-stage LLM arms.

Context
-------
`results/unified_ablation/tristage_llm_sonnet.jsonl` and
`tristage_llm_haiku.jsonl` only cover 148 and 164 of the 180 unified-ablation
cases respectively (the rest were lost to Claude API rate-limiting during
batch execution). The thesis currently compares those arms' full-pool
accuracy (96.3-96.6%) to XGBoost's 95.0%, but that 95.0% figure was scored on
the DIFFERENT, smaller, class-balanced N=60 subset. This script re-scores
XGBoost's existing §6.9 risk-level prediction (`xgb_risk_level`, already
present in `test_set.json` per case -- see `scripts/run_unified_ablation.py`,
which reuses this field directly rather than retraining) against the SAME
148-case and 164-case pools, so the comparison is apples-to-apples.

"Answered" is defined the same way the thesis already counts 148/164: a
case_id present in the tri-stage arm's output JSONL (both files have
parse_error=False for every row, i.e. zero unparsed/erroring predictions
within what was returned -- attrition is 100% missing rows, not partial
parse failures).

Output: results/unified_ablation/xgb_matched_pool.json
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
TEST_SET = ROOT / "results/unified_ablation/test_set.json"
SONNET_JSONL = ROOT / "results/unified_ablation/tristage_llm_sonnet.jsonl"
HAIKU_JSONL = ROOT / "results/unified_ablation/tristage_llm_haiku.jsonl"
OUT_PATH = ROOT / "results/unified_ablation/xgb_matched_pool.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    test_set = json.loads(TEST_SET.read_text(encoding="utf-8"))
    all_cases = {c["case_id"]: c for c in test_set["cases"]}
    all_ids = set(all_cases.keys())
    print(f"Full unified-ablation test set: {len(all_ids)} cases")

    sonnet_rows = load_jsonl(SONNET_JSONL)
    haiku_rows = load_jsonl(HAIKU_JSONL)

    sonnet_ids = {r["case_id"] for r in sonnet_rows}
    haiku_ids = {r["case_id"] for r in haiku_rows}
    print(f"Sonnet answered: {len(sonnet_ids)} (parse_error rows: "
          f"{sum(1 for r in sonnet_rows if r.get('parse_error'))})")
    print(f"Haiku answered:  {len(haiku_ids)} (parse_error rows: "
          f"{sum(1 for r in haiku_rows if r.get('parse_error'))})")

    sonnet_missing = sorted(all_ids - sonnet_ids)
    haiku_missing = sorted(all_ids - haiku_ids)
    print(f"Sonnet missing: {len(sonnet_missing)}")
    print(f"Haiku missing:  {len(haiku_missing)}")

    def class_of(cid: str) -> str:
        return all_cases[cid]["true_attack_class"]

    missing_ids_by_class = {
        "sonnet_pool148": dict(Counter(class_of(cid) for cid in sonnet_missing)),
        "haiku_pool164": dict(Counter(class_of(cid) for cid in haiku_missing)),
    }

    def xgb_accuracy(ids: set[str]) -> tuple[float, int, int, dict]:
        n = len(ids)
        correct = 0
        per_class_correct: Counter = Counter()
        per_class_total: Counter = Counter()
        for cid in ids:
            case = all_cases[cid]
            cls = case["true_attack_class"]
            per_class_total[cls] += 1
            if case["xgb_risk_level"] == case["ground_truth_risk_level"]:
                correct += 1
                per_class_correct[cls] += 1
        acc = correct / n if n else 0.0
        per_class = {
            cls: {
                "correct": per_class_correct.get(cls, 0),
                "total": per_class_total.get(cls, 0),
                "accuracy": (per_class_correct.get(cls, 0) / per_class_total[cls]
                             if per_class_total.get(cls) else 0.0),
            }
            for cls in sorted(per_class_total)
        }
        return acc, correct, n, per_class

    xgb_acc_pool148, xgb_correct_148, n148, xgb_per_class_148 = xgb_accuracy(sonnet_ids)
    xgb_acc_pool164, xgb_correct_164, n164, xgb_per_class_164 = xgb_accuracy(haiku_ids)

    def tristage_accuracy(rows: list[dict]) -> tuple[float, int, int]:
        n = len(rows)
        correct = sum(1 for r in rows if r.get("correct"))
        return (correct / n if n else 0.0), correct, n

    sonnet_acc, sonnet_correct, sonnet_n = tristage_accuracy(sonnet_rows)
    haiku_acc, haiku_correct, haiku_n = tristage_accuracy(haiku_rows)

    result = {
        "description": (
            "XGBoost §6.9 risk-level accuracy scored on the exact case-id "
            "pools answered by the tri-stage LLM arms (matched-pool "
            "comparison), vs. the tri-stage arms' own accuracy on those "
            "same pools."
        ),
        "xgb_acc_pool164": xgb_acc_pool164,
        "xgb_acc_pool148": xgb_acc_pool148,
        "xgb_correct_pool164": xgb_correct_164,
        "xgb_correct_pool148": xgb_correct_148,
        "n_pool164": n164,
        "n_pool148": n148,
        "xgb_per_class_pool148_sonnet": xgb_per_class_148,
        "xgb_per_class_pool164_haiku": xgb_per_class_164,
        "tristage_sonnet_acc_pool148": sonnet_acc,
        "tristage_sonnet_correct_pool148": sonnet_correct,
        "tristage_sonnet_n_pool148": sonnet_n,
        "tristage_haiku_acc_pool164": haiku_acc,
        "tristage_haiku_correct_pool164": haiku_correct,
        "tristage_haiku_n_pool164": haiku_n,
        "missing_ids_by_class": missing_ids_by_class,
        "sonnet_missing_ids": sonnet_missing,
        "haiku_missing_ids": haiku_missing,
        "sonnet_missing_n": len(sonnet_missing),
        "haiku_missing_n": len(haiku_missing),
    }

    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")
    print(f"\nXGBoost matched-pool accuracy:")
    print(f"  Pool148 (Sonnet's pool): {xgb_acc_pool148:.1%} ({xgb_correct_148}/{n148})")
    print(f"  Pool164 (Haiku's pool):  {xgb_acc_pool164:.1%} ({xgb_correct_164}/{n164})")
    print(f"\nTri-stage accuracy on own pool:")
    print(f"  Sonnet (pool148): {sonnet_acc:.1%} ({sonnet_correct}/{sonnet_n})")
    print(f"  Haiku  (pool164): {haiku_acc:.1%} ({haiku_correct}/{haiku_n})")
    print(f"\nMissing-id class distribution:")
    print(f"  Sonnet missing (32 expected): {missing_ids_by_class['sonnet_pool148']}")
    print(f"  Haiku missing (16 expected):  {missing_ids_by_class['haiku_pool164']}")


if __name__ == "__main__":
    main()
