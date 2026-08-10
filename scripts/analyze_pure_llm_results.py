"""Quick analysis of Pure LLM arm results for thesis table update."""
import json, sys
from pathlib import Path
from scipy import stats as scipy_stats

ROOT = Path(__file__).parent.parent


def analyze_arm(path: Path, arm_name: str) -> dict:
    if not path.exists():
        print(f"[MISSING] {path}")
        return {}
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    cases = [json.loads(l) for l in lines]
    print(f"\n=== {arm_name} ({len(cases)} cases) ===")

    total = len(cases)
    correct = sum(1 for c in cases if c.get("risk_level") == c.get("ground_truth_risk_level"))
    parse_errs = sum(1 for c in cases if c.get("parse_error"))
    shacl_fails = sum(1 for c in cases if c.get("shacl_violations"))

    set_a = [c for c in cases if c.get("subset") == "A"]
    set_b = [c for c in cases if c.get("subset") == "B"]
    a_correct = sum(1 for c in set_a if c.get("risk_level") == c.get("ground_truth_risk_level"))
    b_correct = sum(1 for c in set_b if c.get("risk_level") == c.get("ground_truth_risk_level"))

    # Wilson CI for Set B
    n_b = len(set_b)
    p_b = b_correct / n_b if n_b else 0
    ci_lo = ci_hi = 0
    if n_b > 0:
        lo, hi = _wilson_ci(b_correct, n_b, 0.95)
        ci_lo, ci_hi = round(lo * 100, 1), round(hi * 100, 1)

    overall_acc = correct / total if total else 0
    a_acc = a_correct / len(set_a) if set_a else 0
    b_acc = p_b

    print(f"  Overall: {correct}/{total} = {overall_acc:.1%}")
    print(f"  Set A:   {a_correct}/{len(set_a)} = {a_acc:.1%}")
    print(f"  Set B:   {b_correct}/{n_b} = {b_acc:.1%}  [95% CI {ci_lo}%,{ci_hi}%]")
    print(f"  Parse errors: {parse_errs} | SHACL fails: {shacl_fails}")

    # Per-class breakdown for Set A
    from collections import defaultdict
    per_class = defaultdict(lambda: [0, 0])
    for c in set_a:
        cls = c.get("true_attack_class", "?")
        per_class[cls][1] += 1
        if c.get("risk_level") == c.get("ground_truth_risk_level"):
            per_class[cls][0] += 1
    print("  Set A per-class:")
    for cls, (ok, tot) in sorted(per_class.items()):
        print(f"    {cls:<15} {ok}/{tot} = {ok/tot:.0%}")

    return {
        "arm": arm_name, "n": total, "n_a": len(set_a), "n_b": n_b,
        "overall": overall_acc, "set_a": a_acc, "set_b": b_acc,
        "set_b_ci_lo": ci_lo, "set_b_ci_hi": ci_hi,
        "parse_errors": parse_errs, "shacl_fails": shacl_fails,
    }


def _wilson_ci(k, n, conf=0.95):
    from math import sqrt
    z = scipy_stats.norm.ppf(1 - (1 - conf) / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - margin), min(1, center + margin)


def analyze_nl_classification(path: Path, name: str) -> dict:
    if not path.exists():
        print(f"[MISSING] {path}")
        return {}
    d = json.loads(path.read_text())
    acc = d.get("accuracy", 0)
    n = d.get("n_valid", d.get("n_samples", 0))
    errors = d.get("n_errors", 0)
    print(f"\n=== {name} ===")
    print(f"  N valid: {n} ({errors} errors)")
    print(f"  Accuracy: {acc:.2%}")
    print("  Per-class:")
    for cls, a in sorted(d.get("per_class_accuracy", {}).items()):
        cnt = d.get("per_class_counts", {}).get(cls, 0)
        print(f"    {cls:<15} {a:.1%}  (n={cnt})")
    return {"name": name, "accuracy": acc, "n_valid": n, "n_errors": errors,
            "per_class": d.get("per_class_accuracy", {})}


if __name__ == "__main__":
    raw = ROOT / "results" / "llm_experiment" / "raw"

    # Pure LLM arms (Task ii)
    haiku = analyze_arm(raw / "run_1_arm_pure_llm_haiku.jsonl", "Pure LLM / Haiku")
    sonnet = analyze_arm(raw / "run_1_arm_pure_llm_sonnet.jsonl", "Pure LLM / Sonnet")

    # Sonnet Task(i) N=180
    analyze_nl_classification(
        ROOT / "results" / "slm_nl_classification_sonnet_n180.json",
        "Sonnet NL Classification (N=180)"
    )

    # Compare with Rule+LLM baselines
    print("\n=== COMPARISON (Set B) ===")
    print(f"  Rule+LLM Haiku 4.5:   36% [20,55]  (known)")
    print(f"  Rule+LLM Sonnet 4.6:  44% [27,63]  (known)")
    if haiku:
        print(f"  Pure LLM Haiku 4.5:   {haiku['set_b']:.0%} [{haiku['set_b_ci_lo']},{haiku['set_b_ci_hi']}]")
    if sonnet:
        print(f"  Pure LLM Sonnet 4.6:  {sonnet['set_b']:.0%} [{sonnet['set_b_ci_lo']},{sonnet['set_b_ci_hi']}]")
