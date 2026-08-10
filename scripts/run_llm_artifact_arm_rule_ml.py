"""Rule+ML arm — rule handles Set A (unambiguous), ML handles Set B (masked/ambiguous)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts.llm_artifact_lib import stratified_slice
from scripts.run_llm_artifact_arm_rule import run_rule_on_case
from scripts.run_llm_artifact_arm_ml import train_ml_arm, predict_with_ml_arm


def run_rule_ml_on_cases(cases: list[dict], *, set_a_for_training: list[dict]) -> list[dict]:
    model_bundle = train_ml_arm(set_a_for_training)
    set_b = [c for c in cases if c["subset"] == "B"]
    ml_preds = {a["case_id"]: a for a in predict_with_ml_arm(model_bundle, set_b)}

    out = []
    for case in cases:
        if case["subset"] == "A":
            artifact = run_rule_on_case(case)
        else:
            artifact = dict(ml_preds[case["case_id"]])
        artifact["arm"] = "rule_ml"
        artifact["ground_truth_risk_level"] = case.get("ground_truth_risk_level", "")
        artifact["true_attack_class"] = case.get("true_attack_class", "")
        artifact["subset"] = case.get("subset", "")
        out.append(artifact)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", default="results/llm_experiment/test_set.json")
    ap.add_argument("--out", default="results/llm_experiment/raw/run_1_arm_rule_ml.jsonl")
    ap.add_argument("--n-per-class", type=int, default=None)
    args = ap.parse_args()

    all_cases = json.loads(Path(args.test_set).read_text())["cases"]
    cases = stratified_slice(all_cases, args.n_per_class)
    set_a = [c for c in cases if c["subset"] == "A"]
    print(f"Rule+ML arm | Cases: {len(cases)} (Set A: {len(set_a)})")
    artifacts = run_rule_ml_on_cases(cases, set_a_for_training=set_a)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as f:
        for a in artifacts:
            a["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            f.write(json.dumps(a) + "\n")
    correct = sum(1 for a in artifacts if a["risk_level"] == a.get("ground_truth_risk_level"))
    print(f"Wrote {len(artifacts)} rule_ml-arm artifacts to {args.out} | acc={correct/len(artifacts)*100:.1f}%")


if __name__ == "__main__":
    main()
