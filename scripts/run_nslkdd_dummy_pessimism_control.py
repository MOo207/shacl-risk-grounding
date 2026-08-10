"""Constant-worst-class (always-DoS) pessimism control for the NSL-KDD safety battery.

Pre-registered intent (written before running): a zero-information dummy that
always predicts attack class DoS, mapped through the same NFCRM-1:2025 Sec 6.9
lookup and the same most-severe-wins arbitration against the fixed XGBoost
baseline, bounds how much of the tri-stage safety headline is explained by
pessimism alone. Reviewer-requested control (DKE round 5, DA-CRITICAL-2).

Interpretation rule fixed in advance: whatever the dummy scores is reported
verbatim. If the dummy dominates the tri-stage arm on under-escalation and
severe recall, the paper's safety claim must be reframed around calibration
(exact accuracy), bounded over-escalation, parse-failure absorption, and GRC
artefact generation -- not the headline safety metrics.
"""
import json

SEV = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
SEVERE = {"High", "Catastrophic"}
LOOKUP = json.load(open("results/nslkdd_natural_underescalation.json"))["sixnine_lookup_table"]

cases = json.load(open("results/nslkdd_unified_rerun/test_sample.json"))["cases"]
tri = {}
with open("results/nslkdd_unified_rerun/reconciled_tristage_haiku_test.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        tri[r["case_id"]] = r


def sev_i(x):
    return SEV.index(x)


def score(pred_level, gt):
    return {"under": sev_i(pred_level) < sev_i(gt), "over": sev_i(pred_level) > sev_i(gt),
            "exact": pred_level == gt, "pred": pred_level, "gt": gt}


def metrics(rows):
    n = len(rows)
    sg = [r for r in rows if r["gt"] in SEVERE]
    return {"n": n,
            "under_escalation_rate": sum(r["under"] for r in rows) / n,
            "over_escalation_rate": sum(r["over"] for r in rows) / n,
            "exact_accuracy": sum(r["exact"] for r in rows) / n,
            "severe_recall": sum(r["pred"] in SEVERE for r in sg) / len(sg),
            "n_severe_gt": len(sg)}


standalone, arbitrated, percase = [], [], []
for c in cases:
    cid = c["case_id"]
    crit = c["criticality"]
    gt = tri[cid]["ground_truth"]
    dummy_level = LOOKUP["DoS"][crit]
    ml_level = tri[cid]["ml_risk_level"]
    arb_level = max(dummy_level, ml_level, key=sev_i)
    standalone.append(score(dummy_level, gt))
    arbitrated.append(score(arb_level, gt))
    percase.append({"case_id": cid, "true_class": c["true_attack_class"], "criticality": crit,
                    "gt": gt, "dummy_level": dummy_level, "ml_level": ml_level,
                    "arbitrated_level": arb_level})

out = {"description": "Constant-DoS pessimism dummy: standalone and most-severe-wins "
                      "arbitration vs fixed XGBoost baseline, scored identically to the "
                      "tri-stage arms on the clean N=100 NSL-KDD sample.",
       "lookup_source": "results/nslkdd_natural_underescalation.json:sixnine_lookup_table",
       "ground_truth_source": "results/nslkdd_unified_rerun/reconciled_tristage_haiku_test.jsonl",
       "standalone_constant_dos": metrics(standalone),
       "arbitrated_constant_dos_vs_xgb": metrics(arbitrated),
       "per_case": percase}
with open("results/nslkdd_dummy_pessimism_control.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: out[k] for k in ("standalone_constant_dos", "arbitrated_constant_dos_vs_xgb")}, indent=2))
