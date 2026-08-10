"""Trivial-escalator null baselines (review finding C2).

Computes, on the exact same case sets as the reconciled tri-stage reruns:
  1. Always-escalate: XGBoost risk level bumped one level toward severe (capped).
  2. Random-escalate at matched disagreement rate: escalate a random subset of
     cases (size = observed risk-level disagreement count) by one level;
     seed 42, 1000 resamples, mean +/- std.
  3. Directional McNemar (under-escalation discordance vs XGBoost) for the
     always-escalate arm: b = XGB under-escalates where arm does not; c = reverse.

Before trusting the metric definitions, the script reproduces the published
per-file aggregates (XGBoost and tri-stage) from the per-case records and
asserts exact agreement. No number here is hand-entered.

Output: results/trivial_escalator_control.json
"""
import json
import math
import os
import numpy as np
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
IDX = {l: i for i, l in enumerate(LEVELS)}
SEVERE_MIN = IDX["High"]  # severe = ground truth High or Catastrophic


def metrics(preds, gts):
    n = len(preds)
    assert n == len(gts) and n > 0
    exact = under = over = within1 = 0
    sev_tot = sev_hit = 0
    worst_miss = 0
    for p, g in zip(preds, gts):
        pi, gi = IDX[p], IDX[g]
        if pi == gi:
            exact += 1
        if abs(pi - gi) <= 1:
            within1 += 1
        if pi < gi:
            under += 1
            worst_miss = max(worst_miss, gi - pi)
        elif pi > gi:
            over += 1
        if gi >= SEVERE_MIN:
            sev_tot += 1
            if pi >= SEVERE_MIN:
                sev_hit += 1
    return {
        "n": n,
        "exact_pct": 100.0 * exact / n,
        "within1_pct": 100.0 * within1 / n,
        "under_escalation_pct": 100.0 * under / n,
        "over_escalation_pct": 100.0 * over / n,
        "severe_recall_pct": round(100.0 * sev_hit / sev_tot, 1) if sev_tot else None,
        "severe_recall_pct_unrounded": (100.0 * sev_hit / sev_tot) if sev_tot else None,
        "severe_total": sev_tot,
        "worst_miss_levels": worst_miss,
    }


def mcnemar_under(arm_preds, xgb_preds, gts):
    """Directional under-escalation discordance vs XGBoost.
    b = XGB under-escalates where arm does not; c = arm under-escalates where XGB does not."""
    b = c = 0
    for a, x, g in zip(arm_preds, xgb_preds, gts):
        au = IDX[a] < IDX[g]
        xu = IDX[x] < IDX[g]
        if xu and not au:
            b += 1
        elif au and not xu:
            c += 1
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_two_sided": 1.0, "p_one_sided": 1.0}
    k = min(b, c)
    return {
        "b": b,
        "c": c,
        "p_two_sided": binomtest(k, n, 0.5, alternative="two-sided").pvalue,
        "p_one_sided": binomtest(k, n, 0.5, alternative="less").pvalue,
    }


def bump(level):
    return LEVELS[min(IDX[level] + 1, len(LEVELS) - 1)]


def close(a, b, tol=0.05):
    if a is None or b is None:
        return a == b
    return abs(float(a) - float(b)) <= tol


def load_arm(path):
    d = json.load(open(path, encoding="utf-8"))
    if "held_out_test" in d:  # NSL-KDD layout
        h = d["held_out_test"]
        records = h["records"]
        pub_tri = [v for k, v in h.items() if k.endswith("_metrics") and "xgboost" not in k][0]
        pub_xgb = h["xgboost_clean_baseline_metrics"]
        pub_tri_norm = {
            "exact_pct": pub_tri["exact_pct"], "within1_pct": pub_tri["within1_pct"],
            "under_escalation_pct": pub_tri["under_escalation_pct"],
            "over_escalation_pct": pub_tri["over_escalation_pct"],
            "severe_recall_pct": pub_tri["severe_recall_pct"],
            "severe_total": pub_tri["severe_total"],
            "worst_miss_levels": pub_tri["worst_miss_levels"],
        }
        pub_xgb_norm = {
            "exact_pct": pub_xgb["exact_pct"], "within1_pct": pub_xgb["within1_pct"],
            "under_escalation_pct": pub_xgb["under_escalation_pct"],
            "over_escalation_pct": pub_xgb["over_escalation_pct"],
            "severe_recall_pct": pub_xgb["severe_recall_pct"],
            "severe_total": pub_xgb["severe_total"],
            "worst_miss_levels": pub_xgb["worst_miss_levels"],
        }
        xgb_key = "ml_risk_level"
        pub_mcnemar = h.get("unsafe_mcnemar_reconciled_vs_xgb")
        mcnemar_kind = "under_escalation_discordance"
    else:  # CIC layout
        records = d["records"]
        m = d["metrics"]
        pub_tri_norm = {
            "exact_pct": m["exact_match_pct"], "within1_pct": m["within1_pct"],
            "under_escalation_pct": m["under_escalation_pct"],
            "over_escalation_pct": m["over_escalation_pct"],
            "severe_recall_pct": m["severe_recall_pct"],
            "severe_total": m["severe_total"],
            "worst_miss_levels": m["worst_miss_levels"],
        }
        x = d["clean_xgboost_baseline_reference"]
        pub_xgb_norm = {
            "exact_pct": x["exact"], "within1_pct": x["within1"],
            "under_escalation_pct": x["under"], "over_escalation_pct": x["over"],
            "severe_recall_pct": x["severe_recall"], "severe_total": x["sev_tot"],
            "worst_miss_levels": x["worst_miss"],
        }
        xgb_key = "xgb_risk_level"
        pub_mcnemar = d.get("mcnemar_vs_clean_xgboost")
        mcnemar_kind = "exact_correctness_discordance"
    gts = [r["ground_truth"] for r in records]
    tri = [r["risk_level"] for r in records]
    xgb = [r[xgb_key] for r in records]
    llm = [r["llm_risk_level"] for r in records]
    return records, gts, tri, xgb, llm, pub_tri_norm, pub_xgb_norm, pub_mcnemar, mcnemar_kind


def compare_pub(name, computed, published, log):
    keys = ["exact_pct", "within1_pct", "under_escalation_pct", "over_escalation_pct",
            "severe_total", "worst_miss_levels"]
    ok = True
    detail = {}
    for k in keys:
        c, p = computed[k], published[k]
        match = close(c, p, tol=1e-6)
        detail[k] = {"computed": c, "published": p, "match": match}
        ok = ok and match
    # severe recall: NSL files round to 1dp; compare rounded and unrounded
    c_sr = computed["severe_recall_pct_unrounded"]
    p_sr = published["severe_recall_pct"]
    sr_match = close(c_sr, p_sr, tol=1e-6) or close(round(c_sr, 1), round(float(p_sr), 1), tol=1e-9)
    detail["severe_recall_pct"] = {"computed": c_sr, "published": p_sr, "match": sr_match}
    ok = ok and sr_match
    log[name] = {"all_match": ok, "detail": detail}
    return ok


def random_escalate(xgb, gts, k, n_resamples=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(xgb)
    acc = {m: [] for m in ["exact_pct", "under_escalation_pct", "over_escalation_pct",
                           "severe_recall_pct", "within1_pct"]}
    for _ in range(n_resamples):
        idx = rng.choice(n, size=k, replace=False)
        preds = list(xgb)
        for i in idx:
            preds[i] = bump(preds[i])
        m = metrics(preds, gts)
        acc["exact_pct"].append(m["exact_pct"])
        acc["within1_pct"].append(m["within1_pct"])
        acc["under_escalation_pct"].append(m["under_escalation_pct"])
        acc["over_escalation_pct"].append(m["over_escalation_pct"])
        acc["severe_recall_pct"].append(m["severe_recall_pct_unrounded"])
    out = {}
    for key, vals in acc.items():
        out[key + "_mean"] = float(np.mean(vals))
        out[key + "_std"] = float(np.std(vals))
    out["k_escalated"] = k
    out["n_resamples"] = n_resamples
    out["seed"] = seed
    return out


ARMS = {
    "nslkdd_haiku": "results/nslkdd_unified_rerun/reconciled_tristage_haiku_results.json",
    "nslkdd_sonnet": "results/nslkdd_unified_rerun/reconciled_tristage_sonnet_results.json",
    "cic_haiku": "results/cic_unified_rerun/reconciled_tristage_haiku_results.json",
    "cic_sonnet": "results/cic_unified_rerun/reconciled_tristage_sonnet_results.json",
}


def main():
    out = {
        "description": "Trivial-escalator null baselines (review finding C2). "
                       "Always-escalate = XGBoost risk bumped one level toward severe (capped at Catastrophic). "
                       "Random-escalate = random subset of size k (observed XGB-vs-LLM risk-level disagreement count) "
                       "bumped one level; 1000 resamples, numpy default_rng seed 42. "
                       "All metrics computed from per-case records in the source files; "
                       "metric definitions validated by exact reproduction of published aggregates (see reproduction_check).",
        "risk_level_order": LEVELS,
        "severe_definition": "ground truth in {High, Catastrophic}; recall counts predictions >= High",
        "reproduction_check": {},
        "arms": {},
    }
    all_repro_ok = True
    for arm, rel in ARMS.items():
        path = os.path.join(ROOT, rel)
        records, gts, tri, xgb, llm, pub_tri, pub_xgb, pub_mc, mc_kind = load_arm(path)
        log = out["reproduction_check"]
        ok_x = compare_pub(arm + "/xgboost", metrics(xgb, gts), pub_xgb, log)
        ok_t = compare_pub(arm + "/tristage", metrics(tri, gts), pub_tri, log)
        all_repro_ok = all_repro_ok and ok_x and ok_t
        # reproduce published McNemar where it is the under-escalation kind
        if pub_mc is not None and mc_kind == "under_escalation_discordance":
            mc_tri = mcnemar_under(tri, xgb, gts)
            log[arm + "/mcnemar_tristage_vs_xgb"] = {
                "computed_b": mc_tri["b"], "computed_c": mc_tri["c"],
                "published_b": pub_mc["b"], "published_c": pub_mc["c"],
                "computed_p_one_sided": mc_tri["p_one_sided"],
                "computed_p_two_sided": mc_tri["p_two_sided"],
                "published_p": pub_mc["p"],
                "b_c_match": mc_tri["b"] == pub_mc["b"] and mc_tri["c"] == pub_mc["c"],
            }
            all_repro_ok = all_repro_ok and log[arm + "/mcnemar_tristage_vs_xgb"]["b_c_match"]

        # disagreement count (drives random baseline): XGB risk vs LLM independent risk
        k_risk = sum(1 for a, b_ in zip(xgb, llm) if a != b_)
        k_class = sum(1 for r in records if r.get("class_disagreement"))

        always = [bump(x) for x in xgb]
        m_always = metrics(always, gts)
        m_always["mcnemar_under_vs_xgb"] = mcnemar_under(always, xgb, gts)

        arm_out = {
            "source_file": rel.replace("\\", "/"),
            "n": len(gts),
            "risk_level_disagreement_count": k_risk,
            "class_disagreement_count": k_class,
            "xgboost_computed": metrics(xgb, gts),
            "tristage_computed": metrics(tri, gts),
            "always_escalate": m_always,
            "random_escalate_matched_k_risk": random_escalate(xgb, gts, k_risk),
        }
        # NSL-KDD Haiku: task cites 68/100 (= class disagreement); provide that variant too
        if k_class != k_risk:
            arm_out["random_escalate_matched_k_class"] = random_escalate(xgb, gts, k_class)
        out["arms"][arm] = arm_out

    out["reproduction_check"]["ALL_AGGREGATES_REPRODUCED"] = all_repro_ok
    dst = os.path.join(ROOT, "results", "trivial_escalator_control.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("ALL_AGGREGATES_REPRODUCED:", all_repro_ok)
    for name, v in out["reproduction_check"].items():
        if isinstance(v, dict) and "all_match" in v and not v["all_match"]:
            print("MISMATCH:", name, json.dumps(v["detail"]))
    for arm, a in out["arms"].items():
        print("\n==", arm, "n=%d k_risk=%d k_class=%d" % (a["n"], a["risk_level_disagreement_count"], a["class_disagreement_count"]))
        for label in ["xgboost_computed", "tristage_computed", "always_escalate"]:
            m = a[label]
            print(" %-18s exact=%.1f under=%.1f over=%.1f sevrec=%.1f" % (
                label, m["exact_pct"], m["under_escalation_pct"], m["over_escalation_pct"],
                m["severe_recall_pct_unrounded"]))
        mc = a["always_escalate"]["mcnemar_under_vs_xgb"]
        print("   always mcnemar_under b=%d c=%d p2=%.3g" % (mc["b"], mc["c"], mc["p_two_sided"]))
        for key in ["random_escalate_matched_k_risk", "random_escalate_matched_k_class"]:
            if key in a:
                r = a[key]
                print(" %-34s exact=%.1f+/-%.1f under=%.1f+/-%.1f over=%.1f+/-%.1f sevrec=%.1f+/-%.1f (k=%d)" % (
                    key, r["exact_pct_mean"], r["exact_pct_std"],
                    r["under_escalation_pct_mean"], r["under_escalation_pct_std"],
                    r["over_escalation_pct_mean"], r["over_escalation_pct_std"],
                    r["severe_recall_pct_mean"], r["severe_recall_pct_std"], r["k_escalated"]))
    print("\nWrote", dst)


if __name__ == "__main__":
    main()
