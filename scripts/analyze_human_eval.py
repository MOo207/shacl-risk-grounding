"""Analyse returned practitioner rating sheets.

Runs the analysis pre-registered in human_eval/sample/KEY_do_not_distribute.json:

  Primary   H1 (one-sided): SHACL-conforming artefacts receive higher mean
            practitioner ratings than rejected ones. Mann-Whitney U, alpha=0.05.
  Secondary Krippendorff's alpha (ordinal) for inter-rater reliability, reported
            whatever H1 shows.
  Also      per-dimension descriptives with bootstrap CIs, acceptance rate by
            stratum, and an achieved-power note.

Both statistics are implemented here directly rather than pulled from a
dependency, so the numbers are auditable alongside the data.

Usage: python scripts/analyze_human_eval.py human_eval/sample/returned/
Output: results/human_eval_analysis.json
"""
import argparse
import csv
import glob
import json
import os
import random
import statistics
from collections import defaultdict
from itertools import combinations

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
KEY_PATH = os.path.join(BASE_DIR, "human_eval", "sample", "KEY_do_not_distribute.json")
OUT_PATH = os.path.join(BASE_DIR, "results", "human_eval_analysis.json")

DIMENSIONS = ["clause_correctness", "audit_completeness", "control_actionability"]


# ---------------------------------------------------------------- statistics

def mannwhitney_u(a, b):
    """Two-sample Mann-Whitney U with tie-corrected normal approximation.

    Returns (U, z, p_one_sided) testing whether `a` is stochastically greater.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return None, None, None
    pooled = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = [0.0] * len(pooled), 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1

    ra = sum(r for r, (_, grp) in zip(ranks, pooled) if grp == 0)
    ua = ra - na * (na + 1) / 2.0
    mu = na * nb / 2.0

    tie_term, i = 0.0, 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    n = na + nb
    var = na * nb / 12.0 * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return ua, None, None
    z = (ua - mu) / (var ** 0.5)
    # one-sided upper-tail p via the normal CDF (Abramowitz-Stegun 7.1.26)
    p = 0.5 * erfc(z / (2 ** 0.5))
    return ua, z, p


def erfc(x):
    """Complementary error function, Abramowitz & Stegun 7.1.26."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * pow(2.718281828459045, -x * x)
    erf = sign * y
    return 1.0 - erf


def krippendorff_alpha_ordinal(ratings):
    """Krippendorff's alpha for ordinal data.

    `ratings` maps unit -> list of values from different raters. Units with
    fewer than two ratings contribute nothing, per the standard definition.
    """
    units = {u: [v for v in vs if v is not None] for u, vs in ratings.items()}
    units = {u: vs for u, vs in units.items() if len(vs) >= 2}
    if not units:
        return None

    values = sorted({v for vs in units.values() for v in vs})
    if len(values) < 2:
        return 1.0
    counts = defaultdict(int)
    for vs in units.values():
        for v in vs:
            counts[v] += 1
    n_total = sum(counts.values())

    # Ordinal difference function over the marginal-frequency scale.
    def delta(v1, v2):
        i1, i2 = values.index(v1), values.index(v2)
        lo, hi = min(i1, i2), max(i1, i2)
        s = sum(counts[values[k]] for k in range(lo, hi + 1))
        s -= (counts[v1] + counts[v2]) / 2.0
        return s ** 2

    d_obs, pairs_obs = 0.0, 0
    for vs in units.values():
        m = len(vs)
        for v1, v2 in combinations(vs, 2):
            d_obs += delta(v1, v2) / (m - 1)
            pairs_obs += 1 / (m - 1)
    if pairs_obs == 0:
        return None
    do = d_obs / pairs_obs

    d_exp, pairs_exp = 0.0, 0
    vals = [v for vs in units.values() for v in vs]
    for v1, v2 in combinations(vals, 2):
        d_exp += delta(v1, v2)
        pairs_exp += 1
    de = d_exp / pairs_exp if pairs_exp else 0.0
    return 1.0 - do / de if de else None


def bootstrap_ci(vals, iters=5000, seed=42):
    if not vals:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        s = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(statistics.mean(s))
    means.sort()
    return [round(means[int(0.025 * iters)], 3), round(means[int(0.975 * iters)], 3)]


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("returned_dir",
                    help="directory of returned rater_NN.csv sheets")
    args = ap.parse_args()

    with open(KEY_PATH, encoding="utf-8") as f:
        key = json.load(f)
    meta = {it["artefact_id"]: it for it in key["items"]}

    sheets = sorted(glob.glob(os.path.join(args.returned_dir, "rater_*.csv")))
    if not sheets:
        raise SystemExit(f"No rater_*.csv found in {args.returned_dir}")

    # rater -> artefact -> {dimension: score}
    data, accepts, comments = {}, defaultdict(list), []
    for path in sheets:
        rater = os.path.splitext(os.path.basename(path))[0]
        data[rater] = {}
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                aid = (row.get("artefact_id") or "").strip()
                if aid not in meta:
                    continue
                scores = {}
                for d in DIMENSIONS:
                    raw = (row.get(d) or "").strip()
                    if raw:
                        try:
                            v = int(float(raw))
                        except ValueError:
                            continue
                        if 1 <= v <= 5:
                            scores[d] = v
                if scores:
                    data[rater][aid] = scores
                acc = (row.get("accept_into_register") or "").strip().lower()
                if acc in ("yes", "y", "1", "true"):
                    accepts[aid].append(1)
                elif acc in ("no", "n", "0", "false"):
                    accepts[aid].append(0)
                c = (row.get("free_text_comment") or "").strip()
                if c:
                    comments.append({"rater": rater, "artefact_id": aid, "comment": c})

    rated = {a for r in data.values() for a in r}
    if not rated:
        raise SystemExit("Sheets contained no usable ratings.")

    # Mean across dimensions, per rater per artefact; then per artefact.
    per_artefact = defaultdict(list)
    for rater, rows in data.items():
        for aid, sc in rows.items():
            per_artefact[aid].append(statistics.mean(sc.values()))
    artefact_mean = {a: statistics.mean(v) for a, v in per_artefact.items()}

    conforming = [artefact_mean[a] for a in artefact_mean if meta[a]["shacl_conforms"]]
    rejected = [artefact_mean[a] for a in artefact_mean if not meta[a]["shacl_conforms"]]
    u, z, p = mannwhitney_u(conforming, rejected)

    # Reliability, per dimension and pooled.
    reliability = {}
    for d in DIMENSIONS + ["overall"]:
        ratings = defaultdict(list)
        for rater, rows in data.items():
            for aid, sc in rows.items():
                if d == "overall":
                    ratings[aid].append(round(statistics.mean(sc.values())))
                elif d in sc:
                    ratings[aid].append(sc[d])
        reliability[d] = krippendorff_alpha_ordinal(ratings)

    dim_stats = {}
    for d in DIMENSIONS:
        vals = [sc[d] for rows in data.values() for sc in rows.values() if d in sc]
        dim_stats[d] = {
            "n_ratings": len(vals),
            "mean": round(statistics.mean(vals), 3) if vals else None,
            "sd": round(statistics.stdev(vals), 3) if len(vals) > 1 else None,
            "ci95": bootstrap_ci(vals),
        }

    def accept_rate(pred):
        vals = [v for a, vs in accepts.items() if pred(meta[a]) for v in vs]
        return {"n": len(vals),
                "accept_rate": round(sum(vals) / len(vals), 3) if vals else None}

    out = {
        "n_raters": len(data),
        "n_artefacts_rated": len(rated),
        "n_artefacts_in_sample": key["n_artefacts"],
        "pre_registered_hypothesis": key["pre_registered_hypothesis"],
        "primary_test": {
            "test": "Mann-Whitney U, one-sided (conforming > rejected)",
            "n_conforming": len(conforming),
            "n_rejected": len(rejected),
            "mean_conforming": round(statistics.mean(conforming), 3) if conforming else None,
            "mean_rejected": round(statistics.mean(rejected), 3) if rejected else None,
            "U": u, "z": round(z, 3) if z is not None else None,
            "p_one_sided": round(p, 5) if p is not None else None,
            "significant_at_0.05": bool(p is not None and p < 0.05),
        },
        "krippendorff_alpha_ordinal": {
            k: (round(v, 3) if v is not None else None) for k, v in reliability.items()
        },
        "per_dimension": dim_stats,
        "acceptance": {
            "overall": accept_rate(lambda m: True),
            "shacl_conforming": accept_rate(lambda m: m["shacl_conforms"]),
            "shacl_rejected": accept_rate(lambda m: not m["shacl_conforms"]),
            "exp2_risk_scenarios": accept_rate(lambda m: m["experiment"] == "exp2"),
            "exp3_control_recommendations": accept_rate(lambda m: m["experiment"] == "exp3"),
        },
        "free_text_comments": comments,
        "power_note":
            f"With {len(conforming)} conforming and {len(rejected)} rejected artefacts, "
            "a one-sided Mann-Whitney U at alpha=0.05 has roughly 80% power against a "
            "large shift (Cohen's d near 0.9) and is underpowered against small ones. "
            "A null result bounds the effect rather than establishing orthogonality.",
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    pt = out["primary_test"]
    print(f"raters={out['n_raters']} artefacts rated={out['n_artefacts_rated']}"
          f"/{out['n_artefacts_in_sample']}")
    print(f"conforming mean={pt['mean_conforming']} (n={pt['n_conforming']})  "
          f"rejected mean={pt['mean_rejected']} (n={pt['n_rejected']})")
    print(f"Mann-Whitney U={pt['U']} z={pt['z']} p(one-sided)={pt['p_one_sided']}")
    print(f"Krippendorff alpha: {out['krippendorff_alpha_ordinal']}")
    print(f"acceptance: {out['acceptance']['overall']}")
    print(f"  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
