"""Validate the asset-specific §6.9 risk model (Path B).

Three validations:

  (1) DISCRIMINATION (offline, always runs)
      The attack-type-constant model assigns one risk level per attack class.
      Asset-specific scoring should assign multiple distinct levels across the
      asset population. We report distinct risk levels per class and a
      Kruskal-Wallis-free spread summary.

  (2) RESIDUAL EFFECT (offline, always runs)
      §6.7 currently-applied controls should move a measurable fraction of
      scenarios to a lower residual band (§6.12).

  (3) EXTERNAL LIKELIHOOD VALIDATION (network, opt-in via --epss)
      The likelihood exploitability band is built from CVSS + CISA-KEV ONLY.
      EPSS (FIRST exploit-prediction probability) is never used in the band, so
      it is an independent ground truth. We fetch EPSS once (cached to
      results/epss_cache.json) and report Spearman rank correlation between the
      per-CVE exploitability band and EPSS. This is the non-circular test the
      thesis cites; it requires one read-only GET to api.first.org.

Usage:
    python scripts/validate_asset_risk.py            # offline validations only
    python scripts/validate_asset_risk.py --epss     # also fetch EPSS + Spearman
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.nfcrm import (  # noqa: E402
    cvss_kev_to_exploitability,
    load_arg,
    score_arg,
)

ARG_PATH = REPO / "results" / "multisource_arg.json"
EPSS_CACHE = REPO / "results" / "epss_cache.json"
OUT_JSON = REPO / "results" / "asset_risk_validation.json"
EPSS_API = "https://api.first.org/data/v1/epss"


# ─────────────────────────────────────────────────────────────────────────────
# Stdlib Spearman (no scipy dependency)
# ─────────────────────────────────────────────────────────────────────────────
def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average rank for ties (1-based)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    vb = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def spearman_permutation_p(a: list[float], b: list[float], *, n_perm: int = 20000,
                           seed: int = 42) -> float:
    """Two-sided permutation p-value for Spearman rho (stdlib only, reproducible).

    H0: no monotonic association between a and b. We shuffle b n_perm times and
    count how often |rho_perm| >= |rho_observed|. Seeded for reproducibility.
    """
    import random
    rho_obs = abs(spearman(a, b))
    rng = random.Random(seed)
    b2 = list(b)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(b2)
        if abs(spearman(a, b2)) >= rho_obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)  # add-one (never reports p=0)


def spearman_bootstrap_ci(a: list[float], b: list[float], *, n_boot: int = 10000,
                          seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for Spearman rho (stdlib only, reproducible)."""
    import random
    rng = random.Random(seed)
    n = len(a)
    rhos = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        rhos.append(spearman([a[i] for i in idx], [b[i] for i in idx]))
    rhos.sort()
    lo = rhos[int((alpha / 2) * n_boot)]
    hi = rhos[int((1 - alpha / 2) * n_boot)]
    return round(lo, 3), round(hi, 3)


# ─────────────────────────────────────────────────────────────────────────────
# (1) + (2) offline validations
# ─────────────────────────────────────────────────────────────────────────────
def validate_discrimination(rows: list[dict]) -> dict:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["attack_type"]].append(r)
    per_class = {}
    total_distinct_levels = 0
    for atk, rs in by_class.items():
        levels = sorted({r["inherent_risk_level"] for r in rs})
        scores = [r["inherent_risk_score"] for r in rs]
        per_class[atk] = {
            "n_assets": len(rs),
            "distinct_levels": levels,
            "n_distinct_levels": len(levels),
            "score_min": min(scores),
            "score_max": max(scores),
        }
        total_distinct_levels += len(levels)
    n_classes = len(by_class)
    return {
        "per_class": per_class,
        "n_attack_classes": n_classes,
        "constant_model_distinct_levels": n_classes,        # 1 per class by construction
        "asset_specific_distinct_levels": total_distinct_levels,
        "discrimination_gain": total_distinct_levels - n_classes,
    }


def validate_residual(rows: list[dict]) -> dict:
    reduced = [r for r in rows if r["residual_risk_score"] < r["inherent_risk_score"]]
    level_changes = [
        r for r in reduced if r["residual_risk_level"] != r["inherent_risk_level"]
    ]
    return {
        "n_scenarios": len(rows),
        "n_residual_reduced": len(reduced),
        "n_residual_level_changed": len(level_changes),
        "example_level_changes": [
            {
                "asset_id": r["asset_id"],
                "attack_type": r["attack_type"],
                "inherent": f"{r['inherent_risk_score']} ({r['inherent_risk_level']})",
                "residual": f"{r['residual_risk_score']} ({r['residual_risk_level']})",
                "n_controls": r["n_applied_controls"],
            }
            for r in level_changes[:5]
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# (3) external EPSS validation (opt-in)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_epss(cve_ids: list[str]) -> dict[str, float]:
    """Fetch EPSS probabilities once; cache to results/epss_cache.json.

    Read-only GET to api.first.org (no key). Returns {cve_id: epss_prob}.
    """
    if EPSS_CACHE.exists():
        cached = json.loads(EPSS_CACHE.read_text(encoding="utf-8"))
        if all(c in cached for c in cve_ids):
            return {c: cached[c] for c in cve_ids}

    import urllib.request  # stdlib, imported lazily so offline runs never touch it

    url = f"{EPSS_API}?cve={','.join(cve_ids)}"
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (trusted host)
        payload = json.loads(resp.read().decode("utf-8"))
    epss = {row["cve"]: float(row["epss"]) for row in payload.get("data", [])}
    EPSS_CACHE.write_text(json.dumps(epss, indent=2), encoding="utf-8")
    return epss


def validate_epss_external(arg: dict) -> dict:
    """Spearman(per-CVE exploitability band  vs  EPSS) — non-circular GT test."""
    cves = [n for n in arg["nodes"] if n.get("type") == "CVE"]
    cve_ids = [c["id"] for c in cves]
    epss = fetch_epss(cve_ids)

    paired_band, paired_epss, detail = [], [], []
    for c in cves:
        if c["id"] not in epss:
            continue
        band, _ = cvss_kev_to_exploitability(
            c.get("cvss_v3"), in_cisa_kev=bool(c.get("in_cisa_kev"))
        )
        if band == 0:
            continue
        paired_band.append(float(band))
        paired_epss.append(epss[c["id"]])
        detail.append({
            "cve": c["id"], "cvss": c.get("cvss_v3"),
            "kev": bool(c.get("in_cisa_kev")),
            "exploitability_band": band, "epss": epss[c["id"]],
        })

    rho = spearman(paired_band, paired_epss) if len(paired_band) >= 3 else None
    p_perm = (
        spearman_permutation_p(paired_band, paired_epss)
        if rho is not None else None
    )
    ci = (
        spearman_bootstrap_ci(paired_band, paired_epss)
        if rho is not None else None
    )
    return {
        "n_cves_paired": len(paired_band),
        "spearman_rho_band_vs_epss": round(rho, 3) if rho is not None else None,
        "spearman_p_permutation": round(p_perm, 4) if p_perm is not None else None,
        "spearman_rho_ci95": list(ci) if ci is not None else None,
        "n_permutations": 20000,
        "n_bootstrap": 10000,
        "seed": 42,
        "interpretation": (
            "EPSS is independent of the CVSS+KEV exploitability band, so a positive "
            "rho is non-circular evidence that the band tracks real-world "
            "exploitation probability."
        ),
        "detail": detail,
    }


def main() -> int:
    want_epss = "--epss" in sys.argv
    arg = load_arg(ARG_PATH)
    rows = [r.to_dict() for r in score_arg(arg)]

    report = {
        "discrimination": validate_discrimination(rows),
        "residual": validate_residual(rows),
        "epss_external": None,
    }

    d = report["discrimination"]
    print("=== (1) DISCRIMINATION ===")
    print(f"  attack classes: {d['n_attack_classes']}")
    print(f"  distinct risk levels — constant model: {d['constant_model_distinct_levels']} "
          f"| asset-specific: {d['asset_specific_distinct_levels']} "
          f"(gain +{d['discrimination_gain']})")
    for atk, s in sorted(d["per_class"].items()):
        print(f"    {atk:13s} n={s['n_assets']:2d}  levels={s['distinct_levels']}")

    r = report["residual"]
    print("\n=== (2) RESIDUAL EFFECT (§6.7 controls) ===")
    print(f"  residual < inherent: {r['n_residual_reduced']}/{r['n_scenarios']}; "
          f"crossed a band: {r['n_residual_level_changed']}")

    if want_epss:
        print("\n=== (3) EXTERNAL EPSS VALIDATION (network) ===")
        try:
            report["epss_external"] = validate_epss_external(arg)
            e = report["epss_external"]
            print(f"  paired CVEs: {e['n_cves_paired']}")
            print(f"  Spearman rho (CVSS+KEV band vs EPSS): "
                  f"{e['spearman_rho_band_vs_epss']} "
                  f"(p={e['spearman_p_permutation']}, "
                  f"95% CI {e['spearman_rho_ci95']})")
        except Exception as exc:  # noqa: BLE001 (report, do not crash offline parts)
            print(f"  EPSS fetch failed ({exc!r}); offline validations still valid.")
            report["epss_external"] = {"error": repr(exc)}
    else:
        print("\n=== (3) EXTERNAL EPSS VALIDATION ===")
        print("  skipped (offline). Re-run with --epss to fetch EPSS and compute "
              "the non-circular Spearman check.")

    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {OUT_JSON.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
