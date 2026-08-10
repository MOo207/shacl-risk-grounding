"""
Layer 4 Quality Evaluation — N=50
==================================
Evaluates Claude Haiku 4.5 GRC narrative generation quality across 50 flows
(10 per attack class × 5 classes), computing:

  - parse_success_rate       : % of calls returning non-empty text
  - cve_mention_rate         : % of narratives containing a CVE reference
  - nfcrm_mention_rate       : % of narratives citing an NFCRM clause
  - avg_word_count           : mean narrative length (words)
  - per_class_parse_rate     : parse success broken down by attack class
  - per_class_cve_rate       : CVE mention rate by class
  - wilson_ci_parse          : Wilson-score 95% CI on parse rate
  - wilson_ci_cve            : Wilson-score 95% CI on CVE mention rate

Saves: results/layer4_evaluation.json

Usage:
    python scripts/run_layer4_evaluation.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# ---------------------------------------------------------------------------
# Reuse GRC KB + layer helpers from main explanation runner
# ---------------------------------------------------------------------------
GRC_KB = {
    "BruteForce": {
        "cve_examples": ["CVE-2023-23397", "CVE-2022-26134"],
        "nfcrm_clause": "NFCRM-1:2025 §6.7 (Access Control Enforcement)",
        "risk_level": "HIGH",
        "control": "Account lockout + MFA enforcement",
        "nca_ref": "NCA-ECC-2-3-1",
    },
    "DoS": {
        "cve_examples": ["CVE-2023-44487", "CVE-2022-3602"],
        "nfcrm_clause": "NFCRM-1:2025 §6.9 (Inherent Risk Analysis)",
        "risk_level": "HIGH",
        "control": "Rate limiting + traffic scrubbing",
        "nca_ref": "NCA-ECC-3-1-2",
    },
    "DDoS": {
        "cve_examples": ["CVE-2023-44487", "CVE-2022-26143"],
        "nfcrm_clause": "NFCRM-1:2025 §6.9 (Inherent Risk Analysis)",
        "risk_level": "CRITICAL",
        "control": "DDoS mitigation service + CDN scrubbing",
        "nca_ref": "NCA-ECC-3-1-2",
    },
    "WebAttack": {
        "cve_examples": ["CVE-2023-34362", "CVE-2021-44228"],
        "nfcrm_clause": "NFCRM-1:2025 §6.4 (Vulnerability Inventory)",
        "risk_level": "HIGH",
        "control": "WAF + input validation + parameterised queries",
        "nca_ref": "NCA-ECC-4-2-1",
    },
    "Infiltration": {
        "cve_examples": ["CVE-2023-20198", "CVE-2022-30190"],
        "nfcrm_clause": "NFCRM-1:2025 §6.5 (Threat Detection)",
        "risk_level": "CRITICAL",
        "control": "Network segmentation + EDR + privileged access management",
        "nca_ref": "NCA-ECC-5-1-3",
    },
}

PRE_RULES = [
    ("high_volume_flow",    lambda r: r.get("Total Fwd Packets", 0) > 1000),
    ("short_duration",      lambda r: r.get("Flow Duration", 0) < 10),
    ("suspicious_iat",      lambda r: r.get("Flow IAT Std", 999) < 5),
    ("syn_flood_indicator", lambda r: r.get("SYN Flag Count", 0) > 50),
    ("large_data_transfer", lambda r: r.get("Total Length of Fwd Packets", 0) > 100000),
    ("high_pps",            lambda r: r.get("Flow Packets/s", 0) > 100),
]

CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
NFCRM_PATTERN = re.compile(r"NFCRM|§6\.\d+|section 6\.\d+|6\.\d+ \(", re.IGNORECASE)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson-score 95% CI for k successes in n trials."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, round(centre - half, 3)), min(1.0, round(centre + half, 3)))


def layer1_context(row: pd.Series) -> dict:
    fired = {}
    for name, fn in PRE_RULES:
        try:
            fired[name] = bool(fn(row))
        except Exception:
            fired[name] = False
    active = [k for k, v in fired.items() if v]
    return {"active_rules": active, "symbolic_hint": ("possible_attack" if active else "benign_pattern")}


def layer3_grc_trace(label: str) -> dict:
    kb = GRC_KB.get(label, GRC_KB["DDoS"])
    return {
        "predicted_attack_type": label,
        "cve_examples": kb["cve_examples"],
        "risk_level": kb["risk_level"],
        "nfcrm_clause": kb["nfcrm_clause"],
        "recommended_control": kb["control"],
        "nca_reference": kb["nca_ref"],
        "risk_case": f"RC-{label.upper()}-{kb['risk_level']}",
    }


def call_layer4(l1: dict, l3: dict, client, dry_run: bool) -> dict:
    if dry_run:
        # Simulate a real-looking response for testing
        label = l3["predicted_attack_type"]
        cves = " and ".join(l3["cve_examples"][:2])
        clause = l3["nfcrm_clause"]
        return {
            "status": "OK",
            "backend": "dry_run",
            "nl_explanation": (
                f"A {label} attack has been detected with {l3['risk_level']} risk classification "
                f"under {clause}. The incident involves known vulnerability vectors including {cves}. "
                f"Immediate remediation via {l3['recommended_control']} is recommended per {l3['nca_reference']}."
            ),
        }

    system_msg = (
        "You are a cybersecurity GRC expert. Write exactly 3 sentences for a compliance officer. "
        "Be specific: cite the NFCRM clause and at least one CVE identifier. "
        "Be concise and non-technical. Do not use bullet points."
    )
    user_msg = (
        f"Pre-inference symbolic context (Layer 1): {l1['active_rules'] or 'no rules fired'}\n"
        f"Attack type detected: {l3['predicted_attack_type']}\n"
        f"Risk level: {l3['risk_level']}\n"
        f"Regulatory mapping: {l3['nfcrm_clause']}\n"
        f"CVE examples: {', '.join(l3['cve_examples'])}\n"
        f"Recommended control: {l3['recommended_control']}\n\n"
        "Write a 3-sentence audit-ready narrative for a compliance officer. "
        "The narrative MUST include the NFCRM clause reference and at least one CVE identifier."
    )
    try:
        nl_text = client.generate(system_msg, user_msg)
        if nl_text:
            return {"status": "OK", "backend": "claude_api_haiku-4.5", "nl_explanation": nl_text.strip()}
        return {"status": "EMPTY", "backend": "claude_api_haiku-4.5", "nl_explanation": None}
    except Exception as e:
        return {"status": "ERROR", "backend": "claude_api_haiku-4.5", "nl_explanation": None, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip API calls, use simulated responses")
    ap.add_argument("--n-per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="data/processed/cleaned_dataset.csv")
    args = ap.parse_args()

    N_PER_CLASS = args.n_per_class
    SEED = args.seed
    TARGET_CLASSES = ["DDoS", "DoS", "BruteForce", "WebAttack", "Infiltration"]

    print(f"[L4-eval] Loading dataset: {args.data}")
    df = pd.read_csv(ROOT / args.data, low_memory=False)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    print(f"[L4-eval] Dataset shape: {df.shape}, classes: {df['Label'].value_counts().to_dict()}")

    # Sample N_PER_CLASS per class
    samples = []
    for cls in TARGET_CLASSES:
        subset = df[df["Label"] == cls]
        n_avail = len(subset)
        n_take = min(N_PER_CLASS, n_avail)
        if n_take == 0:
            print(f"  [warn] No samples for class {cls}")
            continue
        sampled = subset.sample(n=n_take, random_state=SEED)
        for _, row in sampled.iterrows():
            samples.append((cls, row))
        print(f"  [ok] Sampled {n_take}/{n_avail} flows for {cls}")

    total = len(samples)
    print(f"[L4-eval] Total flows to evaluate: {total}")

    # Init client
    client = None
    if not args.dry_run:
        try:
            from claude_cli_client import ClaudeCLIClient
            client = ClaudeCLIClient(model="haiku", timeout_sec=120)
        except ImportError:
            print("[error] claude_cli_client not found — use --dry-run or fix import")
            sys.exit(1)

    results = []
    t_start = time.time()
    for i, (cls, row) in enumerate(samples):
        l1 = layer1_context(row)
        l3 = layer3_grc_trace(cls)
        l4 = call_layer4(l1, l3, client, args.dry_run)

        nl = l4.get("nl_explanation") or ""
        has_cve = bool(CVE_PATTERN.search(nl))
        has_nfcrm = bool(NFCRM_PATTERN.search(nl))
        word_count = len(nl.split()) if nl else 0

        rec = {
            "flow_idx": i + 1,
            "attack_class": cls,
            "l4_status": l4["status"],
            "nl_explanation": nl,
            "parse_success": l4["status"] == "OK",
            "cve_mentioned": has_cve,
            "nfcrm_mentioned": has_nfcrm,
            "word_count": word_count,
            "cves_found": CVE_PATTERN.findall(nl),
            "active_rules": l1["active_rules"],
        }
        results.append(rec)

        elapsed = time.time() - t_start
        eta = (elapsed / (i + 1)) * (total - i - 1)
        print(f"  [{i+1:3d}/{total}] {cls:15s} | status={l4['status']:5s} | "
              f"CVE={'Y' if has_cve else 'N'} | NFCRM={'Y' if has_nfcrm else 'N'} | "
              f"words={word_count:3d} | ETA {eta:.0f}s")

    # Compute aggregate metrics
    n = len(results)
    n_parse = sum(r["parse_success"] for r in results)
    n_cve = sum(r["cve_mentioned"] for r in results)
    n_nfcrm = sum(r["nfcrm_mentioned"] for r in results)
    words = [r["word_count"] for r in results if r["parse_success"]]

    parse_ci = wilson_ci(n_parse, n)
    cve_ci = wilson_ci(n_cve, n)
    nfcrm_ci = wilson_ci(n_nfcrm, n)

    # Per-class breakdown
    class_stats = {}
    for cls in TARGET_CLASSES:
        cls_rows = [r for r in results if r["attack_class"] == cls]
        nc = len(cls_rows)
        np_ = sum(r["parse_success"] for r in cls_rows)
        nc_ = sum(r["cve_mentioned"] for r in cls_rows)
        nn_ = sum(r["nfcrm_mentioned"] for r in cls_rows)
        class_stats[cls] = {
            "n": nc,
            "parse_rate": round(np_ / nc, 3) if nc else 0,
            "cve_rate": round(nc_ / nc, 3) if nc else 0,
            "nfcrm_rate": round(nn_ / nc, 3) if nc else 0,
            "avg_words": round(sum(r["word_count"] for r in cls_rows if r["parse_success"]) /
                               max(1, np_), 1),
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "claude_api_haiku-4.5" if not args.dry_run else "dry_run",
        "n_total": n,
        "n_per_class": N_PER_CLASS,
        "seed": SEED,
        "target_classes": TARGET_CLASSES,
        "aggregate": {
            "parse_success_rate": round(n_parse / n, 3),
            "parse_success_n": n_parse,
            "parse_success_ci_95": list(parse_ci),
            "cve_mention_rate": round(n_cve / n, 3),
            "cve_mention_n": n_cve,
            "cve_mention_ci_95": list(cve_ci),
            "nfcrm_mention_rate": round(n_nfcrm / n, 3),
            "nfcrm_mention_n": n_nfcrm,
            "nfcrm_mention_ci_95": list(nfcrm_ci),
            "avg_word_count": round(float(np.mean(words)), 1) if words else 0,
            "std_word_count": round(float(np.std(words)), 1) if words else 0,
        },
        "per_class": class_stats,
        "raw_results": results,
    }

    out_path = ROOT / "results" / "layer4_evaluation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[L4-eval] Saved -> {out_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("LAYER 4 EVALUATION SUMMARY")
    print("=" * 60)
    ag = summary["aggregate"]
    print(f"  N total          : {n}")
    print(f"  Parse success    : {n_parse}/{n} = {ag['parse_success_rate']*100:.1f}%  "
          f"95% CI [{ag['parse_success_ci_95'][0]*100:.0f}%, {ag['parse_success_ci_95'][1]*100:.0f}%]")
    print(f"  CVE mention      : {n_cve}/{n} = {ag['cve_mention_rate']*100:.1f}%  "
          f"95% CI [{ag['cve_mention_ci_95'][0]*100:.0f}%, {ag['cve_mention_ci_95'][1]*100:.0f}%]")
    print(f"  NFCRM mention    : {n_nfcrm}/{n} = {ag['nfcrm_mention_rate']*100:.1f}%  "
          f"95% CI [{ag['nfcrm_mention_ci_95'][0]*100:.0f}%, {ag['nfcrm_mention_ci_95'][1]*100:.0f}%]")
    print(f"  Avg word count   : {ag['avg_word_count']} ± {ag['std_word_count']}")
    print()
    print(f"  {'Class':<15} {'N':>4} {'Parse':>7} {'CVE':>7} {'NFCRM':>7} {'Words':>7}")
    print(f"  {'-'*15} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for cls, s in class_stats.items():
        print(f"  {cls:<15} {s['n']:>4} {s['parse_rate']*100:>6.0f}% "
              f"{s['cve_rate']*100:>6.0f}% {s['nfcrm_rate']*100:>6.0f}% {s['avg_words']:>7.1f}")
    print("=" * 60)
    total_elapsed = time.time() - t_start
    print(f"\n  Done in {total_elapsed:.0f}s ({total_elapsed/n:.1f}s/call)")


if __name__ == "__main__":
    main()
