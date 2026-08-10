"""Tri-stage LLM (Haiku 4.5) rerun on the CIC-IDS2018 clean held-out sample.

Context: a prior audit (results/cic_unified_rerun/leakage_audit.json) found that
BOTH LLM-facing text channels in the original unified-ablation pipeline
(scripts/run_unified_ablation.py) leak the true attack-class label into the
prose the LLM reads:
  (a) the NL-description generator (scripts/_archive/run_slm_nl_classification.py
      :: features_to_nl()) templates the attack-category name directly into the
      prose for 5 of 6 classes (22/60 = 37% of the original N=60 subset), and
  (b) the paired CVE description text contains literal class-name tokens for
      DoS/BruteForce/Infiltration cases (21/60 = 35%).

This script:
  1. Reconstructs a LEAK-FREE NL description for each of the 120 held-out cases
     in results/cic_unified_rerun/test_sample.json, by re-deriving the same
     quantitative flow evidence (port, TCP window, packet/byte counts,
     duration) from the source CSV row but with every class-naming phrase
     stripped (`clean_features_to_nl()` below — mirrors features_to_nl()'s
     branch structure exactly, only the labeling clauses differ).
  2. Redacts literal class-naming tokens from the paired CVE description
     (`redact_cve_description()` below), replacing them with neutral
     technical descriptors while preserving the CVSS score and the rest of
     the vulnerability context.
  3. Runs the IDENTICAL tri-stage LLM pipeline (system prompt, user prompt
     template, §6.9 lookup table, override-verification rule with its fixed
     thresholds, JSON artifact schema) as scripts/run_unified_ablation.py's
     `tristage_llm_haiku` arm — importing those exact prompt strings and the
     override/lookup logic, not reimplementing them — against Claude Haiku
     4.5 (claude-haiku-4-5-20251001) via the Claude CLI subprocess wrapper.
     NO other model is called.
  4. Computes safety metrics (exact match, under-escalation, severe recall,
     worst miss), GRC-completeness (all 5 required artifact fields present),
     and a paired McNemar test against the clean XGBoost baseline already
     computed per-case in test_sample.json (xgb_risk_level).

Usage:
    python scripts/run_cic_tristage_haiku_clean.py                # full 120-case run
    python scripts/run_cic_tristage_haiku_clean.py --limit 5       # smoke test
    python scripts/run_cic_tristage_haiku_clean.py --start-offset 40  # resume
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.nfcrm.risk_score import compute_risk_score  # noqa: E402
from scripts.llm_artifact_lib import REQUIRED_FIELDS  # noqa: E402
from scripts.run_unified_ablation import (  # noqa: E402
    SYSTEM_PROMPT_TRISTAGE, PROMPT_TRISTAGE, ATTACK_TO_CONTROL,
    JSON_FENCE, rule_lookup,
)

DATA_PATH = ROOT / "data" / "processed" / "cleaned_dataset.csv"
TEST_SAMPLE_PATH = ROOT / "results" / "cic_unified_rerun" / "test_sample.json"
XGB_BASELINE_PATH = ROOT / "results" / "cic_unified_rerun" / "xgboost_baseline_results.json"
OUT_DIR = ROOT / "results" / "cic_unified_rerun"
OUT_JSONL = OUT_DIR / "tristage_haiku_clean_raw.jsonl"
OUT_SUMMARY = OUT_DIR / "tristage_haiku_clean_results.json"

MODEL = "claude-haiku-4-5-20251001"

LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
IDX = {l: i for i, l in enumerate(LEVELS)}
SEVERE = {"High", "Catastrophic"}

# Same leakage-detection regexes as scripts/build_cic_unified_rerun.py, used
# here to VERIFY the cleaned text no longer trips the original audit's
# triggers (self-check, not a new claim).
NL_NAMED_PATTERNS = [
    (re.compile(r"denial-of-service \(DoS\)", re.I), "DoS"),
    (re.compile(r"distributed denial-of-service \(DDoS\)", re.I), "DDoS"),
    (re.compile(r"brute-force", re.I), "BruteForce"),
    (re.compile(r"web attack", re.I), "WebAttack"),
    (re.compile(r"infiltration", re.I), "Infiltration"),
]
CVE_NAMED_PATTERNS = [
    (re.compile(r"\bDoS\b", re.I), "DoS"),
    (re.compile(r"DDoS|distributed denial", re.I), "DDoS"),
    (re.compile(r"brute-force", re.I), "BruteForce"),
    (re.compile(r"\bweb attack\b", re.I), "WebAttack"),
    (re.compile(r"backdoor|infiltrat", re.I), "Infiltration"),
]


# ── 1. Leak-free NL description (mirrors features_to_nl() branch structure) ──

def describe_port(port: int) -> str:
    PORT_NAMES = {
        21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
        80: "HTTP (web)", 110: "POP3", 143: "IMAP", 443: "HTTPS", 445: "SMB",
        3306: "MySQL", 3389: "RDP", 5900: "VNC", 8080: "HTTP-alt",
    }
    return PORT_NAMES.get(int(port), f"port {int(port)}")


def clean_features_to_nl(row: pd.Series) -> str:
    """Leak-free counterpart to _archive/run_slm_nl_classification.py::features_to_nl().

    Same five signature branches + fallback, same quantitative evidence
    (ports, TCP window, packet/byte counts, duration, rates) — but every
    clause that names the attack category/class (DoS, DDoS, brute-force,
    web attack, infiltration, C2, backdoor, Benign) has been removed or
    replaced with a neutral, purely descriptive phrase.
    """
    lines = []

    dst_port   = int(row.get("Dst Port", 0))
    fwd_pkts   = int(row.get("Tot Fwd Pkts", 0))
    bwd_pkts   = int(row.get("Tot Bwd Pkts", 0))
    fwd_bytes  = int(row.get("TotLen Fwd Pkts", 0))
    bwd_bytes  = int(row.get("TotLen Bwd Pkts", 0))
    dur_us     = float(row.get("Flow Duration", 0))
    dur_s      = dur_us / 1e6
    pps        = float(row.get("Flow Pkts/s", 0))
    win_fwd    = int(row.get("Init Fwd Win Byts", 0))
    pkt_mean   = float(row.get("Pkt Len Mean", 0))
    total_pkts = fwd_pkts + bwd_pkts
    total_bytes = fwd_bytes + bwd_bytes

    port_name = describe_port(dst_port) if dst_port > 0 else "unknown port"

    # ── 1. win=225, port 80, short ──
    if win_fwd == 225 and dst_port == 80:
        lines.append(
            f"HTTP flow to port 80 with TCP initial window of exactly 225 bytes. "
            f"Flow: {fwd_pkts} forward packet(s), {bwd_pkts} backward, duration {dur_s*1000:.1f} ms. "
            "Minimal payload, no server response observed."
        )
        return " ".join(lines)

    # ── 2. win=65535, port 80, short ──
    if win_fwd == 65535 and dst_port == 80 and dur_s < 0.1:
        lines.append(
            f"HTTP flow to port 80 with maximum TCP window (65535 bytes) and a very short "
            f"flow duration ({dur_s*1000:.1f} ms)."
        )
        lines.append(
            f"Packet exchange: {fwd_pkts} forward, {bwd_pkts} backward. "
            f"{'No server response recorded.' if bwd_pkts == 0 else 'Some server responses observed.'}"
        )
        return " ".join(lines)

    # ── 3. long HTTP session, many bidirectional pkts ──
    if dst_port == 80 and dur_s > 30 and fwd_pkts > 50 and bwd_pkts > 30:
        lines.append(
            f"Sustained HTTP session lasting {dur_s:.0f} seconds with {fwd_pkts} forward requests "
            f"and {bwd_pkts} server responses — an extended bidirectional exchange on a web server."
        )
        if fwd_bytes > 0 or bwd_bytes > 0:
            lines.append(
                f"Data transferred: {fwd_bytes/1024:.1f} KB sent, {bwd_bytes/1024:.1f} KB received "
                f"(avg packet size {pkt_mean:.0f} bytes)."
            )
        lines.append(f"TCP window: {win_fwd} bytes — full TCP handshake completed.")
        return " ".join(lines)

    # ── 4. medium-duration HTTP session, few packets, non-zero payload ──
    if (dst_port == 80 and 3 <= dur_s <= 10 and 2 <= fwd_pkts <= 15
            and 1 <= bwd_pkts <= 15 and (fwd_bytes > 100 or bwd_bytes > 100)):
        lines.append(
            f"HTTP session to port 80, duration {dur_s:.1f} seconds, {fwd_pkts} client requests "
            f"and {bwd_pkts} server responses — a slow, deliberate HTTP exchange."
        )
        if bwd_bytes > fwd_bytes and bwd_bytes > 200:
            lines.append(
                f"Server response ({bwd_bytes} bytes) exceeds client payload ({fwd_bytes} bytes)."
            )
        elif fwd_bytes > 400:
            lines.append(
                f"Client sent {fwd_bytes} bytes in {fwd_pkts} packets (avg {fwd_bytes//max(fwd_pkts,1)} B/pkt)."
            )
        lines.append(f"TCP window: {win_fwd} bytes, packet rate: {pps:.1f} pkt/s.")
        return " ".join(lines)

    # ── 5. non-standard port ──
    if dst_port not in {80, 443, 8080, 53, 22, 21, 3389, 3306} and dst_port > 1024:
        lines.append(
            f"Flow targeting non-standard {port_name} — outside the common service-port set "
            "(80/443/8080/53/22/21/3389/3306)."
        )
        if dur_s > 60:
            lines.append(f"Long-running session ({dur_s/60:.1f} min).")
        elif dur_s < 0.01:
            lines.append(f"Very brief contact ({dur_s*1000:.2f} ms).")
        lines.append(f"Packets: {fwd_pkts} forward, {bwd_pkts} backward. TCP window: {win_fwd}.")
        return " ".join(lines)

    # ── 6. fallback ──
    lines.append(f"Network flow targeting {port_name}.")
    if dur_s < 0.01:
        lines.append(f"Duration: {dur_s*1000:.2f} ms (very short).")
    elif dur_s < 60:
        lines.append(f"Duration: {dur_s:.1f}s.")
    else:
        lines.append(f"Duration: {dur_s/60:.1f} min.")
    lines.append(f"Packets: {fwd_pkts} forward, {bwd_pkts} backward.")
    if win_fwd >= 0:
        lines.append(f"TCP initial window: {win_fwd} bytes.")
    if total_bytes > 0:
        lines.append(f"Bytes: {fwd_bytes} sent, {bwd_bytes} received.")
    if pps > 100 and total_pkts > 5:
        lines.append(f"Packet rate: {pps:.0f} pkt/s.")
    if dst_port == 80 and dur_s < 0.1 and total_pkts <= 10:
        lines.append(
            "Short HTTP probe with a non-standard TCP window and minimal packet exchange."
        )
    if total_bytes == 0 and total_pkts <= 5 and dur_s < 10:
        lines.append(
            "No payload data and minimal packet exchange."
        )
    return " ".join(lines)


# ── 2. Leak-free CVE description ─────────────────────────────────────────────

_CVE_REDACTIONS = [
    (re.compile(r"distributed denial[- ]of[- ]service", re.I), "distributed network-availability"),
    (re.compile(r"\bDDoS\b", re.I), "availability-impacting"),
    (re.compile(r"denial[- ]of[- ]service", re.I), "availability-impacting"),
    (re.compile(r"\bDoS\b", re.I), "availability-impacting"),
    (re.compile(r"brute-force", re.I), "credential-guessing"),
    (re.compile(r"\bweb attack(s)?\b", re.I), "application-layer issue"),
    (re.compile(r"backdoor", re.I), "unauthorized-access"),
    (re.compile(r"infiltrat\w*", re.I), "unauthorized-access"),
]


def redact_cve_description(text: str) -> str:
    if not text:
        return text
    out = text
    for pat, repl in _CVE_REDACTIONS:
        out = pat.sub(repl, out)
    return out


def verify_leak_free(nl_text: str, cve_text: str, true_cls: str) -> list[str]:
    hits = []
    for pat, cls in NL_NAMED_PATTERNS:
        if cls == true_cls and pat.search(nl_text):
            hits.append(f"NL still matches {cls} pattern: {pat.pattern!r}")
    for pat, cls in CVE_NAMED_PATTERNS:
        if cls == true_cls and pat.search(cve_text):
            hits.append(f"CVE still matches {cls} pattern: {pat.pattern!r}")
    return hits


def _format_cve(cve: Optional[dict]) -> str:
    if not cve:
        return "No CVE context."
    desc = redact_cve_description(cve.get("shortDescription", cve.get("description", "")))
    return f"CVE: {cve.get('id')}  CVSS: {cve.get('cvss_v3')}  Desc: {desc[:120]}"


def _parse_response(text: str) -> Optional[dict]:
    m = JSON_FENCE.search(text)
    blob = m.group(1) if m else text
    try:
        return json.loads(blob)
    except Exception:
        f, l = blob.find("{"), blob.rfind("}")
        if f >= 0 and l > f:
            try:
                return json.loads(blob[f:l+1])
            except Exception:
                return None
    return None


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(records: list[dict]) -> dict:
    n = under = over = exact = within1 = sev_tot = sev_rec = worst = 0
    for r in records:
        p, g = r["risk_level"], r["ground_truth"]
        n += 1
        if p not in IDX or g not in IDX:
            under += 1
            continue
        d = IDX[p] - IDX[g]
        if d == 0:
            exact += 1
        if abs(d) <= 1:
            within1 += 1
        if d < 0:
            under += 1
            worst = max(worst, -d)
        elif d > 0:
            over += 1
        if g in SEVERE:
            sev_tot += 1
            if p in SEVERE:
                sev_rec += 1
    pct = lambda a, b: (a / b * 100) if b else 0.0
    return dict(n=n, exact_match_pct=pct(exact, n), within1_pct=pct(within1, n),
                under_escalation_pct=pct(under, n), over_escalation_pct=pct(over, n),
                severe_recall_pct=pct(sev_rec, sev_tot), severe_total=sev_tot,
                worst_miss_levels=worst)


def mcnemar(records: list[dict]) -> dict:
    """Paired McNemar (exact binomial on discordant pairs) tri-stage vs XGBoost,
    using each case's XGBoost risk_level (from test_sample.json) as the paired
    baseline prediction."""
    from scipy.stats import binomtest
    b = c = 0  # b: xgb correct, tri-stage wrong. c: xgb wrong, tri-stage correct.
    for r in records:
        xgb_correct = r["xgb_risk_level"] == r["ground_truth"]
        tri_correct = r["risk_level"] == r["ground_truth"]
        if xgb_correct and not tri_correct:
            b += 1
        elif tri_correct and not xgb_correct:
            c += 1
    n_disc = b + c
    if n_disc == 0:
        p = 1.0
    else:
        p = binomtest(min(b, c), n_disc, 0.5, alternative="two-sided").pvalue
    return {"b_xgb_correct_tri_wrong": b, "c_xgb_wrong_tri_correct": c,
            "n_discordant": n_disc, "p_value": p}


def grc_completeness(records: list[dict]) -> dict:
    """Field presence check. `evidence_refs`/`nfcrm_clauses` are lists that may
    legitimately be empty (e.g. Benign cases with no paired CVE) — presence
    means the key exists and is the right type, not that it is non-empty.
    Scalar fields (risk_level/recommended_control_id/narrative) must be
    non-empty strings."""
    list_fields = {"evidence_refs", "nfcrm_clauses"}
    complete = 0
    per_field_missing = {f: 0 for f in REQUIRED_FIELDS}
    for r in records:
        art = r.get("artifact") or {}
        ok = True
        for f in REQUIRED_FIELDS:
            v = art.get(f)
            if f in list_fields:
                present = isinstance(v, list)
            else:
                present = v not in (None, "")
            if not present:
                per_field_missing[f] += 1
                ok = False
        if ok:
            complete += 1
    n = len(records)
    return {
        "n": n, "complete": complete,
        "completeness_pct": (complete / n * 100) if n else 0.0,
        "per_field_missing_count": per_field_missing,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Limit to first N cases (smoke test)")
    ap.add_argument("--start-offset", type=int, default=0, help="Resume from this case index")
    ap.add_argument("--timeout-sec", type=int, default=180)
    args = ap.parse_args()

    test_out = json.loads(TEST_SAMPLE_PATH.read_text(encoding="utf-8"))
    cases = test_out["cases"]
    if args.limit:
        cases = cases[:args.limit]
    print(f"Loaded {len(cases)} cases from {TEST_SAMPLE_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    from scripts.claude_cli_client import ClaudeCLIClient
    client = ClaudeCLIClient(model=MODEL, max_tokens=600, timeout_sec=args.timeout_sec)

    mode = "a" if args.start_offset > 0 else "w"
    leak_check_failures = []
    api_errors = 0
    parse_failures = 0
    retried_cases = 0

    with OUT_JSONL.open(mode, encoding="utf-8") as f:
        for i, case in enumerate(cases[args.start_offset:], start=args.start_offset + 1):
            t0 = time.time()
            cid = case["case_id"]
            crit = case["criticality"]
            gt = case["ground_truth_risk_level"]
            true_cls = case["true_attack_class"]
            cve = case.get("paired_cve")

            row = df.loc[case["source_row_index"]]
            nl_desc_clean = clean_features_to_nl(row)
            cve_blk_clean = _format_cve(cve)

            leak_hits = verify_leak_free(nl_desc_clean, cve_blk_clean, true_cls)
            if leak_hits:
                leak_check_failures.append({"case_id": cid, "hits": leak_hits})

            ml_cls = case["xgb_predicted_class"]
            ml_conf = case.get("xgb_confidence", 1.0)
            asset_j = json.dumps(case.get("asset", {}))

            prompt = PROMPT_TRISTAGE.format(
                attack_class=ml_cls, confidence=f"{ml_conf:.0%}",
                criticality=crit, asset_json=asset_j, cve_block=cve_blk_clean,
                nl_desc=nl_desc_clean[:300],
            )

            try:
                response = client.generate(SYSTEM_PROMPT_TRISTAGE, prompt) or ""
            except Exception:
                response = ""
            if not response:
                api_errors += 1

            artifact = _parse_response(response)
            latency_ms = int((time.time() - t0) * 1000)

            if artifact:
                verified_cls = artifact.get("verified_attack_class", "")
                if verified_cls and verified_cls in ATTACK_TO_CONTROL and verified_cls != ml_cls:
                    risk_lvl = rule_lookup(verified_cls, crit)
                else:
                    risk_lvl = artifact.get("risk_level", "")
                parse_ok = True
            else:
                parse_failures += 1
                # Same fallback convention as the original tri-stage arm: trust ML.
                risk_lvl = case.get("xgb_risk_level", "")
                verified_cls = ml_cls
                parse_ok = False

            record = {
                "case_id": cid,
                "model": MODEL,
                "true_class": true_cls,
                "ml_pred_class": ml_cls,
                "llm_verified_cls": verified_cls,
                "risk_level": risk_lvl,
                "xgb_risk_level": case.get("xgb_risk_level", ""),
                "ground_truth": gt,
                "correct": risk_lvl == gt,
                "xgb_correct": case.get("xgb_risk_level", "") == gt,
                "parse_error": not parse_ok,
                "criticality": crit,
                "nl_description_clean": nl_desc_clean,
                "cve_block_clean": cve_blk_clean,
                "leak_check_hits": leak_hits,
                "artifact": artifact,
                "raw_response": response[:2000],
                "latency_ms": latency_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            match = "OK" if record["correct"] else "MISS"
            print(f"  [{i}/{len(cases)}] {cid}: {risk_lvl or 'PARSE_ERR'} vs GT={gt} [{match}] ({latency_ms}ms)")

    # ── Aggregate ──
    all_records = [json.loads(l) for l in OUT_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    metrics = compute_metrics(all_records)
    mcn = mcnemar(all_records)
    grc = grc_completeness(all_records)

    xgb_baseline = json.loads(XGB_BASELINE_PATH.read_text(encoding="utf-8"))

    summary = {
        "meta": {
            "description": (
                "Tri-stage LLM (Claude Haiku 4.5, CLI) rerun on the CIC-IDS2018 "
                "clean held-out sample (results/cic_unified_rerun/test_sample.json, "
                "N=120, 20/class), using leak-free NL descriptions and redacted CVE "
                "text (this script's clean_features_to_nl() / redact_cve_description()) "
                "in place of the original leaky scripts/_archive/run_slm_nl_classification.py "
                "output. Prompt template, override-verification thresholds, and §6.9 "
                "lookup logic are IDENTICAL to scripts/run_unified_ablation.py's "
                "tristage_llm_haiku arm (imported directly, not reimplemented)."
            ),
            "model": MODEL,
            "n_cases": len(all_records),
            "api_errors": api_errors,
            "parse_failures": parse_failures,
            "leak_check_failures": leak_check_failures,
            "n_leak_check_failures": len(leak_check_failures),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "metrics": metrics,
        "mcnemar_vs_clean_xgboost": mcn,
        "grc_completeness": grc,
        "clean_xgboost_baseline_reference": xgb_baseline["held_out_test_sample_metrics"],
        "records": all_records,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps({"metrics": metrics, "mcnemar": mcn, "grc": {k: v for k, v in grc.items() if k != "per_field_missing_count"},
                       "api_errors": api_errors, "parse_failures": parse_failures,
                       "leak_check_failures": len(leak_check_failures)}, indent=2))
    print(f"\nSaved: {OUT_JSONL}\nSaved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
