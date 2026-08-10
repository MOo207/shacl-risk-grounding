"""Comprehensive NFCRM-1:2025 §6.9 risk-assessment ablation (CIC-IDS2018, Haiku).

Unlike the flat (attack-class x criticality) lookup used in run_unified_ablation.py,
this runner feeds the LLM the *full* risk-assessment process variables (asset, threat,
vulnerability/CVE+KEV, existing controls, flow behaviour) and asks it to perform an
explicit NFCRM §6.9 process: assess LIKELIHOOD, assess IMPACT, derive risk via the
L x I matrix, identify control gap / residual risk, and recommend a control.

Scoring (per user decision "Both"):
  - PRIMARY: stated risk_level vs the §6.9 (class x criticality) ground truth -> comparable
    to the existing ablation.
  - PROCESS-VAR DELTA: map the LLM's (likelihood x impact) through a fixed risk matrix to an
    "implied risk", and log where that diverges from both the lookup GT and the stated level.

Replicability: model=claude-haiku-4-5 (alias 'haiku'), fixed max_tokens, ONE unified system
prompt + template for all six classes, canonical N=60 (first 10 per class = ua_000..ua_063),
seed-free selection (deterministic by class order). Temperature is CLI-managed (not settable);
documented as a known limitation. Config hash printed at run start for audit.

Usage:
  python -m scripts.run_cic_comprehensive_risk --arm pure_llm   --out pure_llm_comprehensive.jsonl
  python -m scripts.run_cic_comprehensive_risk --arm tristage_llm --out tristage_llm_comprehensive.jsonl
"""
from __future__ import annotations
import argparse, hashlib, json, re, time
from pathlib import Path
from typing import Any, Optional

from scripts.claude_cli_client import ClaudeCLIClient

ROOT = Path(__file__).resolve().parents[1]
TEST_SET = ROOT / "results/unified_ablation/test_set.json"
OUT_DIR = ROOT / "results/cic_comprehensive"
LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
LIK = ["Low", "Medium", "High", "Very High"]
IMP = ["Low", "Medium", "High", "Catastrophic"]

# Fixed Likelihood x Impact -> risk-level matrix (NFCRM §6.9-style 4x4 -> 5-band).
RISK_MATRIX = {
    ("Low",       "Low"): "Very Low", ("Low",       "Medium"): "Low",    ("Low",       "High"): "Medium",       ("Low",       "Catastrophic"): "High",
    ("Medium",    "Low"): "Low",      ("Medium",    "Medium"): "Medium",  ("Medium",    "High"): "High",         ("Medium",    "Catastrophic"): "Catastrophic",
    ("High",      "Low"): "Medium",   ("High",      "Medium"): "High",    ("High",      "High"): "Catastrophic",  ("High",      "Catastrophic"): "Catastrophic",
    ("Very High", "Low"): "Medium",   ("Very High", "Medium"): "High",    ("Very High", "High"): "Catastrophic",  ("Very High", "Catastrophic"): "Catastrophic",
}

SYSTEM_PROMPT = """You are a GRC risk analyst performing a NFCRM-1:2025 §6.9 technical asset risk assessment.
Assess risk as LIKELIHOOD x IMPACT using ALL provided process variables — do not shortcut to a single lookup.

LIKELIHOOD (threat exploitation), scale {Low, Medium, High, Very High}. Drivers:
  - CVE CVSS v3 base score (>=9.0 critical, 7.0-8.9 high, 4.0-6.9 medium, <4 low)
  - Known-Exploited (KEV) status: a paired CVE listed in CISA KEV is actively exploited -> raise likelihood
  - Attack-class exploitability: DoS/DDoS/BruteForce/WebAttack are commodity (higher base rate); Infiltration is stealthy; Benign implies no threat
  - Exposure: internet-facing / high traffic flows raise likelihood

IMPACT (consequence to the asset), scale {Low, Medium, High, Catastrophic}. Drivers:
  - Asset criticality {Low, Medium, High, Critical}
  - Asset type / OS (servers, domain controllers, databases carry higher impact)
  - Attack consequence (data exfiltration / C2 = high; volumetric DoS = availability)

RISK = LIKELIHOOD x IMPACT via this matrix:
              Impact:   Low        Medium     High          Catastrophic
  Likelihood Low        Very Low   Low        Medium        High
  Likelihood Medium     Low        Medium     High          Catastrophic
  Likelihood High       Medium     High       Catastrophic  Catastrophic
  Likelihood Very High  Medium     High       Catastrophic  Catastrophic

Benign traffic with no paired CVE = Likelihood Low, Impact Low -> Very Low (do not over-escalate benign flows).

Also assess the CONTROL GAP given the existing controls, and the RESIDUAL RISK after those controls.
Output ONLY a valid JSON object inside a ```json fenced code block."""

TRISTAGE_RULE = """\nVERIFIER-CORRECTOR (tri-stage only): if the ML classifier predicts Benign with confidence < 80% AND the paired CVE has CVSS >= 9.5 AND asset criticality is High/Critical, treat the event as Infiltration (stealthy activity missed by ML); otherwise trust the ML attack class."""

TEMPLATE = """NFCRM §6.9 RISK ASSESSMENT — process variables:

[ASSET]
- hostname: {hostname}
- criticality: {criticality}
- asset_type: {asset_type}
- os: {os}

[THREAT]
- attack_class: {attack_class}
- ML classifier confidence: {confidence}
- flow behaviour: {nl_desc}

[VULNERABILITY]
{cve_block}

[EXISTING CONTROLS]
- {controls}

Perform the §6.9 process:
1. LIKELIHOOD (Low/Medium/High/Very High) with one-line justification from the drivers above.
2. IMPACT (Low/Medium/High/Catastrophic) with one-line justification.
3. RISK_LEVEL = Likelihood x Impact via the matrix (Very Low..Catastrophic).
4. CONTROL_GAP: what is missing given existing controls. RESIDUAL_RISK after controls.
5. Recommend one NFCRM-1:2025 control and write a 2-3 sentence audit narrative citing the CVE (if any), asset criticality, and the likelihood/impact rationale.

```json
{{
  "likelihood": "Low|Medium|High|Very High",
  "impact": "Low|Medium|High|Catastrophic",
  "risk_level": "Very Low|Low|Medium|High|Catastrophic",
  "verified_attack_class": "<class used after any override>",
  "control_gap": "<short>",
  "residual_risk": "Low|Medium|High|Catastrophic",
  "recommended_control_id": "<NFCRM control>",
  "nfcrm_clauses": ["§6.9"],
  "narrative": "<2-3 sentences>",
  "evidence_refs": ["<cve_id if any>"]
}}```"""

JSON_FENCE = re.compile(r"```json\s*\n?(.*?)```", re.DOTALL)


def cve_block(cve: Optional[dict]) -> str:
    if not cve:
        return "- No paired CVE (no known vulnerability context; not KEV-listed)."
    return (f"- CVE: {cve.get('cve_id') or cve.get('id')}\n"
            f"- CVSS v3 base: {cve.get('cvss_v3')}\n"
            f"- KEV (known-exploited): yes — listed in CISA KEV\n"
            f"- KEV description: {(cve.get('shortDescription') or '')[:160]}\n"
            f"- Required action: {(cve.get('requiredAction') or '')[:120]}")


def build_prompt(case: dict, arm: str) -> str:
    asset = case.get("asset", {})
    if arm == "tristage_llm":
        attack_class = case.get("xgb_predicted_class", "Unknown")
        conf = f"{case.get('xgb_confidence', 1.0):.0%}"
    else:  # pure_llm — class unknown
        attack_class = "UNKNOWN — infer from CVE + flow behaviour"
        conf = "n/a"
    return TEMPLATE.format(
        hostname=asset.get("hostname", "?"),
        criticality=case.get("criticality", asset.get("criticality", "Medium")),
        asset_type=asset.get("asset_type", "?"),
        os=asset.get("os", "?"),
        attack_class=attack_class,
        confidence=conf,
        nl_desc=case.get("nl_description", "")[:300],
        cve_block=cve_block(case.get("paired_cve")),
        controls=", ".join(case.get("controls", [])) or "none recorded",
    )


def parse(resp: str) -> Optional[dict]:
    m = JSON_FENCE.search(resp)
    blob = m.group(1) if m else resp
    try:
        return json.loads(blob)
    except Exception:
        f, l = blob.find("{"), blob.rfind("}")
        if f >= 0 and l > f:
            try:
                return json.loads(blob[f:l + 1])
            except Exception:
                return None
    return None


def implied_risk(lik: str, imp: str) -> str:
    return RISK_MATRIX.get((lik, imp), "")


def canonical_cases(all_cases: list, n_per_class: int = 10) -> list:
    from collections import defaultdict
    by = defaultdict(list)
    for c in all_cases:
        by[c["true_attack_class"]].append(c)
    out = []
    for cls in sorted(by):
        out.extend(by[cls][:n_per_class])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["pure_llm", "tristage_llm"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--n-per-class", type=int, default=10)
    args = ap.parse_args()

    all_cases = json.loads(TEST_SET.read_text())["cases"]
    cases = canonical_cases(all_cases, args.n_per_class)
    system = SYSTEM_PROMPT + (TRISTAGE_RULE if args.arm == "tristage_llm" else "")
    cfg = {"arm": args.arm, "model": args.model, "max_tokens": args.max_tokens,
           "n": len(cases), "n_per_class": args.n_per_class,
           "system_prompt_sha1": hashlib.sha1(system.encode()).hexdigest()[:12],
           "template_sha1": hashlib.sha1(TEMPLATE.encode()).hexdigest()[:12]}
    print("CONFIG:", json.dumps(cfg))
    client = ClaudeCLIClient(model=args.model, max_tokens=args.max_tokens, timeout_sec=180)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / args.out
    with out_path.open("w", encoding="utf-8") as f:
        for i, case in enumerate(cases, 1):
            t0 = time.time()
            art = parse(client.generate(system, build_prompt(case, args.arm)) or "")
            gt = case["ground_truth_risk_level"]
            if art:
                rl = art.get("risk_level", "")
                lik, imp = art.get("likelihood", ""), art.get("impact", "")
                impl = implied_risk(lik, imp)
                rec = {
                    "case_id": case["case_id"], "arm": args.arm, "model": args.model,
                    "true_class": case["true_attack_class"],
                    "ml_pred_class": case.get("xgb_predicted_class", ""),
                    "verified_attack_class": art.get("verified_attack_class", ""),
                    "likelihood": lik, "impact": imp,
                    "risk_level": rl, "implied_risk_LxI": impl,
                    "ground_truth": gt,
                    "correct": rl == gt,                          # PRIMARY vs §6.9
                    "implied_correct": impl == gt,                # L x I vs §6.9
                    "internal_consistent": (impl == rl) if impl else None,
                    "lookup_vs_process_diverges": (impl != "" and impl != gt),
                    "control_gap": art.get("control_gap", ""),
                    "residual_risk": art.get("residual_risk", ""),
                    "recommended_control_id": art.get("recommended_control_id", ""),
                    "narrative": art.get("narrative", ""),
                    "parse_error": False,
                }
            else:
                rec = {"case_id": case["case_id"], "arm": args.arm, "model": args.model,
                       "true_class": case["true_attack_class"], "risk_level": "",
                       "ground_truth": gt, "correct": False, "parse_error": True}
            rec["latency_ms"] = int((time.time() - t0) * 1000)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            mark = "OK" if rec.get("correct") else ("ERR" if rec.get("parse_error") else "MISS")
            print(f"  [{i}/{len(cases)}] {case['case_id']} ({case['true_attack_class']}): "
                  f"L={rec.get('likelihood','?')} I={rec.get('impact','?')} "
                  f"risk={rec.get('risk_level') or 'PARSE_ERR'} vs GT={gt} [{mark}] ({rec['latency_ms']}ms)")
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
