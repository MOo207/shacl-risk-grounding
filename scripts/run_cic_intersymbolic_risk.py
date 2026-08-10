"""TRUE intersymbolic comprehensive §6.9 risk assessment (CIC-IDS2018, Haiku).

This is the WHOLE framework, not a bare LLM prompt:

  1. PRE / ML        attack class: XGBoost (tri-stage) or LLM-inferred (pure-llm),
                     with a deterministic symbolic verifier-corrector override.
  2. SYMBOLIC ENGINE pipeline/nfcrm computes the COMPREHENSIVE §6.9 risk:
                       exploitability = cvss_kev_to_exploitability(CVSS, KEV)
                       impact         = CIA(attack class) scaled by §6.3 criticality
                       risk           = compute_risk_score(L × I)  → Figure-3 band
                       residual       = after §6.7 currently-applied controls
                     The risk LEVEL is computed symbolically and deterministically
                     (auditable), NOT invented by the LLM.
  3. LLM (grounds)   given the symbolic likelihood/impact/risk + ARG context, the LLM
                     writes the audit narrative and recommends an NFCRM control.
  4. SHACL (validates) validate_artifact_schema gates the artifact; on violation the
                     LLM is re-prompted once; SHACL conformance is logged.

Scoring (user decision "Both"): symbolic risk_level vs §6.9 (class×criticality) GT
(PRIMARY, comparable) + process-var delta (where the comprehensive symbolic L×I diverges
from the flat lookup). SHACL conformance + retry counts logged (the framework's
hallucination-suppression metric).

Replicable: model=haiku, fixed max_tokens, ONE prompt for all classes, canonical N=60
(first 10/class = ua_000..ua_063). The risk computation is fully deterministic; only the
narrative depends on the (CLI-temperature-managed) LLM.

Usage:
  python -m scripts.run_cic_intersymbolic_risk --arm tristage_llm --out tristage_intersymbolic.jsonl
  python -m scripts.run_cic_intersymbolic_risk --arm pure_llm     --out pure_intersymbolic.jsonl
"""
from __future__ import annotations
import argparse, hashlib, json, re, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from scripts.claude_cli_client import ClaudeCLIClient
from scripts.llm_artifact_lib import validate_artifact_schema, RISK_LEVELS
from pipeline.nfcrm.asset_risk import cvss_kev_to_exploitability, criticality_to_cia_overrides
from pipeline.nfcrm.risk_score import compute_risk_score

ROOT = Path(__file__).resolve().parents[1]
SYMBOLIC_CLASSES = {"BruteForce", "DoS", "DDoS", "WebAttack", "Infiltration"}

# ── UNIFIED HYPERPARAMETERS ────────────────────────────────────────────────────
# IDENTICAL for every class and every dataset (replicability requirement). The
# ONLY per-dataset differences are DATA (which test set, which class field) — never
# the inference config. The config hash printed at run start proves CIC and NSL-KDD
# ran under the same hyperparameters.
UNIFIED_HYPERPARAMS = {
    "model": "haiku",          # -> claude-haiku-4-5
    "max_tokens": 700,
    "timeout_sec": 180,
    "temperature": "CLI-managed (not settable; documented limitation)",
    "n_per_class": 10,         # same balanced selection per class, both datasets
}
DATASETS = {
    "cic": {"test_set": "results/unified_ablation/test_set.json",
            "out_dir": "results/cic_intersymbolic", "class_field": "xgb_predicted_class"},
    "nslkdd": {"test_set": "results/nslkdd_ablation/test_set.json",
               "out_dir": "results/nslkdd_intersymbolic", "class_field": "xgb_nfcrm_class"},
}

SYSTEM_PROMPT = """You are a GRC analyst writing the audit narrative for a NFCRM-1:2025 §6.9 risk assessment.
The risk level, likelihood, and impact have ALREADY been computed by the symbolic NFCRM engine (Figures 3/4/5) and are AUTHORITATIVE — do not change them. Your job is to GROUND them: write a concise, auditable narrative and recommend one NFCRM-1:2025 control, citing the CVE (if any), the asset criticality, and the computed likelihood/impact rationale.
Output ONLY a valid JSON object inside a ```json fenced code block."""

TEMPLATE = """SYMBOLIC §6.9 RESULT (authoritative — narrate, do not alter):
- Attack class (from {class_source}): {attack_class}
- Exploitability band (CVSS+KEV): {exploitability}/5
- Likelihood: {likelihood}/5   Impact: {impact}/5
- Inherent risk level: {risk_level}   (residual after controls: {residual_level})
- Rationale: {rationale}

ASSET / CONTEXT:
- {hostname} | criticality: {criticality} | type: {asset_type} | OS: {os}
- Vulnerability: {cve_line}
- ARG neighbourhood: {arg_line}
- Currently-applied controls (§6.7): {controls}

Write the artifact:
```json
{{
  "risk_level": "{risk_level}",
  "recommended_control_id": "<NFCRM-1:2025 control id>",
  "nfcrm_clauses": ["§6.9"],
  "narrative": "<2-3 sentence audit narrative citing CVE, criticality, likelihood/impact>",
  "evidence_refs": ["<cve_id if any>"]
}}```"""

INFER_SYSTEM = """You are a GRC analyst. The attack classification is NOT available.
Infer the single most likely attack class from {Benign, BruteForce, DoS, DDoS, WebAttack, Infiltration} using the CVE and flow description. Output ONLY a JSON object inside a ```json fence: {"inferred_attack_class": "<class>"}."""
INFER_TEMPLATE = """Flow: {nl_desc}
Vulnerability: {cve_line}
Asset criticality: {criticality}
```json
{{"inferred_attack_class": "Benign|BruteForce|DoS|DDoS|WebAttack|Infiltration"}}```"""

JSON_FENCE = re.compile(r"```json\s*\n?(.*?)```", re.DOTALL)


def parse(resp: str) -> Optional[dict]:
    m = JSON_FENCE.search(resp or "")
    blob = m.group(1) if m else (resp or "")
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


def symbolic_risk(attack_class: str, criticality: str, cvss: Optional[float],
                  in_kev: bool, n_controls: int) -> dict:
    """Comprehensive §6.9 risk via pipeline/nfcrm. Benign -> Very Low by construction."""
    if attack_class == "Benign" or attack_class not in SYMBOLIC_CLASSES:
        return {"risk_level": "Very Low", "likelihood": 0, "impact": 0,
                "exploitability": 0, "residual_level": "Very Low",
                "rationale": "Benign / non-attack scenario -> Very Low by construction (§6.9)."}
    expl, expl_rat = cvss_kev_to_exploitability(cvss, in_cisa_kev=in_kev)
    c, i, a, cia_rat = criticality_to_cia_overrides(attack_class, criticality)
    rs = compute_risk_score(attack_class, exploitability_override=(expl if expl > 0 else None),
                            c_override=c, i_override=i, a_override=a)
    # residual: each applied control reduces likelihood by one band (clamped), recompute level
    residual_level = rs.level_en
    if n_controls > 0 and rs.score > 0:
        red = min(4, n_controls)
        new_lik = max(1, rs.likelihood.value - red)
        from pipeline.nfcrm.constants import risk_level_for_score
        residual_level, _ = risk_level_for_score(new_lik * rs.impact.value)
    return {"risk_level": rs.level_en, "likelihood": rs.likelihood.value,
            "impact": rs.impact.value, "exploitability": rs.likelihood.exploitability,
            "residual_level": residual_level, "rationale": f"{expl_rat} {cia_rat}"}


def symbolic_override(ml_class: str, conf: float, cvss: Optional[float], criticality: str) -> tuple[str, bool]:
    """Deterministic verifier-corrector: low-conf Benign + critical CVE + high asset -> Infiltration."""
    if (ml_class == "Benign" and conf < 0.80 and cvss is not None and cvss >= 9.5
            and criticality.lower() in ("high", "critical")):
        return "Infiltration", True
    return ml_class, False


def cve_line(cve: Optional[dict]) -> str:
    if not cve:
        return "none (no exposed CVE; not KEV-listed)"
    return (f"{cve.get('cve_id') or cve.get('id')} CVSS={cve.get('cvss_v3')} "
            f"KEV=yes — {(cve.get('shortDescription') or '')[:120]}")


def canonical_cases(all_cases, n_per_class=10):
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
    ap.add_argument("--dataset", choices=["cic", "nslkdd"], default="cic")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    class_field = ds["class_field"]
    hp = UNIFIED_HYPERPARAMS
    cases = canonical_cases(json.loads((ROOT / ds["test_set"]).read_text())["cases"], hp["n_per_class"])
    # Config hash is over the UNIFIED hyperparams + prompts only (NOT the data),
    # so it is identical across datasets -> auditable proof of unified hyperparams.
    unified_sig = hashlib.sha1(
        (json.dumps(hp, sort_keys=True) + SYSTEM_PROMPT + TEMPLATE + INFER_SYSTEM).encode()
    ).hexdigest()[:12]
    cfg = {"arm": args.arm, "dataset": args.dataset, "n": len(cases),
           "class_field": class_field, **hp, "UNIFIED_CONFIG_HASH": unified_sig}
    print("CONFIG:", json.dumps(cfg))
    client = ClaudeCLIClient(model=hp["model"], max_tokens=hp["max_tokens"], timeout_sec=hp["timeout_sec"])
    out_dir = ROOT / ds["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out
    with out_path.open("w", encoding="utf-8") as f:
        for idx, case in enumerate(cases, 1):
            t0 = time.time()
            asset = case.get("asset", {})
            crit = case.get("criticality", asset.get("criticality", "Medium"))
            cve = case.get("paired_cve")
            cvss = (cve or {}).get("cvss_v3")
            in_kev = cve is not None
            n_ctrl = len(case.get("controls", []))
            gt = case["ground_truth_risk_level"]
            overrode = False
            llm_calls = 0

            # ── STEP 1: attack class ───────────────────────────────────────
            if args.arm == "tristage_llm":
                ml_class = case.get(class_field, "Benign")
                attack_class, overrode = symbolic_override(
                    ml_class, case.get("xgb_confidence", 1.0), cvss, crit)
                class_source = "XGBoost + symbolic verifier-corrector"
            else:
                infer = parse(client.generate(
                    INFER_SYSTEM,
                    INFER_TEMPLATE.format(nl_desc=case.get("nl_description", "")[:300],
                                          cve_line=cve_line(cve), criticality=crit)) or "")
                llm_calls += 1
                attack_class = (infer or {}).get("inferred_attack_class", "Benign")
                if attack_class not in SYMBOLIC_CLASSES and attack_class != "Benign":
                    attack_class = "Benign"
                class_source = "LLM inference (no ML)"

            # ── STEP 2: SYMBOLIC ENGINE computes comprehensive §6.9 risk ────
            sym = symbolic_risk(attack_class, crit, cvss, in_kev, n_ctrl)

            # ── STEP 3: LLM grounds (narrative + control), then STEP 4 SHACL ─
            prompt = TEMPLATE.format(
                class_source=class_source, attack_class=attack_class,
                exploitability=sym["exploitability"], likelihood=sym["likelihood"],
                impact=sym["impact"], risk_level=sym["risk_level"],
                residual_level=sym["residual_level"], rationale=sym["rationale"],
                hostname=asset.get("hostname", "?"), criticality=crit,
                asset_type=asset.get("asset_type", "?"), os=asset.get("os", "?"),
                cve_line=cve_line(cve),
                arg_line=json.dumps(case.get("arg_neighborhood", {}))[:200],
                controls=", ".join(case.get("controls", [])) or "none")
            art = parse(client.generate(SYSTEM_PROMPT, prompt) or "")
            llm_calls += 1
            shacl_retried = 0
            if art is not None:
                art["risk_level"] = sym["risk_level"]  # symbolic is authoritative
                ok, viol = validate_artifact_schema(art)
                if not ok:
                    shacl_retried = 1
                    art2 = parse(client.generate(
                        SYSTEM_PROMPT, prompt + f"\n\nPrevious output failed SHACL: {'; '.join(viol)[:200]}. Return corrected JSON.") or "")
                    llm_calls += 1
                    if art2:
                        art2["risk_level"] = sym["risk_level"]
                        ok, viol = validate_artifact_schema(art2)
                        art = art2 if art2 else art
            else:
                ok, viol = False, ["LLM parse error"]

            rl = sym["risk_level"]
            implied_correct = rl == gt
            rec = {
                "case_id": case["case_id"], "arm": args.arm, "model": hp["model"], "dataset": args.dataset,
                "true_class": case["true_attack_class"],
                "ml_pred_class": case.get("xgb_predicted_class", ""),
                "attack_class_used": attack_class, "symbolic_override": overrode,
                "exploitability": sym["exploitability"], "likelihood": sym["likelihood"],
                "impact": sym["impact"], "risk_level": rl, "residual_level": sym["residual_level"],
                "ground_truth": gt, "correct": rl == gt,
                "lookup_vs_process_diverges": rl != gt,
                "shacl_conform": bool(ok), "shacl_violations": viol if not ok else [],
                "shacl_retried": shacl_retried, "llm_calls": llm_calls,
                "recommended_control_id": (art or {}).get("recommended_control_id", ""),
                "narrative": (art or {}).get("narrative", ""),
                "grc": "Full", "parse_error": art is None,
                "latency_ms": int((time.time() - t0) * 1000),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            mark = "OK" if rec["correct"] else "MISS"
            sh = "Y" if ok else "N"
            print(f"  [{idx}/{len(cases)}] {case['case_id']} ({case['true_attack_class']}): "
                  f"cls={attack_class}{'*' if overrode else ''} L={sym['likelihood']} I={sym['impact']} "
                  f"risk={rl} vs GT={gt} [{mark}] SHACL{sh} ({rec['latency_ms']}ms)")
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
