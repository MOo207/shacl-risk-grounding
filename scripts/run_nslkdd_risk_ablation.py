"""NSL-KDD 5-paradigm comprehensive risk assessment ablation.

Mirrors run_cic_risk_ablation.py but targets the NSL-KDD test set
(results/nslkdd_ablation/test_set.json, N=50, 10/class).

NSL-KDD raw classes map to NFCRM via NSL_TO_NFCRM; xgb_nfcrm_class in the
test set already carries the mapped class for the ML arm.

Arms:
  ml               - xgb_nfcrm_class -> symbolic SS6.9 (no SHACL, no LLM)
  shacl            - rule pred_class (mapped) -> symbolic + SHACL gate
  llm              - Haiku infers NFCRM class + full SS6.9 end-to-end
  tristage_ml_shacl - xgb_nfcrm_class -> symbolic + SHACL gate
  tristage_llm_shacl - xgb_nfcrm_class -> symbolic -> Haiku narrative -> SHACL

Usage:
  python -m scripts.run_nslkdd_risk_ablation --arms all
  python -m scripts.run_nslkdd_risk_ablation --arms ml shacl tristage_ml_shacl
"""
from __future__ import annotations
import argparse, hashlib, json, re, time
from pathlib import Path
from typing import Optional

from scripts.claude_cli_client import ClaudeCLIClient
from scripts.llm_artifact_lib import validate_artifact_schema, RISK_LEVELS
from pipeline.nfcrm.asset_risk import cvss_kev_to_exploitability, criticality_to_cia_overrides
from pipeline.nfcrm.risk_score import compute_risk_score

ROOT      = Path(__file__).resolve().parents[1]
TEST_SET  = ROOT / "results/nslkdd_ablation/test_set.json"
RULE_JSONL = ROOT / "results/nslkdd_ablation/rule.jsonl"
OUT_DIR   = ROOT / "results/nslkdd_risk_ablation"

# NSL-KDD raw class -> NFCRM canonical class
NSL_TO_NFCRM = {
    "Normal": "Benign",
    "DoS":    "DoS",
    "Probe":  "Infiltration",
    "U2R":    "BruteForce",
    "R2L":    "WebAttack",
}

SYMBOLIC_CLASSES = {"BruteForce", "DoS", "DDoS", "WebAttack", "Infiltration"}

HYPERPARAMS = {
    "model":       "haiku",
    "max_tokens":  700,
    "timeout_sec": 180,
    "n_per_class": 10,
}

CLASS_CONTROL = {
    "DoS":          "SS6.9-DOS-1",
    "DDoS":         "SS6.9-DDoS-1",
    "BruteForce":   "SS6.9-BF-1",
    "WebAttack":    "SS6.9-WA-1",
    "Infiltration": "SS6.9-INF-1",
    "Benign":       "SS5.1-MON-1",
}

RISK_MATRIX = {
    ("Low",       "Low"):          "Very Low",
    ("Low",       "Medium"):       "Low",
    ("Low",       "High"):         "Medium",
    ("Low",       "Catastrophic"): "High",
    ("Medium",    "Low"):          "Low",
    ("Medium",    "Medium"):       "Medium",
    ("Medium",    "High"):         "High",
    ("Medium",    "Catastrophic"): "Catastrophic",
    ("High",      "Low"):          "Medium",
    ("High",      "Medium"):       "High",
    ("High",      "High"):         "Catastrophic",
    ("High",      "Catastrophic"): "Catastrophic",
    ("Very High", "Low"):          "Medium",
    ("Very High", "Medium"):       "High",
    ("Very High", "High"):         "Catastrophic",
    ("Very High", "Catastrophic"): "Catastrophic",
}

JSON_FENCE   = re.compile(r"```json\s*\n?(.*?)```", re.DOTALL)
ALL_ARMS     = ["ml", "shacl", "llm", "tristage_ml_shacl", "tristage_llm_shacl"]
OFFLINE_ARMS = {"ml", "shacl", "tristage_ml_shacl"}
LLM_ARMS     = {"llm", "tristage_llm_shacl"}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_test_set() -> list[dict]:
    return json.loads(TEST_SET.read_text())["cases"]   # already exactly 50


def load_rule_classes() -> dict[str, str]:
    """Return {case_id: nfcrm_class} from rule.jsonl (maps NSL raw -> NFCRM)."""
    mapping: dict[str, str] = {}
    with RULE_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                raw = r.get("pred_class", "Normal")
                mapping[r["case_id"]] = NSL_TO_NFCRM.get(raw, "Benign")
            except (json.JSONDecodeError, KeyError):
                continue
    return mapping


# ── Shared engine helpers ─────────────────────────────────────────────────────

def symbolic_risk(attack_class: str, criticality: str, cvss: Optional[float],
                  in_kev: bool, n_controls: int) -> dict:
    if attack_class == "Benign" or attack_class not in SYMBOLIC_CLASSES:
        if attack_class != "Benign":
            print(f"WARN: unknown attack_class '{attack_class}', treating as Very Low")
        return {"risk_level": "Very Low", "likelihood": 0, "impact": 0,
                "exploitability": 0, "residual_level": "Very Low",
                "rationale": "Benign / non-attack -> Very Low by construction (SS6.9)."}
    expl, expl_rat = cvss_kev_to_exploitability(cvss, in_cisa_kev=in_kev)
    c, i, a, cia_rat = criticality_to_cia_overrides(attack_class, criticality)
    rs = compute_risk_score(attack_class,
                            exploitability_override=(expl if expl > 0 else None),
                            c_override=c, i_override=i, a_override=a)
    residual_level = rs.level_en
    if n_controls > 0 and rs.score > 0:
        red = min(4, n_controls)
        new_lik = max(1, rs.likelihood.value - red)
        from pipeline.nfcrm.constants import risk_level_for_score
        residual_level, _ = risk_level_for_score(new_lik * rs.impact.value)
    expl_val = getattr(rs.likelihood, "exploitability", expl)
    return {"risk_level": rs.level_en, "likelihood": rs.likelihood.value,
            "impact": rs.impact.value, "exploitability": expl_val,
            "residual_level": residual_level, "rationale": f"{expl_rat} {cia_rat}"}


def symbolic_override(ml_class: str, conf: float, cvss: Optional[float],
                       criticality: str) -> tuple[str, bool]:
    if (ml_class == "Benign" and conf < 0.80 and cvss is not None
            and cvss >= 9.5 and criticality.lower() in ("high", "critical")):
        return "Infiltration", True
    return ml_class, False


def cve_info(case: dict) -> tuple[Optional[float], bool, str]:
    cve = case.get("paired_cve")
    if not cve:
        return None, False, "none"
    # in_kev=True for all paired CVEs (matches existing intersymbolic runs)
    return cve.get("cvss_v3"), True, (cve.get("cve_id") or cve.get("id") or "none")


def nfcrm_shacl_gate(artifact: dict) -> tuple[bool, list[str]]:
    ok, errs = validate_artifact_schema(artifact)
    ctrl = artifact.get("recommended_control_id", "")
    if ctrl and not (ctrl.startswith("SS5") or ctrl.startswith("SS6")):
        errs.append(f"recommended_control_id '{ctrl}' must start with SS5 or SS6")
        ok = False
    clauses = artifact.get("nfcrm_clauses", [])
    if isinstance(clauses, list) and len(clauses) == 0:
        errs.append("nfcrm_clauses list is empty")
        ok = False
    narr = artifact.get("narrative", "")
    if isinstance(narr, str) and len(narr) < 10:
        errs.append(f"narrative too short ({len(narr)} chars, min 10)")
        ok = False
    return bool(ok), errs


def build_template_artifact(attack_class: str, case: dict, sym: dict) -> dict:
    asset  = case.get("asset", {})
    cve    = case.get("paired_cve")
    crit   = case.get("criticality", asset.get("criticality", "Medium"))
    cve_id = (cve or {}).get("cve_id") or (cve or {}).get("id") or "none"
    cvss   = (cve or {}).get("cvss_v3", "N/A")
    kev    = "yes" if cve else "no"
    host   = asset.get("hostname", case.get("case_id", "unknown"))
    narr   = (
        f"{attack_class} on {host} (criticality={crit}, CVSS={cvss}, KEV={kev}): "
        f"symbolic NFCRM SS6.9 engine yields LIKELIHOOD={sym['likelihood']}/5, "
        f"IMPACT={sym['impact']}/5 -> {sym['risk_level']} risk."
    )
    return {
        "risk_level":             sym["risk_level"],
        "recommended_control_id": CLASS_CONTROL.get(attack_class, "SS6.9-GEN-1"),
        "nfcrm_clauses":          ["SS6.9"],
        "narrative":              narr,
        "evidence_refs":          [cve_id],
    }


def grounding_score(narrative: str) -> int:
    if not narrative:
        return 0
    score = 0
    if re.search(r'CVE-\d{4}-\d+|CVSS[\s=:]*\d', narrative, re.IGNORECASE):
        score += 1
    if re.search(r'\b(critical|high|medium|low)\b', narrative, re.IGNORECASE):
        score += 1
    if re.search(r'\b(likelihood|impact|very high|very low|catastrophic)\b',
                 narrative, re.IGNORECASE):
        score += 1
    return score


def parse_json(resp: str) -> Optional[dict]:
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


def _safe(s: str) -> str:
    return s.replace("{", "{{").replace("}", "}}")


# ── Offline arm runner ────────────────────────────────────────────────────────

def run_offline_arm(arm: str, cases: list[dict],
                    rule_classes: dict[str, str]) -> list[dict]:
    records = []
    for case in cases:
        t0      = time.time()
        asset   = case.get("asset", {})
        crit    = case.get("criticality", asset.get("criticality", "Medium"))
        cvss, in_kev, _ = cve_info(case)
        n_ctrl  = len(case.get("controls") or [])
        gt      = case["ground_truth_risk_level"]
        overrode = False

        if arm == "ml":
            attack_class = case.get("xgb_nfcrm_class", "Benign")
            class_source = "xgb_nfcrm"
        elif arm == "shacl":
            attack_class = rule_classes.get(case["case_id"], "Benign")
            class_source = "rule_nfcrm"
        else:  # tristage_ml_shacl
            ml_class = case.get("xgb_nfcrm_class", "Benign")
            attack_class, overrode = symbolic_override(
                ml_class, case.get("xgb_confidence", 1.0), cvss, crit)
            class_source = "xgb_nfcrm+override"

        sym = symbolic_risk(attack_class, crit, cvss, in_kev, n_ctrl)

        shacl_conform: Optional[bool] = None
        shacl_violations: list[str]   = []

        if arm == "ml":
            grc_complete = False
        else:
            artifact = build_template_artifact(attack_class, case, sym)
            if arm in ("shacl", "tristage_ml_shacl"):
                ok, shacl_violations = nfcrm_shacl_gate(artifact)
                shacl_conform = ok
            grc_complete = bool(
                artifact.get("risk_level") and artifact.get("recommended_control_id") and
                artifact.get("nfcrm_clauses") and artifact.get("evidence_refs") is not None
            )

        records.append({
            "case_id":             case["case_id"],
            "arm":                 arm,
            "true_class":          case["true_attack_class"],
            "true_nfcrm_class":    case["true_nfcrm_class"],
            "attack_class_used":   attack_class,
            "class_source":        class_source,
            "symbolic_override":   overrode,
            "risk_level":          sym["risk_level"],
            "ground_truth":        gt,
            "correct":             sym["risk_level"] == gt,
            "shacl_conform":       shacl_conform,
            "shacl_violations":    shacl_violations,
            "shacl_retried":       0,
            "grc_complete":        grc_complete,
            "grounding_depth":     None,
            "internal_consistent": None,
            "llm_calls":           0,
            "parse_error":         False,
            "latency_ms":          int((time.time() - t0) * 1000),
        })
        mark = "OK" if sym["risk_level"] == gt else "MISS"
        sh   = f" SHACL={'Y' if shacl_conform else 'N'}" if shacl_conform is not None else ""
        print(f"  [{case['case_id']}] {case['true_attack_class']:8}/{case['true_nfcrm_class']:12} "
              f"cls={attack_class:12} risk={sym['risk_level']:10} GT={gt:10} [{mark}]{sh}")
    return records


# ── LLM arm prompts ───────────────────────────────────────────────────────────

LLM_SYSTEM = """You are a GRC risk analyst performing a NFCRM-1:2025 SS6.9 technical asset risk assessment on NSL-KDD network connection data.
Assess risk as LIKELIHOOD x IMPACT using ALL provided process variables.

The asset hosts network services. The connection description uses NSL-KDD features.
Map the connection to one NFCRM class: {Benign, BruteForce, DoS, WebAttack, Infiltration}.
  - Normal/legitimate traffic -> Benign
  - Denial-of-service patterns (high volume, REJ flags, neptune/smurf) -> DoS
  - Privilege escalation / root compromise (U2R) -> BruteForce
  - Remote-to-local / credential theft (R2L) -> WebAttack
  - Reconnaissance / port scan / probe -> Infiltration

LIKELIHOOD scale {Low, Medium, High, Very High}: driven by CVE CVSS, KEV status, attack class exploitability.
IMPACT scale {Low, Medium, High, Catastrophic}: driven by asset criticality, attack consequence.
RISK = LIKELIHOOD x IMPACT via matrix:
              Impact:  Low     Medium  High         Catastrophic
  Lik Low              Very Low Low     Medium       High
  Lik Medium           Low      Medium  High         Catastrophic
  Lik High             Medium   High    Catastrophic Catastrophic
  Lik Very High        Medium   High    Catastrophic Catastrophic

Output ONLY a valid JSON object inside a ```json fenced code block."""

LLM_TEMPLATE = """NFCRM SS6.9 RISK ASSESSMENT (NSL-KDD):

[ASSET]
- hostname: {hostname}
- criticality: {criticality}
- asset_type: {asset_type}

[CONNECTION BEHAVIOUR]
{nl_desc}

[VULNERABILITY]
{cve_block}

Steps:
1. Infer NFCRM attack_class from {{Benign, BruteForce, DoS, WebAttack, Infiltration}}.
2. LIKELIHOOD (Low/Medium/High/Very High) -- justify citing CVSS and KEV status.
3. IMPACT (Low/Medium/High/Catastrophic) -- justify citing asset criticality.
4. RISK_LEVEL = Likelihood x Impact via the matrix.
5. Recommend one NFCRM-1:2025 control (starting SS5 or SS6) and write a 2-3 sentence audit narrative citing CVE, criticality, and likelihood/impact.

```json
{{
  "attack_class": "Benign|BruteForce|DoS|WebAttack|Infiltration",
  "likelihood": "Low|Medium|High|Very High",
  "impact": "Low|Medium|High|Catastrophic",
  "risk_level": "Very Low|Low|Medium|High|Catastrophic",
  "recommended_control_id": "SS5.x-XXX-n or SS6.x-XXX-n",
  "nfcrm_clauses": ["SS6.9"],
  "narrative": "2-3 sentences citing CVE/CVSS, asset criticality, likelihood/impact",
  "evidence_refs": ["CVE-YYYY-NNNNN or none"]
}}```"""


def _cve_block_llm(cve: Optional[dict]) -> str:
    if not cve:
        return "- No paired CVE (no known vulnerability; not KEV-listed)."
    return (f"- CVE: {cve.get('cve_id') or cve.get('id')}\n"
            f"- CVSS v3 base: {cve.get('cvss_v3')}\n"
            f"- KEV (known-exploited): yes\n"
            f"- Description: {(cve.get('shortDescription') or '')[:150]}")


def run_llm_arm(cases: list[dict], client: ClaudeCLIClient, out_f=None) -> list[dict]:
    records = []
    for i, case in enumerate(cases, 1):
        t0    = time.time()
        asset = case.get("asset", {})
        crit  = case.get("criticality", asset.get("criticality", "Medium"))
        cve   = case.get("paired_cve")
        gt    = case["ground_truth_risk_level"]

        prompt = LLM_TEMPLATE.format(
            hostname   = asset.get("hostname", case.get("case_id", "?")),
            criticality = crit,
            asset_type  = asset.get("asset_type", "Server"),
            nl_desc     = case.get("nl_description", "")[:350],
            cve_block   = _cve_block_llm(cve),
        )
        resp = client.generate(LLM_SYSTEM, prompt) or ""
        art  = parse_json(resp)

        if art:
            rl      = art.get("risk_level", "")
            lik     = art.get("likelihood", "")
            imp     = art.get("impact", "")
            implied = RISK_MATRIX.get((lik, imp), "")
            narr    = art.get("narrative", "")
            internal = (implied == rl) if implied else None
            grc_ok  = bool(
                art.get("risk_level") and art.get("recommended_control_id") and
                art.get("nfcrm_clauses") and art.get("evidence_refs") is not None
            )
            rec = {
                "case_id": case["case_id"], "arm": "llm",
                "true_class": case["true_attack_class"],
                "true_nfcrm_class": case["true_nfcrm_class"],
                "attack_class_used": art.get("attack_class", ""),
                "class_source": "llm",
                "symbolic_override": False,
                "risk_level": rl, "ground_truth": gt, "correct": rl == gt,
                "shacl_conform": None, "shacl_violations": [], "shacl_retried": 0,
                "grc_complete": grc_ok,
                "grounding_depth": grounding_score(narr),
                "internal_consistent": internal,
                "llm_calls": 1, "parse_error": False,
                "latency_ms": int((time.time() - t0) * 1000),
            }
        else:
            rec = {
                "case_id": case["case_id"], "arm": "llm",
                "true_class": case["true_attack_class"],
                "true_nfcrm_class": case["true_nfcrm_class"],
                "attack_class_used": "", "class_source": "llm",
                "symbolic_override": False,
                "risk_level": "", "ground_truth": gt, "correct": False,
                "shacl_conform": None, "shacl_violations": [], "shacl_retried": 0,
                "grc_complete": False, "grounding_depth": 0,
                "internal_consistent": None,
                "llm_calls": 1, "parse_error": True,
                "needs_human_review": True,
                "latency_ms": int((time.time() - t0) * 1000),
            }
        mark = "OK" if rec["correct"] else ("ERR" if rec["parse_error"] else "MISS")
        print(f"  [{i:02}/{len(cases)}] {case['case_id']} ({case['true_attack_class']:6}/{case['true_nfcrm_class']:12}): "
              f"risk={rec['risk_level'] or 'PARSE_ERR':10} GT={gt:10} [{mark}] depth={rec['grounding_depth']}",
              flush=True)
        records.append(rec)
        if out_f is not None:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); out_f.flush()
    return records


# ── Tri-stage LLM+SHACL arm ───────────────────────────────────────────────────

# ── LLM class inference for tristage (NSL-KDD) ────────────────────────────────
# NSL-KDD XGBoost is over-fitted to Normal (majority class) and confidently
# mispredicts R2L and U2R as Benign even at 99-100% confidence.
# On NSL-KDD, LLM inference from NL description + CVE outperforms XGBoost
# (original ablation: LLM Haiku 78% vs XGBoost 56%).
# The Tri-stage LLM+SHACL arm therefore uses LLM class inference (Step 1)
# feeding into the symbolic §6.9 engine (Step 2) + narrative + SHACL (Step 3).
# This mirrors run_cic_intersymbolic_risk.py --arm pure_llm but for NSL-KDD.

INFER_SYSTEM = """You are a network security analyst classifying NSL-KDD network connections into NFCRM-1:2025 attack categories.

Decide FIRST whether the connection is benign or an attack, then pick the attack type.

NSL-KDD classes and their NFCRM mappings:
- Normal legitimate traffic -> Benign
- DoS (denial-of-service, neptune, smurf, back, pod, teardrop): high volume, REJ flags, flooding -> DoS
- Probe (reconnaissance, portsweep, nmap, satan, ipsweep): scanning many hosts/ports, short connections -> Infiltration
- R2L (remote-to-local: ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster):
  credential theft, unauthorized file access, exploiting remote services -> WebAttack
- U2R (user-to-root: buffer_overflow, loadmodule, perl, rootkit):
  privilege escalation, root compromise, exploit execution -> BruteForce

BENIGN indicators (classify as Benign when these dominate -- do NOT over-call attacks):
- flag=SF (connection completed normally: both sides exchanged data and closed cleanly)
- logged_in=1 (an authenticated, valid session is in progress)
- LOW connection counts (count / srv_count / dst_host_count in the single/low-tens, "not flooding")
- no SYN or REJ errors; normal request/response byte exchange to a standard service (e.g. http)
A clean SF connection on a normal service by a logged-in user with low counts is Benign, NOT DoS.

ATTACK indicators:
- R2L: failed logins (guess_passwd, hot/num_failed_logins), small payloads, sensitive service ports (ftp=21, smtp=25)
- U2R: root shell access (root_shell=1, su_attempted), local connections, buffer-overflow patterns (large src_bytes vs duration)
- DoS: VERY high counts (count, srv_count > 200), REJ flags, SYN flood / saturation to a SINGLE repeated service
- Probe: diff_srv_rate > 0.5, dst_host_diff_srv_rate > 0.5, scanning MANY unique destinations/services, REJ sweeps

Output ONLY a JSON object inside a ```json fence."""

INFER_TEMPLATE = """Classify this NSL-KDD connection into the correct NFCRM attack class:

Connection: {nl_desc}

CVE context: {cve_block}
Asset: {hostname} | criticality: {criticality}

```json
{{"inferred_class": "Benign|DoS|Infiltration|WebAttack|BruteForce", "reason": "<one line citing key features>"}}```"""

VALID_NFCRM = {"Benign", "DoS", "Infiltration", "WebAttack", "BruteForce"}


def llm_infer_class(case: dict, client: ClaudeCLIClient) -> tuple[str, int]:
    """LLM infers NFCRM class from NL description. Returns (class, llm_calls)."""
    asset = case.get("asset", {})
    crit  = case.get("criticality", asset.get("criticality", "Medium"))
    cve   = case.get("paired_cve")
    prompt = INFER_TEMPLATE.format(
        nl_desc     = case.get("nl_description", "")[:350],
        cve_block   = _cve_block_llm(cve),
        hostname    = asset.get("hostname", case.get("case_id", "?")),
        criticality = crit,
    )
    resp = client.generate(INFER_SYSTEM, prompt) or ""
    art  = parse_json(resp)
    if art and art.get("inferred_class") in VALID_NFCRM:
        return art["inferred_class"], 1
    return "Benign", 1   # safe fallback


TRISTAGE_SYSTEM = """You are a GRC analyst writing the audit narrative for a NFCRM-1:2025 SS6.9 risk assessment on NSL-KDD data.
The risk level, likelihood, and impact have been computed by the symbolic NFCRM engine and are AUTHORITATIVE -- do not change them.
Write a concise 2-3 sentence narrative citing the CVE (if any), asset criticality, and the computed likelihood/impact rationale. Recommend one NFCRM-1:2025 control (starting SS5 or SS6).
Output ONLY a valid JSON object inside a ```json fenced code block."""

TRISTAGE_TEMPLATE = """SYMBOLIC SS6.9 RESULT (authoritative -- narrate only):
- Attack class: {attack_class}
- Likelihood: {likelihood}/5   Impact: {impact}/5
- Inherent risk level: {risk_level}   Residual after controls: {residual_level}
- Rationale: {rationale}

ASSET / CONTEXT:
- {hostname} | criticality: {criticality} | type: {asset_type}
- Vulnerability: {cve_line}
- Connection: {nl_desc}

```json
{{
  "risk_level": "{risk_level}",
  "recommended_control_id": "<NFCRM-1:2025 control starting SS5 or SS6>",
  "nfcrm_clauses": ["SS6.9"],
  "narrative": "<2-3 sentence audit narrative citing CVE, asset criticality, likelihood/impact>",
  "evidence_refs": ["<cve_id or none>"]
}}```"""


def _cve_line_str(case: dict) -> str:
    cve = case.get("paired_cve")
    if not cve:
        return "none (no exposed CVE; not KEV-listed)"
    return (f"{cve.get('cve_id') or cve.get('id')} CVSS={cve.get('cvss_v3')} "
            f"KEV=yes -- {(cve.get('shortDescription') or '')[:100]}")


def run_tristage_llm_arm(cases: list[dict], client: ClaudeCLIClient, out_f=None) -> list[dict]:
    records = []
    for i, case in enumerate(cases, 1):
        t0    = time.time()
        asset = case.get("asset", {})
        crit  = case.get("criticality", asset.get("criticality", "Medium"))
        cvss, in_kev, _ = cve_info(case)
        n_ctrl  = len(case.get("controls") or [])
        gt      = case["ground_truth_risk_level"]

        # Step 1: LLM infers NFCRM class from NL description + CVE.
        # XGBoost is NOT used for class detection in this arm because NSL-KDD's
        # class imbalance causes XGBoost to confidently mispredict R2L/U2R as
        # Benign (56% accuracy vs LLM's ~78%). The three stages are:
        # (1) LLM class inference, (2) symbolic §6.9 risk, (3) LLM narrative + SHACL.
        attack_class, infer_calls = llm_infer_class(case, client)
        llm_calls = infer_calls
        overrode  = attack_class != case.get("xgb_nfcrm_class", "Benign")

        sym = symbolic_risk(attack_class, crit, cvss, in_kev, n_ctrl)

        prompt = TRISTAGE_TEMPLATE.format(
            attack_class   = attack_class,
            likelihood     = sym["likelihood"],
            impact         = sym["impact"],
            risk_level     = sym["risk_level"],
            residual_level = sym["residual_level"],
            rationale      = _safe(sym["rationale"]),
            hostname       = asset.get("hostname", case.get("case_id", "?")),
            criticality    = crit,
            asset_type     = asset.get("asset_type", "Server"),
            cve_line       = _safe(_cve_line_str(case)),
            nl_desc        = case.get("nl_description", "")[:200],
        )
        resp = client.generate(TRISTAGE_SYSTEM, prompt) or ""
        art  = parse_json(resp)
        llm_calls += 1   # narrative call on top of verify call

        shacl_retried = 0
        if art is not None:
            art["risk_level"] = sym["risk_level"]
            ok, violations = nfcrm_shacl_gate(art)
            if not ok:
                shacl_retried = 1
                resp2 = client.generate(
                    TRISTAGE_SYSTEM,
                    prompt + f"\n\nPrevious output failed SHACL: {'; '.join(violations)[:200]}. Return corrected JSON.") or ""
                art2 = parse_json(resp2)
                llm_calls += 1
                if art2:
                    art2["risk_level"] = sym["risk_level"]
                    ok, violations = nfcrm_shacl_gate(art2)
                    art = art2
            narr   = art.get("narrative", "")
            grc_ok = bool(
                art.get("risk_level") and art.get("recommended_control_id") and
                art.get("nfcrm_clauses") and art.get("evidence_refs") is not None
            )
            rec = {
                "case_id": case["case_id"], "arm": "tristage_llm_shacl",
                "true_class": case["true_attack_class"],
                "true_nfcrm_class": case["true_nfcrm_class"],
                "attack_class_used": attack_class, "class_source": "xgb_nfcrm+override",
                "symbolic_override": overrode,
                "risk_level": sym["risk_level"], "ground_truth": gt,
                "correct": sym["risk_level"] == gt,
                "shacl_conform": bool(ok),
                "shacl_violations": violations if not ok else [],
                "shacl_retried": shacl_retried, "grc_complete": grc_ok,
                "grounding_depth": grounding_score(narr),
                "internal_consistent": None,
                "llm_calls": llm_calls, "parse_error": False,
                "latency_ms": int((time.time() - t0) * 1000),
            }
        else:
            rec = {
                "case_id": case["case_id"], "arm": "tristage_llm_shacl",
                "true_class": case["true_attack_class"],
                "true_nfcrm_class": case["true_nfcrm_class"],
                "attack_class_used": attack_class, "class_source": "xgb_nfcrm+override",
                "symbolic_override": overrode,
                "risk_level": sym["risk_level"], "ground_truth": gt,
                "correct": sym["risk_level"] == gt,
                "shacl_conform": False, "shacl_violations": ["LLM parse error"],
                "shacl_retried": 0, "grc_complete": False, "grounding_depth": 0,
                "internal_consistent": None,
                "llm_calls": 1, "parse_error": True,
                "needs_human_review": True,
                "latency_ms": int((time.time() - t0) * 1000),
            }
        sh   = "Y" if rec["shacl_conform"] else "N"
        mark = "OK" if rec["correct"] else "MISS"
        print(f"  [{i:02}/{len(cases)}] {case['case_id']} ({case['true_attack_class']:6}/{case['true_nfcrm_class']:12}): "
              f"cls={attack_class:12}{'*' if overrode else ' '} "
              f"risk={sym['risk_level']:10} GT={gt:10} [{mark}] SHACL={sh} depth={rec['grounding_depth']}",
              flush=True)
        records.append(rec)
        if out_f is not None:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); out_f.flush()
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="NSL-KDD 5-paradigm risk assessment ablation")
    ap.add_argument("--arms", nargs="+", default=["all"], metavar="ARM",
                    help=f"Arms to run: {ALL_ARMS} or 'all'")
    ap.add_argument("--model",       default=HYPERPARAMS["model"])
    ap.add_argument("--max-tokens",  type=int, default=HYPERPARAMS["max_tokens"])
    args = ap.parse_args()

    arms_to_run = ALL_ARMS if "all" in args.arms else args.arms
    hp = {**HYPERPARAMS, "model": args.model, "max_tokens": args.max_tokens}

    sig = hashlib.sha1(
        (json.dumps(hp, sort_keys=True) +
         LLM_SYSTEM + LLM_TEMPLATE + TRISTAGE_SYSTEM + TRISTAGE_TEMPLATE).encode()
    ).hexdigest()[:12]
    print("CONFIG:", json.dumps({**hp, "arms": arms_to_run, "config_hash": sig}))

    cases        = load_test_set()
    rule_classes = load_rule_classes()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"NSL-KDD test set: {len(cases)} cases, classes={set(c['true_attack_class'] for c in cases)}")

    client: Optional[ClaudeCLIClient] = None
    if any(a in LLM_ARMS for a in arms_to_run):
        client = ClaudeCLIClient(model=hp["model"], max_tokens=hp["max_tokens"],
                                 timeout_sec=HYPERPARAMS["timeout_sec"])

    for arm in ALL_ARMS:
        if arm not in arms_to_run:
            continue
        print(f"\n{'='*60}\nARM: {arm}\n{'='*60}")
        out_path = OUT_DIR / f"{arm}.jsonl"

        if arm in OFFLINE_ARMS:
            records = run_offline_arm(arm, cases, rule_classes)
            with out_path.open("w", encoding="utf-8") as fout:
                for r in records:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            # LLM arms write incrementally (crash-safe; observable progress)
            with out_path.open("w", encoding="utf-8") as fout:
                if arm == "llm":
                    records = run_llm_arm(cases, client, out_f=fout)           # type: ignore[arg-type]
                else:
                    records = run_tristage_llm_arm(cases, client, out_f=fout)  # type: ignore[arg-type]

        n_correct = sum(r["correct"] for r in records)
        print(f"\n-> {out_path.name}  accuracy={n_correct}/{len(records)} ({n_correct/len(records):.1%})", flush=True)

    print(f"\nAll done. Results in {OUT_DIR}")
    print("Run: python -m scripts.analyze_nslkdd_risk_ablation")


if __name__ == "__main__":
    main()
