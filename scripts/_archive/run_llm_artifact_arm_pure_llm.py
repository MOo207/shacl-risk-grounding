"""Pure LLM arm — LLM-only risk artifact generation (no rule or ML pre-processing).

Attack class is always treated as UNKNOWN. LLM must infer risk level entirely from:
  - CVE context (CVSS score, KEV description, required action)
  - ARG neighborhood CVEs
  - Asset metadata (criticality, controls)

This arm establishes the LLM-only baseline: no symbolic pre-filter, no ML prediction.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from scripts.claude_cli_client import ClaudeCLIClient
from scripts.llm_artifact_lib import RISK_LEVELS, validate_artifact_schema, stratified_slice

SYSTEM_PROMPT = """You are a GRC analyst applying NFCRM-1:2025 (Saudi Arabia National Cybersecurity Risk Management framework).

The attack classification is NOT available — you must infer the likely threat from the CVE context and asset details.

NFCRM §6.9 Risk Level Lookup Table:
  Use CVE CVSS score and KEV description to infer attack severity, then combine with asset criticality:

  Inferred high-severity attack (CVSS ≥ 7.5, exploitation in wild, DoS/RCE KEV):
    LOW criticality   → "Medium"
    MEDIUM criticality → "High"
    HIGH criticality   → "Catastrophic"
    CRITICAL criticality → "Catastrophic"

  Inferred medium-severity (CVSS 4–7.4, credential or web exploitation KEV):
    LOW criticality   → "Low"
    MEDIUM criticality → "Medium"
    HIGH criticality   → "High"
    CRITICAL criticality → "High"

  Inferred low/benign (CVSS < 4, no CVE, or monitoring-only):
    Any criticality → "Very Low" or "Low"

Output ONLY a valid JSON object inside a ```json fenced code block."""

PROMPT_TEMPLATE = """Security event (attack class UNKNOWN — infer from evidence below):
- Asset: {asset_json}
- Currently-applied controls: {controls_json}

CVE context (primary signal for threat inference):
{cve_block}

ARG neighborhood (supporting signal for control selection):
{arg_json}

Apply the §6.9 risk matrix from your system instructions:
1. Infer likely attack severity from CVE CVSS score and KEV description
2. Map asset criticality → impact level
3. Derive risk_level from severity × impact
4. Select the most relevant NFCRM control
5. Write a 2-4 sentence narrative citing the CVE, asset criticality, and inferred threat

Required output (JSON inside ```json fence):
{{
  "risk_level": "Very Low | Low | Medium | High | Catastrophic",
  "recommended_control_id": "<NFCRM-1:2025 control reference e.g. SS6.7-AC-1>",
  "nfcrm_clauses": ["§6.9", "<other §references>"],
  "narrative": "<cite CVE + asset criticality + inferred threat in 2-4 sentences>",
  "evidence_refs": ["<cve_id if present>", "<asset_id>"]
}}"""


def _format_cve_block(cve: Optional[dict]) -> str:
    if not cve:
        return "No vulnerability context paired. Reason from asset criticality and ARG neighborhood alone."
    lines = [
        f"- CVE: {cve.get('cve_id')}",
        f"- Vendor/Product: {cve.get('vendor')} / {cve.get('product')}",
        f"- CVSS v3 base score: {cve.get('cvss_v3')}",
        f"- KEV description: {cve.get('shortDescription')}",
        f"- KEV required action: {cve.get('requiredAction')}",
    ]
    return "\n".join(lines)


def build_prompt(case: dict[str, Any]) -> str:
    asset = dict(case.get("asset", {}))
    return PROMPT_TEMPLATE.format(
        asset_json=json.dumps(asset),
        arg_json=json.dumps(case.get("arg_neighborhood", {}))[:600],
        controls_json=json.dumps(case.get("controls", [])),
        cve_block=_format_cve_block(case.get("paired_cve")),
    )


JSON_FENCE = re.compile(r"```json\s*\n?(.*?)```", re.DOTALL)


def parse_llm_response(response: str) -> dict[str, Any]:
    match = JSON_FENCE.search(response)
    blob = match.group(1) if match else response
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        first_brace = blob.find("{")
        last_brace = blob.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            return json.loads(blob[first_brace : last_brace + 1])
        raise


def _shacl_validate(artifact: dict) -> tuple[bool, list[str]]:
    ok, errors = validate_artifact_schema(artifact)
    if not ok:
        return False, errors
    if artifact.get("risk_level") not in RISK_LEVELS:
        return False, [f"risk_level '{artifact.get('risk_level')}' not in RISK_LEVELS"]
    if not artifact.get("recommended_control_id", "").strip():
        return False, ["recommended_control_id is empty"]
    if not artifact.get("narrative", "").strip():
        return False, ["narrative is empty"]
    return True, []


def run_llm_on_case(case: dict, *, client: Any, validate_shacl: bool = True,
                    max_shacl_retries: int = 1) -> dict[str, Any]:
    prompt = build_prompt(case)
    try:
        response = client.generate(SYSTEM_PROMPT, prompt) or ""
    except TypeError:
        try:
            response = client.generate(f"{SYSTEM_PROMPT}\n\n{prompt}") or ""
        except Exception:
            response = ""

    try:
        artifact = parse_llm_response(response)
    except (json.JSONDecodeError, AttributeError, ValueError):
        return {
            "case_id": case["case_id"], "arm": "pure_llm",
            "risk_level": "", "recommended_control_id": "",
            "nfcrm_clauses": [], "narrative": "", "evidence_refs": [],
            "parse_error": True, "raw_response": response[:500],
            "shacl_violations": [], "shacl_retried": 0,
        }

    artifact["case_id"] = case["case_id"]
    artifact["arm"] = "pure_llm"
    artifact.setdefault("raw_response", response[:500])

    if not validate_shacl:
        artifact["shacl_violations"] = []
        artifact["shacl_retried"] = 0
        return artifact

    ok, violations = _shacl_validate(artifact)
    artifact["shacl_violations"] = violations
    artifact["shacl_retried"] = 0
    if ok:
        return artifact

    for _ in range(max_shacl_retries):
        retry_prompt = (
            prompt + "\n\nPrevious response failed validation: "
            + "; ".join(violations)[:500]
            + "\n\nReturn a corrected JSON inside the same fence."
        )
        try:
            retry_response = client.generate(SYSTEM_PROMPT, retry_prompt) or ""
        except TypeError:
            retry_response = client.generate(f"{SYSTEM_PROMPT}\n\n{retry_prompt}") or ""
        try:
            new_artifact = parse_llm_response(retry_response)
        except (json.JSONDecodeError, AttributeError, ValueError):
            artifact["shacl_retried"] += 1
            continue
        new_artifact["case_id"] = case["case_id"]
        new_artifact["arm"] = "pure_llm"
        new_artifact["raw_response"] = retry_response[:500]
        artifact["shacl_retried"] += 1
        ok, violations = _shacl_validate(new_artifact)
        if ok:
            new_artifact["shacl_violations"] = []
            new_artifact["shacl_retried"] = artifact["shacl_retried"]
            return new_artifact
        artifact = new_artifact
        artifact["shacl_violations"] = violations

    return artifact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", default="results/llm_experiment/test_set.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-shacl", action="store_true")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--n-per-class", type=int, default=None)
    ap.add_argument("--start-offset", type=int, default=0)
    ap.add_argument("--max-cases", type=int, default=None)
    args = ap.parse_args()

    all_cases = json.loads(Path(args.test_set).read_text())["cases"]
    cases = stratified_slice(all_cases, args.n_per_class)
    cases = cases[args.start_offset:]
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    model = args.model
    append = args.start_offset > 0
    print(f"Model: {model} | Cases: {len(cases)} | SHACL: {not args.no_shacl} | "
          f"offset={args.start_offset} | append={append}")
    client = ClaudeCLIClient(model=model, max_tokens=800)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with Path(args.out).open(mode) as f:
        for i, case in enumerate(cases, args.start_offset + 1):
            t0 = time.time()
            artifact = run_llm_on_case(case, client=client, validate_shacl=(not args.no_shacl))
            artifact["latency_ms"] = int((time.time() - t0) * 1000)
            artifact["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            artifact["model"] = model
            artifact["ground_truth_risk_level"] = case.get("ground_truth_risk_level", "")
            artifact["true_attack_class"] = case.get("true_attack_class", "")
            artifact["subset"] = case.get("subset", "")
            f.write(json.dumps(artifact) + "\n")
            f.flush()
            level = artifact.get("risk_level") or "PARSE_ERR"
            gt = case.get("ground_truth_risk_level", "?")
            match = "OK" if level == gt else "MISS"
            total = len(cases) + args.start_offset
            print(f"  [{i}/{total}] {case['case_id']} ({case.get('subset','?')}): "
                  f"{level} vs GT={gt} [{match}] ({artifact['latency_ms']}ms)")
    print(f"Done: {len(cases)} artifacts -> {args.out} (mode={mode})")


if __name__ == "__main__":
    main()
