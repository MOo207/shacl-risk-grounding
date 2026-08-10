"""Rule+LLM arm — Claude Code CLI + SHACL-grounded risk artifact generation."""
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

CRITICAL RULE — §6.9 Risk Level Lookup Table (mandatory, do not deviate):
Use the RF-predicted attack class AND asset criticality to look up the risk level EXACTLY:

  Attack class = Benign → risk_level = "Very Low"  (regardless of criticality or CVEs)

  Attack class = BruteForce:
    LOW criticality   → "Medium"
    MEDIUM criticality → "High"
    HIGH criticality   → "Catastrophic"
    CRITICAL criticality → "Catastrophic"
    Unknown/masked → "High" (default Medium criticality)

  Attack class = DoS:
    LOW criticality   → "Medium"
    MEDIUM criticality → "High"
    HIGH criticality   → "Catastrophic"
    CRITICAL criticality → "Catastrophic"
    Unknown/masked → "High"

  Attack class = DDoS:
    LOW criticality   → "Medium"
    MEDIUM criticality → "High"
    HIGH criticality   → "Catastrophic"
    CRITICAL criticality → "Catastrophic"
    Unknown/masked → "High"

  Attack class = Infiltration:
    LOW criticality   → "Low"
    MEDIUM criticality → "Medium"
    HIGH criticality   → "Medium"
    CRITICAL criticality → "High"
    Unknown/masked → "Medium"

ARG neighborhood CVEs → use ONLY for selecting which control to recommend and for the narrative. They do NOT change the risk level.
The paired CVE description → use ONLY for narrative and control justification.

Output ONLY a valid JSON object inside a ```json fenced code block."""

PROMPT_TEMPLATE = """Security event:
- RF-predicted attack class: {attack_class}
- Asset: {asset_json}
- Currently-applied controls: {controls_json}
- Masked/unknown fields: {masked}

CVE context (use for control recommendation and narrative only):
{cve_block}

ARG neighborhood (use for control selection only, NOT for risk level):
{arg_json}

Apply the §6.9 risk matrix from your system instructions:
1. Map attack class → likelihood level
2. Map asset criticality → impact level
3. Look up likelihood × impact → risk_level
4. Select the most relevant NFCRM control for the attack type
5. Write a 2-4 sentence narrative citing the attack class, asset criticality, and CVE

Required output (JSON inside ```json fence):
{{
  "risk_level": "Very Low | Low | Medium | High | Catastrophic",
  "recommended_control_id": "<NFCRM-1:2025 control reference e.g. SS6.7-AC-1>",
  "nfcrm_clauses": ["§6.9", "<other §references>"],
  "narrative": "<cite attack class + asset criticality + CVE in 2-4 sentences>",
  "evidence_refs": ["<attack_class>", "<cve_id if present>", "<asset_id>"]
}}"""


def _format_cve_block(cve: Optional[dict]) -> str:
    if not cve:
        return "No vulnerability context paired. Reason about absence of CVE evidence."
    lines = [
        f"- CVE: {cve.get('cve_id')}",
        f"- Vendor/Product: {cve.get('vendor')} / {cve.get('product')}",
        f"- CVSS v3 base score: {cve.get('cvss_v3')}",
        f"- KEV description: {cve.get('shortDescription')}",
        f"- KEV required action: {cve.get('requiredAction')}",
    ]
    return "\n".join(lines)


def build_prompt(case: dict[str, Any]) -> str:
    masked = case.get("masked_fields", []) or []
    if "true_attack_class" in masked:
        attack_class_str = (
            "MASKED — not available. "
            "Infer the likely risk level from the CVE CVSS score, "
            "KEV description, and asset criticality using the §6.9 matrix."
        )
    else:
        attack_class_str = case["true_attack_class"]

    asset = dict(case.get("asset", {}))
    if "asset_criticality" in masked:
        asset.pop("criticality", None)

    return PROMPT_TEMPLATE.format(
        attack_class=attack_class_str,
        asset_json=json.dumps(asset),
        arg_json=json.dumps(case.get("arg_neighborhood", {}))[:600],
        controls_json=json.dumps(case.get("controls", [])),
        masked=masked or "none",
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
    """Structural SHACL-style validation without pyshacl dependency."""
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
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    # Try generate(system, user) first, fall back to generate(prompt)
    try:
        response = client.generate(SYSTEM_PROMPT, prompt) or ""
    except TypeError:
        try:
            response = client.generate(full_prompt) or ""
        except Exception:
            response = ""

    try:
        artifact = parse_llm_response(response)
    except (json.JSONDecodeError, AttributeError, ValueError):
        return {
            "case_id": case["case_id"], "arm": "rule_llm",
            "risk_level": "", "recommended_control_id": "",
            "nfcrm_clauses": [], "narrative": "", "evidence_refs": [],
            "parse_error": True, "raw_response": response[:500],
            "shacl_violations": [], "shacl_retried": 0,
        }

    artifact["case_id"] = case["case_id"]
    artifact["arm"] = "rule_llm"
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

    # Retry with violation context
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
        new_artifact["arm"] = "rule_llm"
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
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-per-class", type=int, default=None,
                    help="Take at most N cases per (subset, attack_class) bucket (stratified).")
    ap.add_argument("--start-offset", type=int, default=0,
                    help="Skip first N cases in the stratified list (resume support).")
    ap.add_argument("--max-cases", type=int, default=None,
                    help="Process at most N cases (use with --start-offset for batching).")
    args = ap.parse_args()

    all_cases = json.loads(Path(args.test_set).read_text())["cases"]
    cases = stratified_slice(all_cases, args.n_per_class)
    cases = cases[args.start_offset:]
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    model = args.model or "claude-sonnet-4-6"
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
