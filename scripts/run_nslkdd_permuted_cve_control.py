"""Permuted-/ablated-CVE control for the NSL-KDD leak-free safety replication.

PRE-REGISTRATION (fixed before any live call in this run):
  Motivation: round-4 review proved CVE identity in test_sample.json maps
  injectively to attack class and CVE presence separates Normal from attack
  (results/cve_presence_heuristic.json: the bare presence heuristic scores
  0% under-escalation / 100% severe recall). This control isolates how much
  of the reconciled tri-stage Haiku result (10.0% under-esc, 93.9% severe
  recall) the CVE channel contributes.

  Arms (both Haiku, claude-haiku-4-5-20251001, identical Stage-1a prompt,
  parser, and scoring as scripts/run_nslkdd_reconciled_tristage_haiku.py):
    no_cve    -- cve_block replaced by "No CVE context." on ALL 100 cases.
    permuted  -- paired_cve values shuffled across the 80 attack cases
                 (numpy default_rng seed 42), Normal cases stay CVE-free.
                 Breaks CVE-identity->class; preserves presence->attack.
  Scoring: identical metric definitions (exact, under-esc, over-esc,
  severe recall over GT in {High, Catastrophic}); reconciled level =
  max(LLM level, xgb_risk_level) per the pre-registered most-severe-wins
  rule; directional McNemar b/c vs XGBoost, one-sided sign test.
  Unparseable outputs score as the XGBoost passthrough (reconciled arm)
  and as under-escalation-if-below-GT via Very Low (LLM-only arm), the
  same convention as the primary run; unparseable counts are reported.
  Interpretation rule fixed in advance: if the no-CVE reconciled arm's
  under-escalation is within 5 percentage points of the primary run's
  10.0%, the safety result does NOT depend on the CVE channel; if it
  reverts toward XGBoost's 48.0%, the CVE channel is load-bearing.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.claude_cli_client import ClaudeCLIClient  # noqa: E402
from scripts.run_nslkdd_reconciled_tristage_haiku import (  # noqa: E402
    SYSTEM_PROMPT_PURE_LLM,
    PROMPT_PURE_LLM,
    _format_cve,
    _parse_response,
)

TEST_PATH = ROOT / "results/nslkdd_unified_rerun/test_sample.json"
OUT_PATH = ROOT / "results/nslkdd_permuted_cve_control.json"
JSONL_PATH = ROOT / "results/nslkdd_permuted_cve_control_cases.jsonl"

MODEL = "claude-haiku-4-5-20251001"
LEVELS = ["Very Low", "Low", "Medium", "High", "Catastrophic"]
IDX = {l: i for i, l in enumerate(LEVELS)}
SEVERE = {"High", "Catastrophic"}


def load_cases() -> list[dict]:
    d = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    return d if isinstance(d, list) else d["cases"]


def build_arm_cves(cases: list[dict], arm: str) -> list:
    if arm == "no_cve":
        return [None] * len(cases)
    if arm == "permuted":
        rng = np.random.default_rng(42)
        attack_idx = [i for i, c in enumerate(cases) if c.get("paired_cve")]
        cves = [cases[i]["paired_cve"] for i in attack_idx]
        perm = rng.permutation(len(cves))
        out = [None] * len(cases)
        for slot, src in zip(attack_idx, perm):
            out[slot] = cves[src]
        return out
    raise ValueError(arm)


def extract_level(parsed) -> str | None:
    if not parsed:
        return None
    lvl = parsed.get("risk_level")
    return lvl if lvl in IDX else None


def metrics(preds: list[str], cases: list[dict]) -> dict:
    n = len(cases)
    exact = under = over = 0
    sev_tot = sev_hit = 0
    for p, c in zip(preds, cases):
        gt = c["ground_truth_risk_level"]
        if p == gt:
            exact += 1
        if IDX[p] < IDX[gt]:
            under += 1
        if IDX[p] > IDX[gt]:
            over += 1
        if gt in SEVERE:
            sev_tot += 1
            if p in SEVERE:
                sev_hit += 1
    return {
        "exact_pct": 100 * exact / n,
        "under_escalation_pct": 100 * under / n,
        "over_escalation_pct": 100 * over / n,
        "severe_recall_pct": 100 * sev_hit / sev_tot,
        "severe_total": sev_tot,
    }


def mcnemar_vs_xgb(preds: list[str], cases: list[dict]) -> dict:
    b = c = 0
    for p, case in zip(preds, cases):
        gt, xgb = case["ground_truth_risk_level"], case["xgb_risk_level"]
        xgb_unsafe = IDX[xgb] < IDX[gt]
        arm_unsafe = IDX[p] < IDX[gt]
        if xgb_unsafe and not arm_unsafe:
            b += 1
        if arm_unsafe and not xgb_unsafe:
            c += 1
    from math import comb
    n = b + c
    p_one = sum(comb(n, k) for k in range(b, n + 1)) * 0.5 ** n if n else 1.0
    return {"b": b, "c": c, "p_one_sided": p_one}


def main() -> None:
    cases = load_cases()
    client = ClaudeCLIClient(model=MODEL, max_tokens=600, timeout_sec=180)
    results: dict = {"pre_registration": __doc__, "model": MODEL, "arms": {}}
    jsonl = JSONL_PATH.open("w", encoding="utf-8")

    for arm in ("no_cve", "permuted"):
        arm_cves = build_arm_cves(cases, arm)
        llm_levels: list[str] = []
        recon_levels: list[str] = []
        unparseable = 0
        t0 = time.time()
        for i, (case, cve) in enumerate(zip(cases, arm_cves)):
            prompt = PROMPT_PURE_LLM.format(
                criticality=case["criticality"],
                asset_json=json.dumps(case.get("asset", {})),
                cve_block=_format_cve(cve),
                nl_desc=case["nl_description"][:300],
            )
            try:
                raw = client.generate(SYSTEM_PROMPT_PURE_LLM, prompt) or ""
            except Exception as exc:  # noqa: BLE001
                raw = f"__error__ {exc}"
            lvl = extract_level(_parse_response(raw))
            if lvl is None:
                unparseable += 1
                llm_lvl = "Very Low"          # primary-run convention
                recon = case["xgb_risk_level"]  # ml_only_parseable passthrough
            else:
                llm_lvl = lvl
                recon = max(lvl, case["xgb_risk_level"], key=lambda x: IDX[x])
            llm_levels.append(llm_lvl)
            recon_levels.append(recon)
            jsonl.write(json.dumps({
                "arm": arm, "case_id": case["case_id"],
                "cve_id": (cve or {}).get("cve_id"),
                "llm_level": llm_lvl if lvl else None,
                "reconciled_level": recon,
                "gt": case["ground_truth_risk_level"],
                "xgb": case["xgb_risk_level"],
            }, ensure_ascii=False) + "\n")
            jsonl.flush()
            print(f"[{arm}] {i+1}/100 {case['case_id']} llm={llm_lvl if lvl else 'PARSE_FAIL'}",
                  flush=True)
        results["arms"][arm] = {
            "llm_only": metrics(llm_levels, cases),
            "reconciled": metrics(recon_levels, cases),
            "reconciled_mcnemar_vs_xgb": mcnemar_vs_xgb(recon_levels, cases),
            "llm_only_mcnemar_vs_xgb": mcnemar_vs_xgb(llm_levels, cases),
            "unparseable": unparseable,
            "elapsed_s": round(time.time() - t0, 1),
        }
        OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"[{arm}] done: {json.dumps(results['arms'][arm], indent=2)}", flush=True)

    jsonl.close()
    print("ALL DONE ->", OUT_PATH, flush=True)


if __name__ == "__main__":
    main()
