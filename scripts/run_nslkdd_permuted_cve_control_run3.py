"""Replicate 3 of the permuted-/ablated-CVE control (scripts/run_nslkdd_permuted_cve_control.py).

Identical prompt, parser, scoring, and pre-registration as runs 1-2 -- only the
output paths differ, so runs 1-2's artefacts are never overwritten. Brings the
CVE-ablation/CVE-permutation arms to the same n=3 replicate count already
reported for the primary tri-stage arms (results/nslkdd_replicate_variance.json)
and matches the convention requested for the paper's most-cited single number
("channel-independent effect").

PRE-REGISTRATION: identical to runs 1-2 (see scripts/run_nslkdd_permuted_cve_control.py
docstring). No change to arms, prompt, scoring, or interpretation rule. Fixed
before this run started: report mean/sd/min/max of b, c, p_one_sided, and
under_escalation_pct (reconciled, no_cve arm) across all 3 runs; no cherry-picking.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_nslkdd_permuted_cve_control as base  # noqa: E402

base.OUT_PATH = ROOT / "results/nslkdd_permuted_cve_control_run3.json"
base.JSONL_PATH = ROOT / "results/nslkdd_permuted_cve_control_run3_cases.jsonl"

if __name__ == "__main__":
    base.main()
