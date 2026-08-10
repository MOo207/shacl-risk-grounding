"""Replicate 2 of the permuted-/ablated-CVE control (scripts/run_nslkdd_permuted_cve_control.py).

Identical prompt, parser, scoring, and pre-registration as run 1 -- only the
output paths differ, so run 1's artefact is never overwritten. Exists to give
the CVE-ablation/CVE-permutation arms the same 3-run variance quantification
Table~\\ref{tab:leakfree} already reports for the two primary tri-stage arms
(reviewer-requested: these arms currently carry the paper's most-cited single
number, "channel-independent effect," on a single, temperature-uncontrolled
run).

PRE-REGISTRATION: identical to run 1 (see scripts/run_nslkdd_permuted_cve_control.py
docstring). No change to arms, prompt, scoring, or interpretation rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_nslkdd_permuted_cve_control as base  # noqa: E402

base.OUT_PATH = ROOT / "results/nslkdd_permuted_cve_control_run2.json"
base.JSONL_PATH = ROOT / "results/nslkdd_permuted_cve_control_run2_cases.jsonl"

if __name__ == "__main__":
    base.main()
