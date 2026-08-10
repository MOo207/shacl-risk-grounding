"""Independent re-implementation check of the NSL-KDD Stage-1a risk_level parser.

Motivation: every published NSL-KDD safety number (under-escalation, severe
recall, McNemar b/c) is scored by a single parser, written and run by the
same person who wrote the pipeline. A reviewer cannot tell whether the score
reflects the model's actual output or a permissive/buggy extractor. This
script is a *second*, independently-written extractor -- different control
flow, different library calls, no shared code with
scripts/run_nslkdd_reconciled_tristage_haiku.py::_parse_response -- run
over the raw LLM text already stored on disk (raw_response_stage1a field).
Agreement rate between the two parsers on llm_risk_level is the credibility
number; it does not require any new API calls.

Independent extraction method: scan the raw text with json.JSONDecoder.raw_decode
at every '{' position (not just inside a ```json fence) until a dict containing
a "risk_level" key parses successfully; normalise against the fixed level set.
This deliberately does NOT use the fenced-block regex the original parser uses,
so a bug/leniency specific to that regex would show up as a disagreement here.

Scope: the three files with raw text saved (raw_response_stage1a) --
reconciled_tristage_{haiku,sonnet}_test.jsonl (run 1) and their run2/run3
replicates under results/nslkdd_unified_rerun/replicates/. The permuted-CVE
control files do not store raw text (see run_nslkdd_permuted_cve_control*.py)
and are out of scope for this check.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/nslkdd_unified_rerun"
REP = BASE / "replicates"

LEVELS = {"Very Low", "Low", "Medium", "High", "Catastrophic"}

FILES = {
    "haiku_run1": BASE / "reconciled_tristage_haiku_test.jsonl",
    "haiku_run2": REP / "reconciled_tristage_haiku_test_run2.jsonl",
    "haiku_run3": REP / "reconciled_tristage_haiku_test_run3.jsonl",
    "sonnet_run1": BASE / "reconciled_tristage_sonnet_test.jsonl",
    "sonnet_run2": REP / "reconciled_tristage_sonnet_test_run2.jsonl",
    "sonnet_run3": REP / "reconciled_tristage_sonnet_test_run3.jsonl",
}


def independent_extract(raw: str) -> str | None:
    """Second implementation: brace-scan + raw_decode, no fence regex."""
    if not raw:
        return None
    dec = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(raw, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "risk_level" in obj:
            lvl = obj["risk_level"]
            if isinstance(lvl, str):
                lvl = lvl.strip()
                if lvl in LEVELS:
                    return lvl
    return None


def main() -> None:
    out = {"description": __doc__.strip(), "files": {}}
    total_n = total_agree = total_both_none = total_disagree = 0
    disagreements = []

    for tag, path in FILES.items():
        if not path.exists():
            print(f"MISSING {path}")
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        n = agree = both_none = disagree = 0
        for r in rows:
            raw = r.get("raw_response_stage1a", "") or ""
            original = r.get("llm_risk_level") or None  # normalise "" (empty-raw parse fail) to None
            independent = independent_extract(raw)
            n += 1
            if original is None and independent is None:
                both_none += 1
                agree += 1
            elif original == independent:
                agree += 1
            else:
                disagree += 1
                disagreements.append({
                    "file": tag, "case_id": r.get("case_id"),
                    "original_parser": original, "independent_parser": independent,
                })
        out["files"][tag] = {
            "n": n, "agree": agree, "disagree": disagree,
            "both_none": both_none,
            "agreement_pct": round(100 * agree / n, 2) if n else None,
        }
        total_n += n
        total_agree += agree
        total_disagree += disagree
        total_both_none += both_none
        print(f"{tag}: {agree}/{n} agree ({100*agree/n:.1f}%), {disagree} disagree")

    out["overall"] = {
        "n": total_n, "agree": total_agree, "disagree": total_disagree,
        "both_none": total_both_none,
        "agreement_pct": round(100 * total_agree / total_n, 2) if total_n else None,
    }
    out["disagreements"] = disagreements

    out_path = ROOT / "results/nslkdd_parser_agreement_check.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOVERALL: {total_agree}/{total_n} = {100*total_agree/total_n:.2f}% agreement")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
