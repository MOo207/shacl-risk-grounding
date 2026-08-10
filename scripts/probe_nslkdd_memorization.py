"""NSL-KDD memorization probes for the channel-independent safety effect.

Reviewer-requested (DKE round 5, item 7 / methodology W7). Two cheap probes,
40 Haiku calls total, interpretation rules pre-registered here BEFORE running:

Probe A -- dataset identification. 20 neutral NL descriptions (the exact
strings the Stage-1a arm reads, CVE and asset context stripped) are sent with
the question "Which public benchmark dataset, if any, does this connection
record most likely come from?". Rule: if the model names NSL-KDD / KDD Cup 99
/ KDDTest in >=50% of cases, benchmark recognition is demonstrated and the
paper must downgrade the channel-independent effect to "possibly
memorisation-assisted"; below 20%, recognition is weak; between, ambiguous.

Probe B -- perturbation stability. The same 20 cases have non-diagnostic
numeric features jittered (duration +/-7%, src/dst_bytes +/-10%,
dst_host_count +/-10%; zeros stay zero; categoricals untouched), the neutral
NL is re-rendered, and the Stage-1a classification prompt is re-run. Rule:
high class-agreement with the unperturbed run (>=80%) means the inference
rides feature *patterns* rather than exact memorized rows (memorization of
row-level values not supported); low agreement means the inference is
brittle and row-memorization cannot be separated from pattern use.

Model: claude-haiku-4-5-20251001 via the Claude CLI wrapper (as all
experiments in this project; disclosed as CLI access).
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline.nfcrm.nslkdd_neutral_nl import features_to_nl_neutral  # noqa: E402
from scripts.claude_cli_client import ClaudeCLIClient  # noqa: E402

TEST_SAMPLE = ROOT / "results/nslkdd_unified_rerun/test_sample.json"
BASELINE_JSONL = ROOT / "results/nslkdd_unified_rerun/reconciled_tristage_haiku_test.jsonl"
OUT = ROOT / "results/nslkdd_memorization_probe.json"

MODEL = "claude-haiku-4-5-20251001"
N_PROBE = 20
SEED = 42

JSON_FENCE = re.compile(r"```json\s*\n?(.*?)```", re.DOTALL)

DATASET_ID_SYSTEM = (
    "You are a network-security data analyst. Answer concisely and honestly."
)
DATASET_ID_PROMPT = """Below is a natural-language rendering of a single network connection record.

{nl_desc}

Question: which public benchmark dataset, if any, does this connection record most likely come from? Answer with a single dataset name (e.g. 'NSL-KDD', 'CIC-IDS2018', 'UNSW-NB15') or 'unknown'. One line only."""

# Stage-1a classification prompt, reproduced from
# scripts/run_nslkdd_reconciled_tristage_haiku.py (verbatim), with CVE and
# asset context REMOVED so only the flow description drives the inference --
# matching what Probe B varies.
CLS_SYSTEM = """You are a GRC analyst applying NFCRM-1:2025.
The attack classification is NOT available — infer the most likely NSL-KDD attack class from the flow description.

Steps: (1) infer NSL-KDD class (Normal/DoS/Probe/R2L/U2R) from the flow description.
Output ONLY a valid JSON object inside a ```json fenced code block."""

CLS_PROMPT = """Security event (NSL-KDD network flow, attack class UNKNOWN):
- Network flow NL description: {nl_desc}

Infer the most likely NSL-KDD attack class (Normal/DoS/Probe/R2L/U2R) from the flow description alone.

```json
{{
  "inferred_attack_class": "<your inferred class>"
}}```"""

JITTER_PCT = {"duration": 0.07, "src_bytes": 0.10, "dst_bytes": 0.10,
              "dst_host_count": 0.10}


def jitter_features(raw: dict, rng: random.Random) -> dict:
    out = dict(raw)
    for k, pct in JITTER_PCT.items():
        v = out.get(k, 0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v == 0:
            continue
        jittered = v * (1.0 + rng.uniform(-pct, pct))
        out[k] = int(round(jittered)) if float(v).is_integer() else jittered
    return out


def parse_json(text: str) -> dict | None:
    m = JSON_FENCE.search(text or "")
    blob = m.group(1) if m else (text or "")
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


def main() -> None:
    rng = random.Random(SEED)
    cases = json.load(open(TEST_SAMPLE))["cases"]
    # 4 per class for coverage (20 total), file order within class.
    by_class: dict[str, list] = {}
    for c in cases:
        by_class.setdefault(c["true_attack_class"], []).append(c)
    probe_cases = [c for cls in sorted(by_class) for c in by_class[cls][:4]]
    assert len(probe_cases) == N_PROBE

    client = ClaudeCLIClient(model=MODEL, max_tokens=300, timeout_sec=180)

    KDD_NAMES = re.compile(r"nsl[- ]?kdd|kdd[- ]?(cup)?[- ]?('?9?9|99)|kddtest", re.I)

    probe_a, probe_b = [], []
    for i, c in enumerate(probe_cases, 1):
        nl = features_to_nl_neutral(c["raw_features"])

        # Probe A: dataset identification.
        resp_a = ""
        try:
            resp_a = client.generate(DATASET_ID_SYSTEM,
                                     DATASET_ID_PROMPT.format(nl_desc=nl)) or ""
        except Exception:
            pass
        named_kdd = bool(KDD_NAMES.search(resp_a))
        probe_a.append({"case_id": c["case_id"], "true_class": c["true_attack_class"],
                        "response": resp_a.strip()[:300], "named_kdd_family": named_kdd})

        # Probe B: unperturbed vs perturbed classification.
        def classify(desc: str) -> str | None:
            try:
                r = client.generate(CLS_SYSTEM, CLS_PROMPT.format(nl_desc=desc)) or ""
            except Exception:
                return None
            art = parse_json(r)
            return (art or {}).get("inferred_attack_class")

        cls_orig = classify(nl)
        nl_pert = features_to_nl_neutral(jitter_features(c["raw_features"], rng))
        cls_pert = classify(nl_pert)
        probe_b.append({"case_id": c["case_id"], "true_class": c["true_attack_class"],
                        "class_unperturbed": cls_orig, "class_perturbed": cls_pert,
                        "agree": (cls_orig is not None and cls_orig == cls_pert)})
        print(f"[{i}/{N_PROBE}] {c['case_id']}: kdd_named={named_kdd} "
              f"orig={cls_orig} pert={cls_pert}")

    n_named = sum(r["named_kdd_family"] for r in probe_a)
    n_valid_b = sum(1 for r in probe_b if r["class_unperturbed"] is not None
                    and r["class_perturbed"] is not None)
    n_agree = sum(r["agree"] for r in probe_b)

    out = {
        "description": __doc__.strip(),
        "model": MODEL,
        "n_probe_cases": N_PROBE,
        "probe_a_dataset_identification": {
            "n_named_kdd_family": n_named,
            "rate": n_named / N_PROBE,
            "records": probe_a,
        },
        "probe_b_perturbation_stability": {
            "n_both_parsed": n_valid_b,
            "n_agree": n_agree,
            "agreement_rate_over_parsed": (n_agree / n_valid_b) if n_valid_b else None,
            "jitter_spec": JITTER_PCT,
            "records": probe_b,
        },
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nProbe A: {n_named}/{N_PROBE} named the KDD family")
    print(f"Probe B: {n_agree}/{n_valid_b} agreement over parsed pairs")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
