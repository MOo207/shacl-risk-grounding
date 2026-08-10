"""Draw the practitioner-evaluation sample and emit blinded rater packets.

Design
------
The paper's largest open gap is that every coverage and traceability figure
confirms implementation correctness, not audit acceptability: no GRC
practitioner has judged whether a conformant artefact is actually usable. This
builds the instrument for that study.

Sampling frame: the stored generated artefacts from Experiment 2 (risk
scenarios, N=99) and Experiment 3 (control recommendations, N=99).

Stratification is by real SHACL verdict, taken from
results/generation_shacl_revalidation.json -- half the drawn artefacts conform,
half were rejected. Raters are blinded to the verdict. This turns the study into
a test of the question the paper cannot currently answer: does SHACL conformance
predict practitioner-judged quality, or is it orthogonal to it?

Pre-registered primary analysis (see scripts/analyze_human_eval.py):
    H1: conforming artefacts receive higher mean overall ratings than rejected
        ones (Mann-Whitney U, one-sided, alpha=0.05).
    H0: conformance is orthogonal to practitioner-judged quality.

Emits one CSV per rater with artefacts in an independently randomised order, a
README with instructions, and a key file that must not be shown to raters.

Usage: python scripts/build_human_eval_sample.py [--per-stratum 10] [--raters 5]
Output: human_eval/sample/
"""
import argparse
import csv
import json
import os
import random
import textwrap

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR = os.path.join(BASE_DIR, "human_eval", "sample")

DIMENSIONS = [
    ("clause_correctness",
     "Is the NFCRM-1:2025 clause reference appropriate for the described situation?"),
    ("audit_completeness",
     "Would this text support a compliance audit trail without further explanation?"),
    ("control_actionability",
     "Is the recommended control practical and appropriate for the described risk?"),
]
ACCEPT_FIELD = ("accept_into_register",
                "Would you accept this into a risk register without edits? (yes/no)")


def load_verdicts():
    path = os.path.join(RESULTS_DIR, "generation_shacl_revalidation.json")
    if not os.path.exists(path):
        raise SystemExit("Run scripts/revalidate_generation_shacl.py first.")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for exp in data["experiments"]:
        key = "exp2" if "risk scenario" in exp["experiment"] else "exp3"
        out[key] = {c["index"]: c for c in exp["per_case"]}
    return out


def load_results(fname):
    with open(os.path.join(RESULTS_DIR, fname), encoding="utf-8") as f:
        return json.load(f)["results"]


def render_exp2(parsed):
    controls = ", ".join(str(c) for c in parsed.get("recommended_controls", []) or []) or "(none)"
    return textwrap.dedent(f"""\
        Scenario title : {parsed.get('scenario_title', '(missing)')}
        Affected asset : {parsed.get('affected_asset', '(missing)')}
        Threat vector  : {parsed.get('threat_vector', '(missing)')}
        Impact level   : {parsed.get('impact_level', '(missing)')}
        NFCRM clause   : {parsed.get('nfcrm_clause', '(missing)')}
        Controls       : {controls}

        Description:
        {parsed.get('description', '(missing)')}""")


def render_exp3(parsed):
    lines = [f"Target asset : {parsed.get('target_asset', '(missing)')}",
             f"Threat class : {parsed.get('threat_class', '(missing)')}",
             "", "Recommendations:"]
    recs = parsed.get("recommendations", []) or []
    if not recs:
        lines.append("  (none)")
    for r in recs:
        if not isinstance(r, dict):
            lines.append(f"  - (malformed entry: {r!r})")
            continue
        lines.append(f"  - {r.get('control_clause_id', '(no clause)')}"
                     f" {r.get('control_name', '')}".rstrip())
        lines.append(f"      justification: {r.get('justification', '(none)')}")
    return "\n".join(lines)


def draw(experiment, fname, renderer, verdicts, per_stratum, rng):
    results = load_results(fname)
    v = verdicts[experiment]
    conforming = [i for i in range(len(results)) if v.get(i, {}).get("shacl_conforms")]
    rejected = [i for i in range(len(results)) if not v.get(i, {}).get("shacl_conforms")]

    # Rejected strata are small (4 for exp2, 28 for exp3); take all of them when
    # fewer than per_stratum exist and record the shortfall rather than
    # silently rebalancing.
    take_rej = min(per_stratum, len(rejected))
    take_con = min(per_stratum, len(conforming))
    picked = ([(i, True) for i in rng.sample(conforming, take_con)] +
              [(i, False) for i in rng.sample(rejected, take_rej)])

    items = []
    for idx, conforms in picked:
        parsed = results[idx].get("parsed_response")
        body = renderer(parsed) if isinstance(parsed, dict) else \
            f"(model returned a non-object response: {json.dumps(parsed)[:400]})"
        items.append({
            "experiment": experiment,
            "source_index": idx,
            "asset_id": results[idx].get("asset_id"),
            "shacl_conforms": conforms,
            "artefact_text": body,
        })
    return items, {"conforming_available": len(conforming),
                   "rejected_available": len(rejected),
                   "conforming_drawn": take_con, "rejected_drawn": take_rej}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=10)
    ap.add_argument("--raters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    verdicts = load_verdicts()

    items2, meta2 = draw("exp2", "llm_risk_generation.json", render_exp2,
                         verdicts, args.per_stratum, rng)
    items3, meta3 = draw("exp3", "llm_control_recommendation.json", render_exp3,
                         verdicts, args.per_stratum, rng)
    items = items2 + items3
    for n, it in enumerate(items, 1):
        it["artefact_id"] = f"A{n:03d}"

    os.makedirs(OUT_DIR, exist_ok=True)

    # Key file -- verdicts and provenance. Never given to raters.
    key = {
        "seed": args.seed,
        "per_stratum_requested": args.per_stratum,
        "strata": {"exp2": meta2, "exp3": meta3},
        "n_artefacts": len(items),
        "blinding": "Raters receive artefact text only. SHACL verdict, experiment "
                    "of origin and source index appear here and nowhere in the packets.",
        "pre_registered_hypothesis":
            "H1 (one-sided): artefacts that conform to the SHACL shape library "
            "receive higher mean overall practitioner ratings than rejected ones. "
            "Primary test Mann-Whitney U on the mean of the three Likert "
            "dimensions, alpha=0.05. Secondary: Krippendorff's alpha (ordinal) "
            "for inter-rater reliability, reported regardless of H1 outcome.",
        "items": [{k: it[k] for k in
                   ("artefact_id", "experiment", "source_index", "asset_id",
                    "shacl_conforms")} for it in items],
    }
    with open(os.path.join(OUT_DIR, "KEY_do_not_distribute.json"), "w",
              encoding="utf-8") as f:
        json.dump(key, f, indent=2)

    # Artefact texts, shared across raters.
    with open(os.path.join(OUT_DIR, "artefacts.md"), "w", encoding="utf-8") as f:
        f.write("# Artefacts for evaluation\n\n")
        f.write("Each artefact was generated automatically by the "
                "INTERSYMBOLIC-GRC pipeline. Rate each one in your rating "
                "sheet using the artefact ID shown.\n\n")
        for it in items:
            f.write(f"\n---\n\n## {it['artefact_id']}\n\n```\n{it['artefact_text']}\n```\n")

    # Per-rater sheets, each in its own randomised order.
    for r in range(1, args.raters + 1):
        order = items[:]
        random.Random(args.seed + r).shuffle(order)
        path = os.path.join(OUT_DIR, f"rater_{r:02d}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["artefact_id"] + [d for d, _ in DIMENSIONS] +
                       [ACCEPT_FIELD[0], "free_text_comment"])
            for it in order:
                w.writerow([it["artefact_id"], "", "", "", "", ""])

    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""\
            # Practitioner evaluation packet

            {len(items)} artefacts, {args.raters} raters, fully crossed: every
            rater rates every artefact. Expect 45-60 minutes.

            ## What to do

            1. Open `artefacts.md`.
            2. Open your own sheet (`rater_01.csv` ... `rater_{args.raters:02d}.csv`).
               Artefacts appear in a different order in each sheet; work down
               your own sheet in the order given.
            3. For each artefact fill four columns.

            ## Scales

            Three dimensions, 1-5:

            | Score | Meaning |
            |-------|---------|
            | 1 | Completely incorrect / unusable |
            | 2 | Mostly incorrect, major issues |
            | 3 | Partially correct, significant issues |
            | 4 | Mostly correct, minor issues |
            | 5 | Correct and audit-ready |

            {chr(10).join(f'- **{d}** - {q}' for d, q in DIMENSIONS)}

            Then one binary judgement:

            - **{ACCEPT_FIELD[0]}** - {ACCEPT_FIELD[1]}

            `free_text_comment` is optional and read qualitatively.

            ## Notes

            - Rate the artefact as written. Do not assume missing context.
            - If an artefact is malformed, rate it as you find it; that is data.
            - Do not confer with other raters until all sheets are returned.
            - Return your sheet unchanged in name and column order.

            ## Analysis

            Run `python scripts/analyze_human_eval.py human_eval/sample/returned/`
            once sheets are back. The primary hypothesis and reliability
            statistic are pre-registered in `KEY_do_not_distribute.json`, which
            is written before any rating occurs and must not be circulated.
            """))

    print(f"{len(items)} artefacts -> {OUT_DIR}")
    print(f"  exp2 strata: {meta2}")
    print(f"  exp3 strata: {meta3}")
    print(f"  {args.raters} rater sheets, README.md, artefacts.md, "
          f"KEY_do_not_distribute.json")
    if meta2["rejected_drawn"] < args.per_stratum or meta3["rejected_drawn"] < args.per_stratum:
        print("  NOTE: a rejected stratum was smaller than requested; "
              "drawn counts recorded in the key file.")


if __name__ == "__main__":
    main()
