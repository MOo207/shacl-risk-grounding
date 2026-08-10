"""Thesis evidence audit.

Read-only diagnostic: classifies result files / scripts as
core / superseded / tangential / orphan / infrastructure and
flags stale numeric claims in the thesis.

Outputs:
    audit/inventory.md   - human-readable report
    audit/inventory.json - machine-readable mirror
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THESIS_PATH = REPO_ROOT / "thesis" / "INTERSYMBOLIC-GRC_Thesis.tex"
RESULTS_DIR = REPO_ROOT / "results"
SCRIPTS_DIR = REPO_ROOT / "scripts"
FIGURES_DIR = REPO_ROOT / "thesis" / "figures"
AUDIT_DIR = REPO_ROOT / "audit"
INVENTORY_MD = AUDIT_DIR / "inventory.md"
INVENTORY_JSON = AUDIT_DIR / "inventory.json"
CLAIM_MAPPING_PATH = AUDIT_DIR / "claim_mapping.yml"

# Window (in lines) around a thesis citation to extract numeric claims.
CITATION_WINDOW = 25

# Tolerance for numeric matching.
PCT_TOLERANCE = 0.005   # for percentages: 87.70 vs 87.71 OK
FRAC_TOLERANCE = 0.001  # for fractions: 0.880 vs 0.881 OK
INT_TOLERANCE = 0       # exact for integers


def flatten_json_numerics(obj, prefix: str = ""):
    """Yield (json_path, numeric_value) for each numeric leaf in obj.

    Strings, booleans, and None are skipped. Dicts use dotted paths;
    lists use bracket indexing.
    """
    if isinstance(obj, dict):
        for key, val in obj.items():
            sub = f"{prefix}.{key}" if prefix else key
            yield from flatten_json_numerics(val, sub)
    elif isinstance(obj, list):
        for idx, val in enumerate(obj):
            sub = f"{prefix}[{idx}]"
            yield from flatten_json_numerics(val, sub)
    elif isinstance(obj, bool):
        return  # bools are int subclass — skip explicitly
    elif isinstance(obj, (int, float)):
        yield (prefix, obj)


def _filename_patterns(filename: str) -> list[re.Pattern]:
    """Patterns that match `filename` in thesis or script text."""
    bare = re.escape(filename)
    # LaTeX-escapes underscores: foo_bar.json -> foo\_bar.json
    latex_escaped = re.escape(filename.replace("_", r"\_"))
    return [
        re.compile(bare),
        re.compile(latex_escaped),
    ]


def build_reference_graph(
    thesis_text: str,
    filenames: list[str],
    scripts_text: dict[str, str],
) -> dict[str, dict]:
    """For each filename, find which thesis lines and which scripts reference it.

    Returns: {filename: {"thesis_lines": [int], "script_refs": [path]}}
    """
    thesis_lines = thesis_text.splitlines()
    refs: dict[str, dict] = {}

    for fname in filenames:
        patterns = _filename_patterns(fname)

        # Pass over thesis line by line so we can record line numbers.
        thesis_hits: list[int] = []
        for lineno, line in enumerate(thesis_lines, start=1):
            if any(p.search(line) for p in patterns):
                thesis_hits.append(lineno)

        # Pass over each script as a single string (line numbers not needed).
        script_hits: list[str] = []
        for script_path, text in scripts_text.items():
            if any(p.search(text) for p in patterns):
                script_hits.append(script_path)

        refs[fname] = {
            "thesis_lines": thesis_hits,
            "script_refs": script_hits,
        }

    return refs


# Order matters: percent regex must run before bare-decimal regex,
# else "87.70" inside "87.70\%" will be consumed as a fraction.
_NUM_PATTERNS = [
    # 87.70\% or 87.70%
    (re.compile(r"(\d+\.\d+)\\?%"), "percent"),
    # N=2{,}122 or N=2,122 or N=2122
    (re.compile(r"N\s*=\s*(\d{1,3}(?:[,{}]+\d{3})*|\d+)"), "ncount"),
    # "423 edges", "32 CVEs" (number BEFORE keyword)
    (re.compile(r"\b(\d+)\s*(?:nodes?|edges?|CVEs?|controls?|clauses?)\b", re.IGNORECASE), "graph_count"),
    # "Total nodes: 111", "CVE: 19", "NFCRM Control: 18" (number AFTER keyword)
    (re.compile(r"(?:nodes?|edges?|CVEs?|controls?|clauses?)[:\s]+(\d+)", re.IGNORECASE), "graph_count"),
    # LaTeX tabular cell: "CVE & 19 &" or "NFCRM Control & 18 &"
    (re.compile(r"(?:nodes?|edges?|CVEs?|controls?|clauses?)\s*&\s*(\d+)", re.IGNORECASE), "graph_count"),
    # bare fraction 0.896 or 0.74
    (re.compile(r"\b(0\.\d+)\b"), "fraction"),
    # bare integer with thousands separator: 2,122 or 2{,}122
    (re.compile(r"\b(\d{1,3}(?:[,{}]+\d{3})+)\b"), "thousands_int"),
]


def _normalize(token: str, kind: str) -> tuple[str, float]:
    """Return (display_token, normalized_numeric_value)."""
    if kind == "percent":
        return (f"{token}%", float(token) / 100.0)
    if kind == "ncount" or kind == "thousands_int":
        cleaned = token.replace("{,}", "").replace(",", "")
        return (cleaned, float(int(cleaned)))
    if kind == "graph_count":
        return (token, float(int(token)))
    if kind == "fraction":
        return (token, float(token))
    return (token, float(token))


def extract_thesis_numbers(text: str) -> list[tuple[str, float]]:
    """Extract numeric tokens from thesis text.

    Returns list of (display_token, normalized_value).
    Same numeric value extracted by multiple patterns is deduplicated.
    """
    found: dict[float, str] = {}
    for pattern, kind in _NUM_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1)
            display, value = _normalize(raw, kind)
            # Don't overwrite — first pattern to match wins.
            found.setdefault(value, display)
    return [(display, value) for value, display in found.items()]


def _matches_any(value: float, json_leaves: dict[str, float]) -> tuple[bool, str | None, float]:
    """Check if value matches any leaf within tolerance.

    Returns (matched, closest_path, abs_distance).
    """
    closest_path = None
    closest_dist = float("inf")
    for path, leaf_val in json_leaves.items():
        dist = abs(value - leaf_val)
        # Pick tolerance based on value magnitude.
        if value == int(value) and leaf_val == int(leaf_val):
            tol = INT_TOLERANCE
        elif 0 < value < 1:
            tol = FRAC_TOLERANCE
        else:
            tol = PCT_TOLERANCE
        if dist <= tol:
            return (True, path, dist)
        if dist < closest_dist:
            closest_dist = dist
            closest_path = path
    return (False, closest_path, closest_dist)


def cross_check_numbers(
    thesis_text: str,
    json_leaves: dict[str, float],
) -> list[dict]:
    """Flag thesis numbers that don't match any JSON leaf.

    Returns list of flag dicts with keys: thesis_value, closest_json_path,
    closest_json_value, distance.
    """
    flags = []
    for token, value in extract_thesis_numbers(thesis_text):
        matched, closest_path, dist = _matches_any(value, json_leaves)
        if not matched:
            flags.append({
                "thesis_value": token,
                "closest_json_path": closest_path,
                "closest_json_value": json_leaves[closest_path] if closest_path else None,
                "distance": dist,
            })
    return flags


# Default claim mapping by filename pattern. Override per-file via audit/claim_mapping.yml.
# Claim numbers map to the 6 main claims in the spec.
_CLAIM_PATTERNS: list[tuple[re.Pattern, list[int]]] = [
    (re.compile(r"ablation"), [1, 6]),
    (re.compile(r"statistical_tests|kfold|threshold_calibration"), [1, 6]),
    (re.compile(r"slm.*nl|slm_nslkdd"), [2]),
    (re.compile(r"slm_nl_feature|cicids_slm|nslkdd_slm"), [2]),  # superseded SLM variants still tied to claim 2
    (re.compile(r"^rf_baseline|cleaned_rf|nslkdd_rf|nslkdd_xgb|xgb_baseline"), [1]),
    (re.compile(r"rule_baseline|nslkdd_rule"), [1]),
    (re.compile(r"arg|shacl"), [3, 4]),
    (re.compile(r"grc_metrics"), [4]),
    (re.compile(r"intersymbolic_explanation|llm_risk|llm_control"), [5]),
    (re.compile(r"ssh|in_inference"), [1]),
    (re.compile(r"shap|butterfly|e2e_pipeline_benchmark|slm_multisource_demo|slm_explanations"), []),
]

# Utility scripts (not result-producers).
_UTILITY_PATTERNS = [
    re.compile(r"download_"),
    re.compile(r"prepare_"),
    re.compile(r"validate_dataset"),
    re.compile(r"enrich_nvd"),
    re.compile(r"generate_cmdb"),
    re.compile(r"llm_client"),
    re.compile(r"verify_"),
    re.compile(r"fix_fabricated"),
    re.compile(r"calibrate_thresholds"),
    re.compile(r"run_all_experiments"),
]

# Stem families for supersession detection. Sibling stems collapse to the same family.
_SUPERSESSION_FAMILIES = [
    {"cicids_slm", "cicids_slm_v2", "slm_nl_classification", "slm_nl_feature_v2"},
    {"slm_nslkdd_nl_classification", "nslkdd_slm", "nslkdd_slm_explanations"},
    {"rf_baseline", "cleaned_rf_baseline"},
    {"statistical_tests", "statistical_tests_v2"},
    {"ablation_study_v2", "ablation_validation"},
    {"slm_explanations_v2", "slm_multisource_demo"},
    {"in_inference_ssh", "ssh_feature_weighting"},
]


def infer_claim_support(path: str) -> list[int]:
    """Return list of claim numbers (1-6) this file supports based on filename patterns."""
    fname = Path(path).name.lower()
    claims: set[int] = set()
    for pattern, claim_list in _CLAIM_PATTERNS:
        if pattern.search(fname):
            claims.update(claim_list)
    return sorted(claims)


def is_utility_script(path: str) -> bool:
    fname = Path(path).name
    return any(p.search(fname) for p in _UTILITY_PATTERNS)


def classify_file(
    path: str,
    thesis_lines: list[int],
    script_refs: list[str],
    claim_support: list[int],
    siblings: list[dict],
    is_utility: bool = False,
) -> dict:
    """Return classification dict for a single file.

    siblings: list of {"path": str, "thesis_lines": list[int]} for other
              files in the same supersession family.
    """
    if is_utility:
        return {"category": "infrastructure", "claim_support": claim_support}

    # Look for a sibling that IS referenced in thesis to detect supersession.
    cited_sibling = next(
        (s for s in siblings if s["thesis_lines"]),
        None,
    )

    has_thesis_ref = bool(thesis_lines)
    has_script_ref = bool(script_refs)
    has_claim = bool(claim_support)

    if cited_sibling and not has_thesis_ref:
        return {
            "category": "superseded",
            "superseded_by": cited_sibling["path"],
            "confidence": "high" if claim_support else "medium",
            "claim_support": claim_support,
        }

    if has_claim and (has_thesis_ref or has_script_ref):
        return {"category": "core", "claim_support": claim_support}

    if has_thesis_ref and not has_claim:
        return {"category": "tangential", "claim_support": []}

    if not has_thesis_ref and not has_script_ref:
        return {"category": "orphan", "claim_support": claim_support}

    # Fallback: referenced in scripts but no thesis ref and no claim.
    return {"category": "orphan", "claim_support": claim_support}


_CLAIM_LABELS = {
    1: "Tri-stage preserves ML accuracy with GRC traceability",
    2: "SLM classifies network flows from NL descriptions when signatures are discriminative",
    3: "ARG integrates multi-source data into a queryable graph",
    4: "SHACL enforces NFCRM-1:2025 clause coverage",
    5: "Layer 4 generates auditor-readable NL explanations via Claude Haiku 4.5",
    6: "Pre-inference enrichment is statistically indistinguishable from pure ML, study adequately powered",
}


def _files_in(audit_data, category):
    return [f for f in audit_data["files"] if f["category"] == category]


def render_inventory_md(audit_data: dict) -> str:
    """Render the audit data dict as a markdown report string."""
    summary = audit_data["summary"]
    lines = [
        "# Thesis Evidence Audit",
        "",
        f"_Generated: {audit_data['generated']}_",
        "",
        "## Summary",
        "",
        f"- Core: **{summary['core']}**",
        f"- Superseded: **{summary['superseded']}**",
        f"- Tangential: **{summary['tangential']}**",
        f"- Orphan: **{summary['orphan']}**",
        f"- Infrastructure: **{summary['infrastructure']}**",
        f"- Stale-number flags: **{summary['stale_numbers']}**",
        "",
        "## Stale numbers (HIGHEST PRIORITY — fix in Phase 1)",
        "",
    ]

    if not audit_data["stale_numbers"]:
        lines.append("_No stale numbers detected._")
    else:
        lines.append("| Thesis line | Thesis value | Cited file | JSON path | JSON value | Recommendation |")
        lines.append("|---|---|---|---|---|---|")
        for s in audit_data["stale_numbers"]:
            lines.append(
                f"| L{s['thesis_line']} | `{s['thesis_value']}` | `{s['json_file']}` | "
                f"`{s.get('json_path', '?')}` | `{s.get('json_value', '?')}` | {s['recommendation']} |"
            )
    lines.append("")

    # Core grouped by claim
    lines.append("## Core files (KEEP)")
    lines.append("")
    core_by_claim: dict[int, list] = defaultdict(list)
    for f in _files_in(audit_data, "core"):
        for c in f["claim_support"] or [0]:
            core_by_claim[c].append(f)
    for claim_num in sorted(core_by_claim):
        label = _CLAIM_LABELS.get(claim_num, "Uncategorised")
        lines.append(f"### Claim {claim_num} — {label}")
        for f in core_by_claim[claim_num]:
            refs = f"thesis L{','.join(map(str, f['thesis_refs']))}" if f['thesis_refs'] else "no thesis ref"
            lines.append(f"- `{f['path']}` — {refs}")
        lines.append("")

    # Superseded
    lines.append("## Superseded → Archive")
    lines.append("")
    for f in _files_in(audit_data, "superseded"):
        lines.append(
            f"- `{f['path']}` — superseded by `{f['superseded_by']}` "
            f"(confidence: {f['confidence']})"
        )
    lines.append("")

    # Tangential
    lines.append("## Tangential → Archive (aggressive prune)")
    lines.append("")
    tangentials = _files_in(audit_data, "tangential")
    if not tangentials:
        lines.append("_None._")
    for f in tangentials:
        refs = ",".join(f"L{n}" for n in f["thesis_refs"])
        lines.append(f"- `{f['path']}` — supports no main claim; cited at thesis {refs}. Recommend archiving + removing citing paragraph.")
    lines.append("")

    # Orphan
    lines.append("## Orphan → Archive")
    lines.append("")
    orphans = _files_in(audit_data, "orphan")
    if not orphans:
        lines.append("_None._")
    for f in orphans:
        lines.append(f"- `{f['path']}` — referenced nowhere")
    lines.append("")

    # Infrastructure
    lines.append("## Infrastructure (KEEP, no action)")
    lines.append("")
    for f in _files_in(audit_data, "infrastructure"):
        lines.append(f"- `{f['path']}`")
    lines.append("")

    return "\n".join(lines)


# Match a line like:  path/to/file.json: [1, 2]
_CLAIM_OVERRIDE_LINE = re.compile(r"^\s*([^:#][^:]*):\s*\[\s*((?:\d+\s*,?\s*)*)\s*\]\s*$")


def load_claim_mapping_overrides(path: Path) -> dict[str, list[int]]:
    """Load a tiny YAML subset: lines of form `path: [1, 2]`.

    Stdlib-only (no PyYAML). Comments (# ...) and blank lines ignored.
    Returns empty dict if file is absent.
    """
    if not path.exists():
        return {}
    overrides: dict[str, list[int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _CLAIM_OVERRIDE_LINE.match(line)
        if not m:
            continue
        key = m.group(1).strip()
        nums_text = m.group(2).strip()
        nums = [int(n.strip()) for n in nums_text.split(",") if n.strip()]
        overrides[key] = nums
    return overrides


def write_inventory_json(audit_data: dict, path: Path) -> None:
    path.write_text(json.dumps(audit_data, indent=2, default=str), encoding="utf-8")


def write_inventory_md(audit_data: dict, path: Path) -> None:
    path.write_text(render_inventory_md(audit_data), encoding="utf-8")


def _supersession_siblings(filename_stem: str) -> set[str]:
    """Return the family of stems this file belongs to (or {} if no family)."""
    for family in _SUPERSESSION_FAMILIES:
        if filename_stem in family:
            return family - {filename_stem}
    return set()


def _collect_candidate_files() -> tuple[list[Path], list[Path], list[Path]]:
    results = sorted(p for p in RESULTS_DIR.glob("*.json"))
    scripts = sorted(p for p in SCRIPTS_DIR.glob("*.py"))
    figures = sorted(p for p in FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    return results, scripts, figures


def main() -> int:
    if not THESIS_PATH.exists():
        print(f"ERROR: thesis not found at {THESIS_PATH}")
        return 1

    AUDIT_DIR.mkdir(exist_ok=True)
    thesis_text = THESIS_PATH.read_text(encoding="utf-8")
    overrides = load_claim_mapping_overrides(CLAIM_MAPPING_PATH)

    results, scripts, figures = _collect_candidate_files()
    all_paths = results + scripts + figures
    filenames = [p.name for p in all_paths]

    # Read every script's text for cross-script reference detection.
    scripts_text = {p.name: p.read_text(encoding="utf-8", errors="replace") for p in scripts}

    # Pass 1: reference graph
    ref_graph = build_reference_graph(thesis_text, filenames, scripts_text)

    # Build per-file metadata.
    files_info: list[dict] = []
    by_stem: dict[str, dict] = {}

    # First pass: gather metadata + claim support, no classification yet.
    for p in all_paths:
        rel = p.relative_to(REPO_ROOT).as_posix()
        stem = p.stem
        # Apply override if present, else infer.
        claims = overrides.get(rel) if rel in overrides else infer_claim_support(rel)
        is_util = is_utility_script(rel) if p.suffix == ".py" else False
        info = {
            "path": rel,
            "stem": stem,
            "claim_support": claims,
            "thesis_refs": ref_graph[p.name]["thesis_lines"],
            "script_refs": ref_graph[p.name]["script_refs"],
            "is_utility": is_util,
        }
        files_info.append(info)
        by_stem[stem] = info

    # Second pass: classify, with sibling lookup.
    classified = []
    for info in files_info:
        sibling_stems = _supersession_siblings(info["stem"])
        siblings = [
            {"path": by_stem[s]["path"], "thesis_lines": by_stem[s]["thesis_refs"]}
            for s in sibling_stems
            if s in by_stem
        ]
        cls = classify_file(
            path=info["path"],
            thesis_lines=info["thesis_refs"],
            script_refs=info["script_refs"],
            claim_support=info["claim_support"],
            siblings=siblings,
            is_utility=info["is_utility"],
        )
        classified.append({
            "path": info["path"],
            "category": cls["category"],
            "claim_support": cls.get("claim_support", info["claim_support"]),
            "thesis_refs": info["thesis_refs"],
            "script_refs": info["script_refs"],
            "supersedes": None,
            "superseded_by": cls.get("superseded_by"),
            "confidence": cls.get("confidence"),
        })

    # Pass 2: stale-number cross-check, per cited result file.
    stale_flags: list[dict] = []
    # Two citation forms: \texttt{results/foo.json} and bare results/foo.json
    citation_re_texttt = re.compile(r"\\texttt\{results/([\w\\]+\.json)\}")
    citation_re_bare = re.compile(r"(?<!\w)results/([\w\\]+\.json)")
    thesis_lines_list = thesis_text.splitlines()
    for lineno, line in enumerate(thesis_lines_list, start=1):
        # Collect cited filenames from BOTH patterns, dedup per line.
        cited_fnames: set[str] = set()
        for match in citation_re_texttt.finditer(line):
            cited_fnames.add(match.group(1).replace("\\_", "_"))
        for match in citation_re_bare.finditer(line):
            cited_fnames.add(match.group(1).replace("\\_", "_"))
        for cited_fname in cited_fnames:
            cited_path = RESULTS_DIR / cited_fname
            if not cited_path.exists():
                continue
            try:
                cited_json = json.loads(cited_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            json_leaves = dict(flatten_json_numerics(cited_json))

            window_start = max(0, lineno - CITATION_WINDOW - 1)
            window_end = min(len(thesis_lines_list), lineno + CITATION_WINDOW)
            window_text = "\n".join(thesis_lines_list[window_start:window_end])
            flags = cross_check_numbers(window_text, json_leaves)
            for flag in flags:
                stale_flags.append({
                    "thesis_line": lineno,
                    "thesis_value": flag["thesis_value"],
                    "json_file": f"results/{cited_fname}",
                    "json_path": flag["closest_json_path"],
                    "json_value": flag["closest_json_value"],
                    "context": line.strip()[:120],
                    "recommendation": (
                        f"Verify: thesis claims {flag['thesis_value']} but JSON closest is "
                        f"{flag['closest_json_value']} at `{flag['closest_json_path']}` "
                        f"(distance {flag['distance']:.4f})"
                    ),
                })

    # Summary.
    summary = {
        "core": sum(1 for f in classified if f["category"] == "core"),
        "superseded": sum(1 for f in classified if f["category"] == "superseded"),
        "tangential": sum(1 for f in classified if f["category"] == "tangential"),
        "orphan": sum(1 for f in classified if f["category"] == "orphan"),
        "infrastructure": sum(1 for f in classified if f["category"] == "infrastructure"),
        "stale_numbers": len(stale_flags),
    }

    audit_data = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "stale_numbers": stale_flags,
        "files": classified,
    }

    write_inventory_md(audit_data, INVENTORY_MD)
    write_inventory_json(audit_data, INVENTORY_JSON)

    print(f"Audit complete. {summary}")
    print(f"  -> {INVENTORY_MD.relative_to(REPO_ROOT)}")
    print(f"  -> {INVENTORY_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
