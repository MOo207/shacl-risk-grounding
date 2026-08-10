"""
Butterfly Effect Propagation — SafecareOnto Risk Cascade Algorithm
===================================================================

Implements the SafecareOnto butterfly effect formula on the CIC-IDS2018 Testbed ARG
(Asset Relationship Graph) loaded from data/arg_seed.json.

Formula (Hannou et al. 2022):
  impactScore(a) = 1 - γ(Σ protectionDegree_j(a))  ∈ [0, 1]

where γ is an aggregation function.  Two aggregation modes are supported:

  --combination-mode max       (default)
      impactScore(a) = 1 - max(protectionDegree of controls on a)

  --combination-mode additive
      impactScore(a) = max(0, 1 - min(1.0, Σ protectionDegree_j(a)))
      This models compounding controls (e.g., firewall + IDS together
      providing full protection against DDOS — the BARRIER scenario).

Propagation rule:
  If impactScore(a) == 0.0 on any intermediate asset → STOP on that path.
  Otherwise continue BFS upward through traversal edges.

Traversal edge types (upward direction):
  hostsSoftware  : SoftwareAsset → HardwareAsset
  hostsDevice    : HardwareAsset → Building
  composedOf     : SimpleBuilding → ComplexBuilding
  leadsTo        : NetworkSegment → (target asset, forward direction)
  supports       : SupportingAsset → BusinessAsset
  depends        : BusinessAsset ← SupportingAsset  (reverse traversal)

Usage
-----
  # Single scenario:
  python pipeline/loaders/butterfly_effect.py --threat FTP-BruteForce --source ftp-srv-01

  # DDOS barrier demo (additive mode):
  python pipeline/loaders/butterfly_effect.py \\
      --threat DDOS --source DMZ-Segment --combination-mode additive

  # Run all six CIC-IDS2018 scenarios:
  python pipeline/loaders/butterfly_effect.py --all

  # Import as module:
  from pipeline.loaders.butterfly_effect import ButterflyEffect, ARGGraph
  g = ARGGraph.from_json("data/arg_seed.json")
  be = ButterflyEffect(g, combination_mode="max")
  result = be.propagate("ftp-srv-01", "FTP-BruteForce")

Outputs
-------
  data/butterfly_effect_results.json — one result object per scenario

References
----------
  SafecareOnto: Hannou et al. (2022)
  NFCRM-1:2025: National Framework for Cybersecurity Risk Management
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"
_ARG_SEED_PATH = _DATA_DIR / "arg_seed.json"
_RESULTS_PATH = _REPO_ROOT / "results" / "butterfly_effect_results.json"

# ---------------------------------------------------------------------------
# Edge types that represent upward propagation in the ARG
# ---------------------------------------------------------------------------

# Edges traversed in the FORWARD (source→target) direction — these point
# upward in the SafecareOnto hierarchy.
UPWARD_FORWARD_EDGES = {
    "supports",   # SupportingAsset → BusinessAsset
    "leadsTo",    # NetworkSegment → remote asset (lateral movement)
}

# Edges traversed in the REVERSE (target→source) direction — the ontology
# defines them as parent→child but we traverse child→parent.
UPWARD_REVERSE_EDGES = {
    "hostsSoftware",  # HardwareAsset → SoftwareAsset  (reverse: SW → HW)
    "hostsDevice",    # Building → HardwareAsset        (reverse: HW → Building)
    "composedOf",     # ComplexBuilding → SimpleBuilding (reverse: SB → CB)
    "depends",        # BusinessAsset → SupportingAsset (reverse: SA → BA)
}

# All upward edge types (for adjacency construction)
ALL_TRAVERSAL_EDGES = UPWARD_FORWARD_EDGES | UPWARD_REVERSE_EDGES

# ---------------------------------------------------------------------------
# Predefined scenarios mapping CIC-IDS2018 attack families to ARG source nodes
# ---------------------------------------------------------------------------

SCENARIOS: list[dict[str, str]] = [
    {
        "threat":      "FTP-BruteForce",
        "source":      "ftp-srv-01",
        "threat_node": "Threat-FTPBrute-T1110",
        "note":        "vsftpd 3.0.3 / CVE-2011-2523 -- account lockout pd=0.65",
    },
    {
        "threat":      "DoS",
        "source":      "web-srv-01",
        "threat_node": "Threat-DoS-T1499",
        "note":        "IIS 10.0 / CVE-2015-1635 -- WAF rule pd=0.70",
    },
    {
        "threat":      "Infiltration",
        "source":      "web-srv-01",
        "threat_node": "Threat-Infiltration-T1190",
        "note":        "Microsoft 365 / CVE-2021-26855 ProxyLogon -- patch mgmt pd=0.80",
    },
    {
        "threat":      "WebAttack-SQLi",
        "source":      "db-srv-01",
        "threat_node": "Threat-SQLi-T1190",
        "note":        "SQL Server 2019 / CVE-2019-0819 -- input validation pd=0.75",
    },
    {
        "threat":      "Botnet",
        "source":      "ws-01",
        "threat_node": "Threat-Botnet-T1071",
        "note":        "Chrome Browser / CVE-2023-2033 -- endpoint detection pd=0.60",
    },
    {
        "threat":      "DDOS",
        "source":      "DMZ-Segment",
        "threat_node": "Threat-DDOS-T1498",
        "combination_mode": "additive",
        "note":        "DDOS flood -- fw-01 pd=0.85 + ids-01 pd=0.75 -> additive barrier",
    },
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PropagationStep:
    """One node in a butterfly effect propagation chain."""
    level: int
    asset_id: str
    asset_labels: list[str]
    asset_properties: dict[str, Any]
    controls: list[dict[str, Any]]          # [{id, protectionDegree, controlName}]
    max_protection_degree: float
    additive_protection_degree: float
    impact_score_max: float                 # using max aggregation
    impact_score_additive: float            # using additive aggregation
    propagation_barrier_max: bool
    propagation_barrier_additive: bool
    note: str = ""


@dataclass
class PropagationResult:
    """Full propagation result for one threat/source scenario."""
    scenario: str                           # threat label
    source_asset: str
    threat_node_id: str
    combination_mode: str                   # "max" | "additive"
    propagation_chain: list[PropagationStep] = field(default_factory=list)
    barriers_max: list[str] = field(default_factory=list)
    barriers_additive: list[str] = field(default_factory=list)
    max_affected_level: int = 0
    summary: str = ""


# ---------------------------------------------------------------------------
# ARGGraph — loads and indexes the ARG JSON
# ---------------------------------------------------------------------------

class ARGGraph:
    """
    In-memory representation of the ARG loaded from arg_seed.json.

    Provides:
      - Node lookup by id
      - Forward and reverse adjacency lists, filtered by edge type
      - Control lookup per asset (for impactScore computation)
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._raw = data
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []

        # Adjacency: forward[src][rel] = [tgt, ...]
        self._forward: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        # Reverse adjacency: reverse[tgt][rel] = [src, ...]
        self._reverse: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

        self._build_index()

    def _build_index(self) -> None:
        for n in self._raw.get("nodes", []):
            self.nodes[n["id"]] = n
        for e in self._raw.get("edges", []):
            self.edges.append(e)
            src, tgt, rel = e["source"], e["target"], e["type"]
            self._forward[src][rel].append(tgt)
            self._reverse[tgt][rel].append(src)

    @classmethod
    def from_json(cls, path: Path | str) -> "ARGGraph":
        """Load an ARGGraph from a JSON file produced by arg_seed.py."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data)

    def get_node(self, nid: str) -> dict[str, Any] | None:
        return self.nodes.get(nid)

    def get_labels(self, nid: str) -> list[str]:
        n = self.nodes.get(nid)
        return n["labels"] if n else []

    def get_prop(self, nid: str, prop: str, default: Any = None) -> Any:
        n = self.nodes.get(nid)
        if n is None:
            return default
        return n.get("properties", {}).get(prop, default)

    def upward_neighbours(self, nid: str) -> list[tuple[str, str]]:
        """
        Return all (neighbour_id, edge_type) pairs reachable upward from nid.

        Forward traversal for UPWARD_FORWARD_EDGES (nid is source).
        Reverse traversal for UPWARD_REVERSE_EDGES (nid is target, parent is source).
        """
        result: list[tuple[str, str]] = []

        for rel in UPWARD_FORWARD_EDGES:
            for tgt in self._forward[nid].get(rel, []):
                result.append((tgt, rel))

        for rel in UPWARD_REVERSE_EDGES:
            for parent in self._reverse[nid].get(rel, []):
                result.append((parent, rel))

        return result

    def get_controls_for_asset(self, nid: str) -> list[dict[str, Any]]:
        """
        Return a list of control property dicts for all Controls protecting nid.

        Looks for hasControl edges where nid is the source (asset has the control).
        """
        controls = []
        for ctrl_id in self._forward[nid].get("hasControl", []):
            ctrl_node = self.nodes.get(ctrl_id)
            if ctrl_node:
                props = ctrl_node.get("properties", {})
                controls.append({
                    "id": ctrl_id,
                    "controlName": props.get("controlName", ""),
                    "protectionDegree": float(props.get("protectionDegree", 0.0)),
                    "nfcrmClause": props.get("nfcrmClause", ""),
                    "controlId": props.get("controlId", ctrl_id),
                })
        return controls


# ---------------------------------------------------------------------------
# ButterflyEffect — core propagation algorithm
# ---------------------------------------------------------------------------

class ButterflyEffect:
    """
    Implements SafecareOnto butterfly effect propagation on an ARGGraph.

    Parameters
    ----------
    graph : ARGGraph
        The loaded asset relationship graph.
    combination_mode : str
        "max"      — impactScore = 1 - max(protectionDegrees)
        "additive" — impactScore = max(0, 1 - min(1, sum(protectionDegrees)))
    """

    def __init__(self, graph: ARGGraph, combination_mode: str = "max") -> None:
        self.graph = graph
        if combination_mode not in ("max", "additive"):
            raise ValueError(f"combination_mode must be 'max' or 'additive', got {combination_mode!r}")
        self.combination_mode = combination_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propagate(
        self,
        source_id: str,
        threat_label: str,
        threat_node_id: str = "",
    ) -> PropagationResult:
        """
        Run butterfly effect BFS from source_id.

        Parameters
        ----------
        source_id : str
            ID of the asset where the incident occurs.
        threat_label : str
            Human-readable threat name (e.g., "FTP-BruteForce").
        threat_node_id : str
            ID of the Threat node in the ARG (optional, for JSON output).

        Returns
        -------
        PropagationResult
            Full propagation chain with impactScores and barrier flags.
        """
        result = PropagationResult(
            scenario=threat_label,
            source_asset=source_id,
            threat_node_id=threat_node_id,
            combination_mode=self.combination_mode,
        )

        if source_id not in self.graph.nodes:
            result.summary = f"ERROR: source asset '{source_id}' not found in ARG."
            return result

        # BFS state: queue holds (asset_id, level)
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()
        queue.append((source_id, 0))

        while queue:
            current_id, level = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            step = self._compute_step(current_id, level)
            result.propagation_chain.append(step)
            result.max_affected_level = max(result.max_affected_level, level)

            # Determine whether this step blocks propagation
            if self.combination_mode == "max":
                barrier = step.propagation_barrier_max
            else:
                barrier = step.propagation_barrier_additive

            if barrier:
                # Record barrier and do NOT enqueue upward neighbours
                if self.combination_mode == "max":
                    result.barriers_max.append(current_id)
                else:
                    result.barriers_additive.append(current_id)
                continue

            # Enqueue upward neighbours
            for (neighbour_id, edge_type) in self.graph.upward_neighbours(current_id):
                if neighbour_id not in visited:
                    queue.append((neighbour_id, level + 1))

        result.summary = self._build_summary(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_step(self, asset_id: str, level: int) -> PropagationStep:
        """Compute impact scores and protection info for one asset."""
        controls = self.graph.get_controls_for_asset(asset_id)
        pds = [c["protectionDegree"] for c in controls]

        max_pd = max(pds) if pds else 0.0
        additive_pd = min(1.0, sum(pds)) if pds else 0.0

        impact_max = round(1.0 - max_pd, 4)
        impact_additive = round(max(0.0, 1.0 - additive_pd), 4)

        node = self.graph.get_node(asset_id) or {}
        props = node.get("properties", {})

        # Build a note summarising the protection situation
        if not controls:
            note = "No controls registered -- fully unprotected"
        elif impact_max == 0.0:
            note = f"Full barrier (max mode): max pd={max_pd:.2f}"
        elif impact_additive == 0.0:
            # Build individual pd strings separately to avoid nested f-string parsing issues
            pd_strings = [f"{c['id']}={c['protectionDegree']:.2f}" for c in controls]
            note = (
                f"Full barrier (additive mode): sum pd={additive_pd:.2f} "
                f"(individual pds: {', '.join(pd_strings)})"
            )
        else:
            ctrl_summary = ", ".join(
                f"{c['id']} pd={c['protectionDegree']:.2f}" for c in controls
            )
            note = f"Controls: {ctrl_summary} -> max_pd={max_pd:.2f}"

        return PropagationStep(
            level=level,
            asset_id=asset_id,
            asset_labels=node.get("labels", []),
            asset_properties={
                k: v for k, v in props.items()
                if k in ("hostname", "buildingName", "assetName", "softwareName",
                         "softwareVersion", "ipAddress", "osName", "role",
                         "impactScore", "propagationBarrier")
            },
            controls=controls,
            max_protection_degree=round(max_pd, 4),
            additive_protection_degree=round(additive_pd, 4),
            impact_score_max=impact_max,
            impact_score_additive=impact_additive,
            propagation_barrier_max=(impact_max == 0.0),
            propagation_barrier_additive=(impact_additive == 0.0),
            note=note,
        )

    def _build_summary(self, result: PropagationResult) -> str:
        """Build a human-readable one-line propagation summary."""
        chain_ids = [s.asset_id for s in result.propagation_chain]
        chain_str = " -> ".join(chain_ids)

        mode_barriers = (
            result.barriers_max
            if self.combination_mode == "max"
            else result.barriers_additive
        )

        if mode_barriers:
            barrier_str = f" | BARRIER at: {', '.join(mode_barriers)}"
        else:
            barrier_str = ""

        return (
            f"{result.scenario} on {result.source_asset} -> "
            f"{chain_str}"
            f"{barrier_str}"
        )


# ---------------------------------------------------------------------------
# Pretty-print a single propagation result
# ---------------------------------------------------------------------------

def print_result(result: PropagationResult, combination_mode: str) -> None:
    """Print a formatted propagation report to stdout."""
    score_key = "impact_score_max" if combination_mode == "max" else "impact_score_additive"
    barrier_key = "propagation_barrier_max" if combination_mode == "max" else "propagation_barrier_additive"

    print(f"\n{'-' * 70}")
    print(f"  Incident   : {result.scenario}")
    print(f"  Source     : {result.source_asset}")
    print(f"  Threat node: {result.threat_node_id}")
    print(f"  Mode       : {combination_mode}")
    print(f"{'-' * 70}")

    for step in result.propagation_chain:
        score = getattr(step, score_key)
        is_barrier = getattr(step, barrier_key)
        labels_str = "/".join(step.asset_labels[:2]) if step.asset_labels else "?"

        name = (
            step.asset_properties.get("hostname")
            or step.asset_properties.get("buildingName")
            or step.asset_properties.get("softwareName")
            or step.asset_properties.get("assetName")
            or step.asset_id
        )

        barrier_flag = " [BARRIER]" if is_barrier else ""
        indent = "  " * step.level
        print(
            f"  Level {step.level}: {indent}{step.asset_id} ({labels_str})"
            f" - impactScore={score:.4f}{barrier_flag}"
        )
        print(f"           {indent}  name='{name}', note: {step.note}")

    barriers = (
        result.barriers_max
        if combination_mode == "max"
        else result.barriers_additive
    )
    if barriers:
        print(f"\n  BARRIER(S): {', '.join(barriers)}")
        print("  Propagation stopped at barrier -- upstream assets are protected.")
    else:
        print("\n  No barriers -- full cascade propagated to top of hierarchy.")

    print(f"\n  Summary: {result.summary}")
    print(f"{'-' * 70}")


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def _result_to_dict(result: PropagationResult) -> dict[str, Any]:
    """Convert a PropagationResult to a JSON-serialisable dict."""
    d = asdict(result)
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="butterfly_effect",
        description=(
            "SafecareOnto butterfly effect propagation on the INTERSYMBOLIC-GRC ARG.\n"
            "Loads data/arg_seed.json and computes impactScore cascades."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FTP brute-force scenario:
  python pipeline/loaders/butterfly_effect.py --threat FTP-BruteForce --source ftp-srv-01

  # DDOS with additive protection (barrier demo):
  python pipeline/loaders/butterfly_effect.py \\
      --threat DDOS --source DMZ-Segment --combination-mode additive

  # Run all six CIC-IDS2018 scenarios:
  python pipeline/loaders/butterfly_effect.py --all
        """,
    )
    p.add_argument(
        "--threat", "-t",
        metavar="THREAT_LABEL",
        help="Threat label (e.g., FTP-BruteForce, DoS, DDOS, Botnet, ...)",
    )
    p.add_argument(
        "--source", "-s",
        metavar="ASSET_ID",
        help="Source asset ID where the incident occurs (e.g., ftp-srv-01)",
    )
    p.add_argument(
        "--combination-mode", "-m",
        metavar="MODE",
        choices=["max", "additive"],
        default="max",
        help=(
            "Protection aggregation mode: "
            "'max' (default) = 1-max(pds); "
            "'additive' = max(0, 1-sum(pds))"
        ),
    )
    p.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all six predefined CIC-IDS2018 scenarios and write results JSON.",
    )
    p.add_argument(
        "--arg-json",
        metavar="PATH",
        default=str(_ARG_SEED_PATH),
        help=f"Path to arg_seed.json (default: {_ARG_SEED_PATH})",
    )
    p.add_argument(
        "--output", "-o",
        metavar="PATH",
        default=str(_RESULTS_PATH),
        help=f"Output JSON path (default: {_RESULTS_PATH})",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-step console output (still writes JSON).",
    )
    return p


def _ensure_arg_seed(arg_json_path: Path) -> None:
    """Auto-generate arg_seed.json if it does not exist."""
    if not arg_json_path.exists():
        print(f"[butterfly_effect] {arg_json_path} not found -- running arg_seed.py first...")
        from pipeline.loaders.arg_seed import main as seed_main  # noqa: PLC0415
        seed_main()


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    arg_json_path = Path(args.arg_json)
    _ensure_arg_seed(arg_json_path)

    print(f"[butterfly_effect] Loading ARG from {arg_json_path}...")
    graph = ARGGraph.from_json(arg_json_path)
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")

    results: list[PropagationResult] = []

    if args.all:
        # Run all predefined scenarios
        for s in SCENARIOS:
            mode = s.get("combination_mode", "max")
            be = ButterflyEffect(graph, combination_mode=mode)
            result = be.propagate(
                source_id=s["source"],
                threat_label=s["threat"],
                threat_node_id=s.get("threat_node", ""),
            )
            results.append(result)
            if not args.quiet:
                print_result(result, mode)

    elif args.threat and args.source:
        mode = args.combination_mode
        be = ButterflyEffect(graph, combination_mode=mode)
        result = be.propagate(
            source_id=args.source,
            threat_label=args.threat,
            threat_node_id="",
        )
        results.append(result)
        if not args.quiet:
            print_result(result, mode)

    else:
        parser.print_help()
        print(
            "\n[butterfly_effect] Tip: use --all to run all six CIC-IDS2018 scenarios,"
            " or supply --threat and --source for a specific scenario."
        )
        return

    # Write JSON output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "description": "Butterfly effect propagation results for INTERSYMBOLIC-GRC ARG",
            "formula": "impactScore(a) = 1 - aggregation(protectionDegrees of controls on a)",
            "aggregation_modes": {
                "max":      "impactScore = 1 - max(protectionDegrees)",
                "additive": "impactScore = max(0, 1 - min(1.0, sum(protectionDegrees)))",
            },
            "reference": "Hannou et al. (2022) SafecareOnto",
            "scenario_count": len(results),
        },
        "results": [_result_to_dict(r) for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output_data, fh, indent=2, ensure_ascii=False)

    print(f"\n[butterfly_effect] Results written to {output_path}")
    print(f"  Scenarios processed : {len(results)}")

    # Print barrier summary
    all_barriers = []
    for r in results:
        all_barriers.extend(
            r.barriers_max if r.combination_mode == "max" else r.barriers_additive
        )
    if all_barriers:
        print(f"  Propagation barriers: {', '.join(set(all_barriers))}")
    else:
        print("  Propagation barriers: none (all scenarios fully cascaded)")


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def run_scenario(
    threat: str,
    source: str,
    combination_mode: str = "max",
    arg_json: Path | str = _ARG_SEED_PATH,
) -> PropagationResult:
    """
    Convenience function for notebook / pipeline integration.

    Parameters
    ----------
    threat : str
        Threat label (e.g., "FTP-BruteForce").
    source : str
        Source asset ID (e.g., "ftp-srv-01").
    combination_mode : str
        "max" or "additive".
    arg_json : Path or str
        Path to arg_seed.json.

    Returns
    -------
    PropagationResult
        Full propagation result.

    Example
    -------
    >>> from pipeline.loaders.butterfly_effect import run_scenario
    >>> result = run_scenario("FTP-BruteForce", "ftp-srv-01")
    >>> print(result.summary)
    """
    graph = ARGGraph.from_json(arg_json)
    be = ButterflyEffect(graph, combination_mode=combination_mode)
    return be.propagate(source_id=source, threat_label=threat)


if __name__ == "__main__":
    main()
