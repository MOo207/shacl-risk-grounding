"""
4-Layer Intersymbolic Explanation Runner (T-A7)
===============================================
Generates a complete 4-layer explanation for 5 sample flows
(one per attack class where possible):

  Layer 1: Pre-inference symbolic context
           c(x) = {rule(x) | rule ∈ pre_rules}
  Layer 2: SHAP top-3 features
           φᵢ(f, x) from results/shap_values.npy + shap_top20_features.json
  Layer 3: Post-inference GRC trace
           τ(x) = (CVE → RiskCase → NFCRM-1:2025 clause)
  Layer 4: NL synthesis (Claude Haiku 4.5 via Claude API)

Saves: results/intersymbolic_explanations.json

Usage:
    python scripts/run_intersymbolic_explanation.py
    python scripts/run_intersymbolic_explanation.py
    python scripts/run_intersymbolic_explanation.py --data data/processed/cleaned_dataset.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# GRC Knowledge Base: attack_type → (CVE examples, risk_level, NFCRM clause)
# ---------------------------------------------------------------------------
GRC_KB = {
    "BruteForce": {
        "cve_examples": ["CVE-2023-23397", "CVE-2022-26134"],
        "nfcrm_clause": "NFCRM-1:2025 §6.7 (Access Control Enforcement)",
        "iso_clause": "ISO/IEC 27005:2022 §8.2 (Risk Identification)",
        "risk_level": "HIGH",
        "control": "Account lockout + MFA enforcement",
        "nca_ref": "NCA-ECC-2-3-1",
    },
    "DoS": {
        "cve_examples": ["CVE-2023-44487 (HTTP/2 Rapid Reset)", "CVE-2022-3602"],
        "nfcrm_clause": "NFCRM-1:2025 §6.9 (Inherent Risk Analysis)",
        "iso_clause": "ISO/IEC 27005:2022 §8.3 (Risk Analysis)",
        "risk_level": "HIGH",
        "control": "Rate limiting + traffic scrubbing",
        "nca_ref": "NCA-ECC-3-1-2",
    },
    "DDoS": {
        "cve_examples": ["CVE-2023-44487", "CVE-2022-26143"],
        "nfcrm_clause": "NFCRM-1:2025 §6.9 (Inherent Risk Analysis)",
        "iso_clause": "ISO/IEC 27005:2022 §8.3 (Risk Analysis)",
        "risk_level": "CRITICAL",
        "control": "DDoS mitigation service + CDN scrubbing",
        "nca_ref": "NCA-ECC-3-1-2",
    },
    "WebAttack": {
        "cve_examples": ["CVE-2023-34362 (SQL Injection)", "CVE-2021-44228 (Log4Shell XSS)"],
        "nfcrm_clause": "NFCRM-1:2025 §6.4 (Vulnerability Inventory)",
        "iso_clause": "ISO/IEC 27005:2022 §8.3 (Risk Analysis)",
        "risk_level": "HIGH",
        "control": "WAF + input validation + parameterised queries",
        "nca_ref": "NCA-ECC-4-2-1",
    },
    "Infiltration": {
        "cve_examples": ["CVE-2023-20198", "CVE-2022-30190 (Follina)"],
        "nfcrm_clause": "NFCRM-1:2025 §6.5 (Threat Detection)",
        "iso_clause": None,  # §6.5 not yet aligned to ISO 27005; future work
        "risk_level": "CRITICAL",
        "control": "Network segmentation + EDR + privileged access management",
        "nca_ref": "NCA-ECC-5-1-3",
    },
    "Bot": {
        "cve_examples": ["CVE-2023-35078", "CVE-2022-1388"],
        "nfcrm_clause": "NFCRM-1:2025 §6.5 (Threat Monitoring)",
        "iso_clause": None,  # §6.5 not yet aligned to ISO 27005; future work
        "risk_level": "MEDIUM",
        "control": "DNS sinkholing + C2 blocklists + endpoint isolation",
        "nca_ref": "NCA-ECC-6-1-1",
    },
    "Benign": {
        "cve_examples": [],
        "nfcrm_clause": "N/A",
        "iso_clause": None,
        "risk_level": "NONE",
        "control": "No intervention required",
        "nca_ref": "N/A",
    },
}

# ---------------------------------------------------------------------------
# Layer 1: Pre-inference symbolic context rules
# ---------------------------------------------------------------------------
# Column names follow CIC-IDS2018 short form (the processed dataset schema);
# CIC-2017 long-form fallbacks kept via _col().
def _col(r, *names, default=0):
    for n in names:
        if n in r.index:
            return r[n]
    return default


PRE_RULES = [
    ("high_volume_flow",    lambda r: _col(r, "Tot Fwd Pkts", "Total Fwd Packets") > 1000),
    ("short_duration",      lambda r: _col(r, "Flow Duration") < 10),
    ("suspicious_iat",      lambda r: _col(r, "Flow IAT Std", default=999) < 5),
    ("syn_flood_indicator", lambda r: _col(r, "SYN Flag Cnt", "SYN Flag Count") > 50),
    ("large_data_transfer", lambda r: _col(r, "TotLen Fwd Pkts", "Total Length of Fwd Packets") > 100000),
    ("high_pps",            lambda r: _col(r, "Flow Pkts/s", "Flow Packets/s") > 100),
]


def layer1_context(row: pd.Series) -> dict:
    """Layer 1: which symbolic context rules fire for this flow."""
    fired = {}
    for name, fn in PRE_RULES:
        try:
            fired[name] = bool(fn(row))
        except Exception:
            fired[name] = False
    active = [k for k, v in fired.items() if v]
    return {
        "all_rules": fired,
        "active_rules": active,
        "symbolic_hint": ("possible_attack" if active else "benign_pattern"),
    }


# ---------------------------------------------------------------------------
# Layer 2: SHAP attribution
# ---------------------------------------------------------------------------
def layer2_shap_exact(row: pd.Series, clf, feature_cols: list,
                      explainer) -> dict:
    """Layer 2: per-row SHAP top-3 computed on this exact flow."""
    x = pd.DataFrame([{c: row.get(c, 0) for c in feature_cols}])
    sv = explainer.shap_values(x)
    # list[n_classes] (old shap) | 3-D (1, features, classes) | 2-D (1, features)
    if isinstance(sv, list):
        abs_per_feature = np.mean([np.abs(s[0]) for s in sv], axis=0)
    elif getattr(sv, "ndim", 2) == 3:
        abs_per_feature = np.abs(sv[0]).mean(axis=1)
    else:
        abs_per_feature = np.abs(sv[0])
    top_idx = np.argsort(abs_per_feature)[::-1][:3]
    top3 = [{"feature": feature_cols[int(j)],
             "mean_abs_shap": float(abs_per_feature[int(j)]),
             "rank": r + 1}
            for r, j in enumerate(top_idx)]
    return {"top3_features": top3}


# ---------------------------------------------------------------------------
# Layer 3: GRC trace
# ---------------------------------------------------------------------------
def layer3_grc_trace(predicted_label: str) -> dict:
    """Layer 3: map prediction to CVE → RiskCase → NFCRM clause → ISO 27005 clause."""
    kb = GRC_KB.get(predicted_label, GRC_KB["Benign"])
    return {
        "predicted_attack_type": predicted_label,
        "cve_examples": kb["cve_examples"],
        "risk_level": kb["risk_level"],
        "nfcrm_clause": kb["nfcrm_clause"],
        "iso_clause": kb.get("iso_clause"),
        "recommended_control": kb["control"],
        "nca_reference": kb["nca_ref"],
        "risk_case": f"RC-{predicted_label.upper()}-{kb['risk_level']}",
    }


# ---------------------------------------------------------------------------
# Layer 4: NL synthesis (Claude API)
# ---------------------------------------------------------------------------
def layer4_nl_synthesis(l1: dict, l2: dict, l3: dict,
                        use_ollama: bool) -> dict:
    """Layer 4: natural language explanation via Claude API (Haiku 4.5)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from claude_cli_client import ClaudeCLIClient
    except ImportError:
        return {"status": "FAILED", "reason": "claude_cli_client not found", "nl_explanation": None}

    system_msg = (
        "You are a cybersecurity GRC expert. Write exactly 3 sentences for a compliance officer. "
        "Be concise and non-technical. Do not use bullet points."
    )
    user_msg = (
        f"Symbolic context (Layer 1 pre-inference rules fired): {l1['active_rules']}\n"
        f"Top SHAP features (Layer 2): {[f['feature'] for f in l2.get('top3_features', [])]}\n"
        f"GRC trace (Layer 3): {l3['predicted_attack_type']} detected → "
        f"{l3['nfcrm_clause']} → risk level {l3['risk_level']} → "
        f"recommended control: {l3['recommended_control']}\n\n"
        "Write a 3-sentence audit-ready explanation for a compliance officer."
    )
    try:
        client = ClaudeCLIClient(model="haiku")
        nl_text = client.generate(system_msg, user_msg)
        if nl_text:
            return {"status": "OK", "backend": "claude_api_haiku-4.5", "nl_explanation": nl_text.strip()}
        return {"status": "FAILED", "reason": "empty response from Claude API", "nl_explanation": None}
    except Exception as e:
        return {"status": "FAILED", "reason": str(e), "nl_explanation": None}


def run_intersymbolic_explanation(data_path: Path, output_dir: Path,
                                  use_ollama: bool) -> list:
    # ---- Load data ---------------------------------------------------------
    print(f"[T-A7] Loading {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    # ---- SHAP explainer (per-row, exact) -----------------------------------
    try:
        import shap
        shap_available = True
    except ImportError:
        shap_available = False
        print("[T-A7] WARNING: shap not installed — Layer 2 will be empty")
        print("  pip install shap (T-A6)")

    # ---- Load RF model for predictions -------------------------------------
    model_path = output_dir / "rf_model.joblib"
    if model_path.exists():
        import joblib
        artifact = joblib.load(model_path)
        clf = artifact["model"]
        feature_cols = artifact["feature_cols"]
        model_available = True
        print(f"[T-A7] Loaded RF model ({len(feature_cols)} features)")
    else:
        clf = None
        feature_cols = []
        model_available = False
        print("[T-A7] WARNING: RF model not found — using ground-truth labels as predictions")

    explainer = None
    if shap_available and model_available:
        explainer = shap.TreeExplainer(clf)
        print("[T-A7] SHAP TreeExplainer ready (per-row exact attributions)")

    # ---- Sample 5 flows (1 per attack class) --------------------------------
    all_labels = df["Label"].unique()
    priority_labels = ["DDoS", "BruteForce", "DoS", "WebAttack", "Infiltration", "Bot"]
    selected_labels = [l for l in priority_labels if l in all_labels][:5]
    # Fill with remaining labels if needed
    for l in all_labels:
        if len(selected_labels) >= 5:
            break
        if l not in selected_labels:
            selected_labels.append(l)

    samples = []
    for label in selected_labels[:5]:
        rows = df[df["Label"] == label]
        if len(rows) > 0:
            samples.append(rows.sample(1, random_state=42).iloc[0])

    print(f"[T-A7] Selected {len(samples)} sample flows: {[s['Label'] for s in samples]}")

    # ---- Generate 4-layer explanation for each sample -----------------------
    explanations = []
    for i, row in enumerate(samples):
        true_label = row["Label"]

        # Get model prediction
        if model_available and feature_cols:
            try:
                missing = [c for c in feature_cols if c not in row.index]
                x = pd.DataFrame([{c: row.get(c, 0) for c in feature_cols}])
                predicted_label = clf.predict(x)[0]
            except Exception:
                predicted_label = true_label
        else:
            predicted_label = true_label

        # Layer 1
        l1 = layer1_context(row)

        # Layer 2 — exact per-row SHAP for this flow
        if explainer is not None:
            l2 = layer2_shap_exact(row, clf, feature_cols, explainer)
        else:
            l2 = {"top3_features": [], "note": "SHAP not computed — run T-A6 first"}

        # Layer 3
        l3 = layer3_grc_trace(predicted_label)

        # Layer 4
        l4 = layer4_nl_synthesis(l1, l2, l3, use_ollama)

        explanation = {
            "flow_id": i + 1,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct_prediction": (true_label == predicted_label),
            "layer1": l1,
            "layer2": l2,
            "layer3": l3,
            "layer4_optional": l4,
            "key_flow_stats": {
                "Flow Duration":       float(_col(row, "Flow Duration")),
                "Dst Port":            float(_col(row, "Dst Port", "Destination Port")),
                "Tot Fwd Pkts":        float(_col(row, "Tot Fwd Pkts", "Total Fwd Packets")),
                "Tot Bwd Pkts":        float(_col(row, "Tot Bwd Pkts", "Total Backward Packets")),
                "Flow Pkts/s":         float(_col(row, "Flow Pkts/s", "Flow Packets/s")),
                "SYN Flag Cnt":        float(_col(row, "SYN Flag Cnt", "SYN Flag Count")),
                "TotLen Fwd Pkts":     float(_col(row, "TotLen Fwd Pkts", "Total Length of Fwd Packets")),
                "Init Fwd Win Byts":   float(_col(row, "Init Fwd Win Byts")),
                "Fwd Seg Size Min":    float(_col(row, "Fwd Seg Size Min")),
            },
        }
        explanations.append(explanation)
        print(f"  [T-A7] Flow {i+1}: true={true_label}  pred={predicted_label}  "
              f"L1_active={l1['active_rules'][:2]}")

    # ---- Save results -------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "intersymbolic_explanations.json"
    with open(out_path, "w") as f:
        json.dump(explanations, f, indent=2)
    print(f"\n[T-A7] Saved {len(explanations)} explanations to {out_path}")
    print(f"[T-A7] Keys per entry: {list(explanations[0].keys())}")

    return explanations


def main():
    parser = argparse.ArgumentParser(description="4-layer intersymbolic explanation (T-A7)")
    parser.add_argument("--data", default="data/processed/cleaned_dataset.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-ollama", action="store_true",
                        help="(deprecated, no-op) Layer 4 uses the Claude API")
    args = parser.parse_args()

    run_intersymbolic_explanation(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        use_ollama=True,   # always True — Claude API backend, no local GPU needed
    )


if __name__ == "__main__":
    main()
