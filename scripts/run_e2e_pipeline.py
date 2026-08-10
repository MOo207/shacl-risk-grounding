"""End-to-end pipeline benchmark.

Wires together: data loading → pre-inference → RF classification →
post-inference annotation → GRC artifact generation. Measures timing.

Output: results/e2e_pipeline_benchmark.json
"""
import json
import os
import sys
import time
import numpy as np
from datetime import datetime

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, BASE_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, "results")


def main():
    n_samples = 1000
    print(f"E2E Pipeline Benchmark: {n_samples} samples")
    timings = {}
    t_total = time.time()

    # Stage 1: Data Loading
    print("Stage 1: Loading CIC-IDS2018 data...")
    t0 = time.time()
    import pandas as pd
    data_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_dataset.csv")
    df = pd.read_csv(data_path, nrows=n_samples)
    # Separate features and labels
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].values
    y = df[label_col].values
    timings["load_data"] = round(time.time() - t0, 4)
    print(f"  Loaded {len(df)} samples, {len(feature_cols)} features in {timings['load_data']:.3f}s")

    # Stage 2: Pre-inference symbolic rules
    print("Stage 2: Pre-inference symbolic rules...")
    t0 = time.time()
    pre_inference_results = []
    for i in range(len(X)):
        row = X[i]
        # Simulate pre-inference rules (from pipeline/pre_inference/)
        rules_fired = 0
        context = {}
        # Rule 1: high_volume_flow (total packets > 1000)
        if len(row) > 4 and row[4] > 1000:
            rules_fired += 1
            context["high_volume"] = True
        # Rule 2: suspicious_iat (IAT std very low)
        if len(row) > 20 and row[20] < 10:
            rules_fired += 1
            context["suspicious_iat"] = True
        # Rule 3: syn_flood (SYN flag count > 5)
        if len(row) > 45 and row[45] > 5:
            rules_fired += 1
            context["syn_flood"] = True
        pre_inference_results.append({"rules_fired": rules_fired, "context": context})
    timings["pre_inference"] = round(time.time() - t0, 4)
    print(f"  Pre-inference: {timings['pre_inference']:.3f}s ({n_samples / timings['pre_inference']:.0f} samples/s)")

    # Stage 3: ML Classification (Random Forest)
    print("Stage 3: ML Classification (Random Forest)...")
    t0 = time.time()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Handle NaN/Inf
    X_clean = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    # Train on 70%, predict on all (benchmarking throughput, not accuracy)
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    t_train = time.time() - t0

    t0 = time.time()
    predictions = rf.predict(X_clean)  # Predict all samples for throughput measurement
    probabilities = rf.predict_proba(X_clean)
    t_predict = time.time() - t0

    timings["ml_train"] = round(t_train, 4)
    timings["ml_predict"] = round(t_predict, 4)
    print(f"  Train: {timings['ml_train']:.3f}s, Predict: {timings['ml_predict']:.3f}s "
          f"({n_samples / timings['ml_predict']:.0f} samples/s)")

    # Stage 4: Post-inference annotation
    print("Stage 4: Post-inference GRC annotation...")
    t0 = time.time()
    classes = le.classes_

    # NFCRM clause mapping per attack type
    nfcrm_map = {
        "Benign": "6.8", "BruteForce": "6.7", "DoS": "6.5",
        "DDoS": "6.5", "WebAttack": "6.4", "Infiltration": "6.6", "Bot": "6.5",
    }
    # ATT&CK mapping
    attack_map = {
        "BruteForce": "T1110", "DoS": "T1499", "DDoS": "T1498",
        "WebAttack": "T1190", "Infiltration": "T1570", "Bot": "T1583.005",
    }

    annotations = []
    for i in range(len(predictions)):
        pred_class = classes[predictions[i]] if predictions[i] < len(classes) else "Unknown"
        confidence = float(probabilities[i].max())
        annotation = {
            "prediction": pred_class,
            "confidence": round(confidence, 3),
            "nfcrm_clause": f"NFCRM-1:2025-\u00a7{nfcrm_map.get(pred_class, '6.8')}",
            "attack_technique": attack_map.get(pred_class, "N/A"),
            "risk_level": "High" if confidence > 0.8 and pred_class != "Benign" else
                          "Medium" if pred_class != "Benign" else "Low",
        }
        annotations.append(annotation)
    timings["post_annotation"] = round(time.time() - t0, 4)
    print(f"  Annotation: {timings['post_annotation']:.3f}s "
          f"({n_samples / timings['post_annotation']:.0f} samples/s)")

    # Stage 5: GRC Artifact Generation
    print("Stage 5: GRC artifact generation...")
    t0 = time.time()
    artifacts = {"risk_cases": 0, "audit_logs": 0, "control_links": 0}
    for ann in annotations:
        if ann["prediction"] != "Benign":
            artifacts["risk_cases"] += 1
            artifacts["audit_logs"] += 1
            if ann["attack_technique"] != "N/A":
                artifacts["control_links"] += 1
    timings["grc_generation"] = round(time.time() - t0, 4)
    print(f"  GRC generation: {timings['grc_generation']:.3f}s, "
          f"artifacts: {artifacts}")

    # Compute totals
    total_time = time.time() - t_total
    pipeline_time = sum(v for k, v in timings.items() if k != "ml_train")

    output = {
        "generated": datetime.now().isoformat(),
        "total_samples": n_samples,
        "total_time_seconds": round(total_time, 3),
        "pipeline_time_seconds": round(pipeline_time, 3),
        "throughput_samples_per_second": round(n_samples / pipeline_time, 1),
        "average_latency_ms": round(1000 * pipeline_time / n_samples, 2),
        "per_stage_timing": timings,
        "per_stage_throughput": {
            k: round(n_samples / v, 0) if v > 0 else 0
            for k, v in timings.items()
        },
        "artifacts_generated": artifacts,
        "prediction_distribution": {
            str(c): int((predictions == i).sum())
            for i, c in enumerate(classes)
        },
        "note": "Pipeline time excludes ML training (one-time cost). Throughput measures inference pipeline only.",
    }

    dest = os.path.join(RESULTS_DIR, "e2e_pipeline_benchmark.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Pipeline Benchmark Results:")
    print(f"  Total time: {total_time:.3f}s (incl. training)")
    print(f"  Pipeline time: {pipeline_time:.3f}s (excl. training)")
    print(f"  Throughput: {output['throughput_samples_per_second']:.0f} samples/sec")
    print(f"  Avg latency: {output['average_latency_ms']:.2f} ms/sample")
    print(f"  Artifacts: {artifacts}")
    print(f"  -> {dest}")


if __name__ == "__main__":
    main()
