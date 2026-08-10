"""SLM ablation: stripped-down NL features, no signature hints.

Replicates run_slm_nl_classification.py (CIC-IDS2018) with TWO key differences:
  1. NL descriptions contain only raw facts (port, duration, packet counts,
     bytes, window size). No interpretive language ("Slowloris", "LOIC",
     "characteristic of", "signature", etc.).
  2. Few-shot examples are factual-only: same flat description format with the
     true label. No discriminative hints.

The model, sample seed (random_state=42), and class distribution are held
constant to isolate the contribution of the NL feature engineering layer
versus LLM reasoning.

Output: results/slm_nl_classification_stripped.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from llm_client import ZAIClient


# ─────────────────────────────────────────────────────────────────────────────
# Stripped NL feature conversion: facts only, no interpretation
# ─────────────────────────────────────────────────────────────────────────────

def describe_port(port: int) -> str:
    return {80: "HTTP", 443: "HTTPS", 22: "SSH", 21: "FTP", 53: "DNS",
            3389: "RDP"}.get(int(port), f"port {int(port)}")


def features_to_nl_stripped(row: pd.Series) -> str:
    """Pure factual description. No keywords like Slowloris/LOIC/signature/etc."""
    dst_port = int(row.get("Dst Port", 0))
    dur_us = float(row.get("Flow Duration", 0))
    dur_s = dur_us / 1_000_000.0
    fwd_pkts = int(row.get("Tot Fwd Pkts", 0))
    bwd_pkts = int(row.get("Tot Bwd Pkts", 0))
    win_fwd = int(row.get("Init Fwd Win Byts", 0))
    sent = int(row.get("TotLen Fwd Pkts", 0))
    recv = int(row.get("TotLen Bwd Pkts", 0))
    pps = float(row.get("Flow Pkts/s", 0))

    # Format duration human-readably without judging it
    if dur_s >= 1.0:
        dur_str = f"{dur_s:.2f} s"
    elif dur_s >= 0.001:
        dur_str = f"{dur_s * 1000:.2f} ms"
    else:
        dur_str = f"{dur_us:.0f} microseconds"

    return (
        f"Destination: {describe_port(dst_port)}. "
        f"Duration: {dur_str}. "
        f"Packets: {fwd_pkts} forward, {bwd_pkts} backward. "
        f"Bytes: {sent} forward, {recv} backward. "
        f"TCP initial window: {win_fwd}. "
        f"Packet rate: {pps:.1f} pkt/s."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stripped few-shot prompt: same format, true labels, NO interpretation
# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOT_STRIPPED = """Examples (factual descriptions with true labels):

Example 1 (Benign): "Destination: HTTP. Duration: 5.30 s. Packets: 3 forward, 1 backward. Bytes: 0 forward, 0 backward. TCP initial window: 8192. Packet rate: 0.7 pkt/s."

Example 2 (Benign): "Destination: DNS. Duration: 1.06 ms. Packets: 1 forward, 1 backward. Bytes: 29 forward, 61 backward. TCP initial window: -1. Packet rate: 1887.0 pkt/s."

Example 3 (DoS): "Destination: HTTP. Duration: 1.50 ms. Packets: 2 forward, 0 backward. Bytes: 0 forward, 0 backward. TCP initial window: 225. Packet rate: 1333.3 pkt/s."

Example 4 (DoS): "Destination: HTTP. Duration: 10.10 ms. Packets: 2 forward, 0 backward. Bytes: 0 forward, 0 backward. TCP initial window: 225. Packet rate: 198.0 pkt/s."

Example 5 (DDoS): "Destination: HTTP. Duration: 1.50 ms. Packets: 3 forward, 4 backward. Bytes: 0 forward, 0 backward. TCP initial window: 65535. Packet rate: 4666.7 pkt/s."

Example 6 (DDoS): "Destination: HTTP. Duration: 8.70 ms. Packets: 3 forward, 4 backward. Bytes: 0 forward, 0 backward. TCP initial window: 65535. Packet rate: 805.0 pkt/s."

Example 7 (BruteForce): "Destination: HTTP. Duration: 56.00 s. Packets: 153 forward, 104 backward. Bytes: 53700 forward, 70800 backward. TCP initial window: 8192. Packet rate: 4.6 pkt/s."

Example 8 (BruteForce): "Destination: HTTP. Duration: 57.00 s. Packets: 203 forward, 104 backward. Bytes: 54800 forward, 185500 backward. TCP initial window: 8192. Packet rate: 5.4 pkt/s."

Example 9 (WebAttack): "Destination: HTTP. Duration: 5.00 s. Packets: 4 forward, 4 backward. Bytes: 660 forward, 526 backward. TCP initial window: 8192. Packet rate: 1.6 pkt/s."

Example 10 (WebAttack): "Destination: HTTP. Duration: 5.00 s. Packets: 4 forward, 4 backward. Bytes: 483 forward, 1910 backward. TCP initial window: 8192. Packet rate: 1.6 pkt/s."

Example 11 (Infiltration): "Destination: port 51678. Duration: 0 microseconds. Packets: 2 forward, 0 backward. Bytes: 0 forward, 0 backward. TCP initial window: 31. Packet rate: 0.0 pkt/s."

Example 12 (Infiltration): "Destination: port 54045. Duration: 0 microseconds. Packets: 1 forward, 1 backward. Bytes: 0 forward, 0 backward. TCP initial window: 1024. Packet rate: 0.0 pkt/s."

"""


SYSTEM_MSG = (
    "You are a cybersecurity classifier. "
    "You will be given a natural-language description of a network flow and a set of "
    "labelled examples. "
    "Your task is to classify the flow into exactly one of the provided categories. "
    "Reply with ONLY the category name — no explanation, no punctuation, nothing else."
)


def classify_flow_zai(nl_description: str, classes: list, client: ZAIClient) -> str:
    user_msg = (
        FEW_SHOT_STRIPPED
        + f"Categories: {', '.join(classes)}\n\n"
        + f"Now classify this flow:\n{nl_description}\n\n"
        + "Reply with ONLY the category name."
    )
    response = client.generate(SYSTEM_MSG, user_msg)
    if response is None:
        return "ERROR"
    response = response.strip()
    for cls in classes:
        if cls.lower() in response.lower():
            return cls
    return response[:50]


def run(data_path: Path, n_samples: int, model: str) -> dict:
    print(f"[ablation-stripped] Loading {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    classes = sorted(df["Label"].unique().tolist())
    print(f"  {len(df):,} rows, classes: {classes}")

    samples_per_class = max(2, n_samples // len(classes))
    sample_dfs = []
    for cls in classes:
        cls_df = df[df["Label"] == cls]
        n = min(len(cls_df), samples_per_class)
        sample_dfs.append(cls_df.sample(n=n, random_state=42))
    sample = pd.concat(sample_dfs, ignore_index=True).sample(frac=1, random_state=42)
    print(f"  Sampled: {len(sample)} rows ({samples_per_class} per class) — same seed=42 as v3 run")

    print("\n[ablation-stripped] Sample stripped NL conversions:")
    for i in range(min(3, len(sample))):
        row = sample.iloc[i]
        print(f"  [{row['Label']}] {features_to_nl_stripped(row)}")

    print(f"\n[ablation-stripped] Calling ZAI API (model={model}) ...")
    client = ZAIClient(model=model, temperature=0, max_tokens=4096)

    results = []
    correct = 0
    errors = 0
    t0 = time.time()

    for i, (_, row) in enumerate(sample.iterrows()):
        true_label = row["Label"]
        nl = features_to_nl_stripped(row)
        pred = classify_flow_zai(nl, classes, client)
        is_correct = (pred == true_label)
        if pred == "ERROR":
            errors += 1
        if is_correct:
            correct += 1
        results.append({
            "true": true_label,
            "predicted": pred,
            "correct": is_correct,
            "nl_description": nl,
        })
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            valid_so_far = i + 1 - errors
            acc_so_far = correct / valid_so_far if valid_so_far > 0 else 0
            print(f"  [{i+1}/{len(sample)}] acc={acc_so_far:.2%}  errors={errors}  elapsed={elapsed:.1f}s")

    valid = len(sample) - errors
    accuracy = correct / valid if valid > 0 else 0.0

    per_class_acc = {}
    per_class_counts = {}
    for cls in classes:
        cls_results = [r for r in results if r["true"] == cls and r["predicted"] != "ERROR"]
        if cls_results:
            per_class_acc[cls] = sum(r["correct"] for r in cls_results) / len(cls_results)
            per_class_counts[cls] = len(cls_results)

    elapsed_total = time.time() - t0
    print(f"\n[ablation-stripped] Final accuracy: {accuracy:.2%}  ({correct}/{valid} correct, {errors} API errors)")
    print(f"[ablation-stripped] Total time: {elapsed_total:.1f}s")

    return {
        "mode": "ablation_stripped",
        "model": model,
        "backend": "zai_api",
        "dataset": "cicids",
        "n_samples": len(sample),
        "n_valid": valid,
        "n_errors": errors,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_acc,
        "per_class_counts": per_class_counts,
        "elapsed_seconds": elapsed_total,
        "note": (
            "Ablation: stripped NL features (no signature hints) + factual few-shot "
            "examples. Held constant: random_state=42 (same 60 samples as W17 v3 run), "
            "model=glm-4.7, temperature=0. Compares against slm_nl_classification.json "
            "to isolate LLM reasoning from NL feature engineering contribution."
        ),
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--model", default="glm-4.7")
    args = parser.parse_args()

    out = run(
        data_path=ROOT / "data" / "processed" / "cleaned_dataset.csv",
        n_samples=args.samples,
        model=args.model,
    )

    out_path = ROOT / "results" / "slm_nl_classification_stripped.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[ablation-stripped] Saved to {out_path}")
