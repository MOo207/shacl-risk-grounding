#!/usr/bin/env python3
"""SLM evaluation via local Ollama (qwen3.5:4b). No API key needed."""
import json, sys, time, argparse
from pathlib import Path
import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:0.8b"
RESULTS_DIR = Path("results")

def classify_flow(features_text, classes, timeout=300):
    prompt = (
        "You are a network intrusion detection expert.\n"
        "Given these network flow features:\n"
        f"{features_text}\n\n"
        f"Classify this flow as EXACTLY one of: {', '.join(classes)}\n"
        "Reply with ONLY the class label, nothing else."
    )
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_predict": 20}
        }, timeout=timeout)
        resp = r.json().get("response", "").strip()
        for cls in classes:
            if cls.lower() in resp.lower():
                return cls
        return resp[:50]
    except Exception as e:
        return f"ERROR: {e}"

def load_dataset(name):
    if name == "cicids":
        df = pd.read_csv("data/processed/cleaned_dataset.csv")
        classes = sorted(df["Label"].unique().tolist())
        return df, classes, "Label"
    elif name == "nslkdd":
        df = pd.read_csv("data/processed/nslkdd_cleaned.csv")
        classes = sorted(df["label"].unique().tolist())
        return df, classes, "label"
    else:
        sys.exit(f"Unknown dataset: {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cicids", "nslkdd"])
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()

    df, classes, label_col = load_dataset(args.dataset)
    # Get stratified sample
    samples_per_class = max(2, args.samples // len(classes))
    sampled_dfs = []
    for cls in classes:
        class_df = df[df[label_col] == cls]
        n_samples = min(len(class_df), samples_per_class)
        sampled_dfs.append(class_df.sample(n_samples, random_state=42))
    sample = pd.concat(sampled_dfs, ignore_index=True)
    feature_cols = [c for c in df.columns if c != label_col]

    print(f"Evaluating {len(sample)} samples from {args.dataset} with {MODEL}")
    results = []
    correct = 0
    per_class = {c: {"tp": 0, "total": 0} for c in classes}

    for i, (_, row) in enumerate(sample.iterrows()):
        true_label = row[label_col]
        feats = sorted([(c, float(row[c])) for c in feature_cols if row[c] != 0],
                       key=lambda x: abs(x[1]), reverse=True)[:10]
        feat_text = ", ".join(f"{k}={v:.2f}" for k, v in feats)
        pred = classify_flow(feat_text, classes)
        is_correct = pred == true_label
        if is_correct: correct += 1
        per_class[true_label]["total"] += 1
        if is_correct: per_class[true_label]["tp"] += 1
        results.append({"true": true_label, "predicted": pred, "correct": is_correct})
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(sample)}] acc: {correct/(i+1):.2%}")

    accuracy = correct / len(sample)
    per_class_acc = {c: v["tp"]/v["total"] if v["total"] > 0 else 0 for c, v in per_class.items()}
    output = {"dataset": args.dataset, "model": MODEL, "samples": len(sample),
              "accuracy": accuracy, "per_class_accuracy": per_class_acc, "results": results}
    out_path = RESULTS_DIR / f"{args.dataset}_slm.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Per-class: {per_class_acc}")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
