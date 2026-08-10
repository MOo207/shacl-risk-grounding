#!/usr/bin/env python3
"""
CSE-CIC-IDS2018 SLM Dataset Preparation
=========================================
Reads the 6 downloaded CSV files, cleans them, and produces:

  1. data/processed/cleaned_dataset.csv     -- merged clean dataset for ML baselines
  2. data/processed/slm_train.jsonl         -- instruction-tuning JSONL for Qwen3.5:4b
  3. data/processed/slm_test.jsonl          -- held-out test set (20%)

Cleaning steps applied:
  - Strip whitespace from column names
  - Drop rows where Label == "Label" (corrupt embedded headers)
  - Replace Inf with NaN, then fill NaN with 0
  - Clip negative IAT values to 0
  - Drop exact duplicate rows
  - Normalise label strings (Infilteration -> Infiltration, BENIGN -> Benign)

Stratified sampling:
  - Target: 50,000 training rows + 12,500 test rows (80/20)
  - Per-class cap: min(class_size, target_per_class) so rare classes are not over-dropped
  - Benign rows are downsampled to match total attack rows (balanced dataset)

Instruction-tuning format (Qwen3.5:4b chat template):
  System: "You are a network intrusion detection expert..."
  User:   "<feature-table or summary>"
  Assistant: "<label + brief explanation>"

Usage:
    python scripts/prepare_slm_dataset.py
    python scripts/prepare_slm_dataset.py --rows 100000  # larger sample
    python scripts/prepare_slm_dataset.py --dry-run      # stats only, no write
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_DIR   = Path(__file__).parent.parent / "data" / "raw" / "CSE-CIC-IDS2018"
OUT_DIR    = Path(__file__).parent.parent / "data" / "processed"
RANDOM_SEED = 42

# Canonical 8-class taxonomy (Benign + 7 attack super-classes)
LABEL_MAP = {
    # Benign
    "Benign":                       "Benign",
    "BENIGN":                       "Benign",
    # Brute Force
    "FTP-BruteForce":               "BruteForce",
    "SSH-Bruteforce":               "BruteForce",
    "Brute Force -Web":             "BruteForce",
    "Brute Force -XSS":             "BruteForce",
    # DoS
    "DoS attacks-Hulk":             "DoS",
    "DoS attacks-SlowHTTPTest":     "DoS",
    "DoS attacks-GoldenEye":        "DoS",
    "DoS attacks-Slowloris":        "DoS",
    # Web Attack
    "SQL Injection":                "WebAttack",
    "XSS":                          "WebAttack",
    # Infiltration
    "Infiltration":                 "Infiltration",
    "Infilteration":                "Infiltration",
    "infilteration":                "Infiltration",
    # Bot
    "Bot":                          "Bot",
    # DDoS
    "DDOS attack-HOIC":             "DDoS",
    "DDoS attacks-LOIC-HTTP":       "DDoS",
    "DDOS attack-LOIC-UDP":         "DDoS",
    "DDoS attacks-LOIC-UDP":        "DDoS",
    "DDOS-LOIC-HTTP":               "DDoS",
    "DDOS-HOIC":                    "DDoS",
}

ATTACK_DESCRIPTIONS = {
    "Benign":       "Normal network traffic — no attack present.",
    "BruteForce":   "Credential brute-force attack (FTP, SSH, or web login repeated attempts).",
    "DoS":          "Denial-of-Service attack flooding the target with requests to exhaust resources.",
    "WebAttack":    "Web application attack: SQL injection or cross-site scripting (XSS) payload.",
    "Infiltration": "Stealthy infiltration: attacker gains internal access via exploit or lateral movement.",
    "Bot":          "Bot/botnet activity: compromised host executing C2 commands (Ares botnet).",
    "DDoS":         "Distributed Denial-of-Service: coordinated flood from multiple sources (LOIC/HOIC).",
}

FEATURE_COLS = [
    "Dst Port", "Protocol",
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean",
    "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags",
    "Fwd Header Len", "Bwd Header Len",
    "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
    "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts",
    "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Load one CSV, apply all cleaning steps, return cleaned DataFrame."""
    df = None
    for enc in ("utf-8", "latin-1", "ISO-8859-1"):
        try:
            df = pd.read_csv(csv_path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        print(f"  [ERROR] Could not decode {csv_path.name}")
        return pd.DataFrame()

    # Strip column name whitespace
    df.columns = [c.strip() for c in df.columns]

    # Drop embedded header rows (Label == "Label")
    if "Label" in df.columns:
        before = len(df)
        df = df[df["Label"] != "Label"]
        dropped = before - len(df)
        if dropped:
            print(f"    Dropped {dropped} corrupt header rows")

    # Replace Inf with NaN, fill with 0
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Clip negative IAT columns to 0
    iat_cols = [c for c in df.columns if "iat" in c.lower()]
    for c in iat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)

    # Drop duplicates
    df = df.drop_duplicates()

    # Normalise labels
    if "Label" in df.columns:
        df["Label"] = df["Label"].str.strip().map(lambda x: LABEL_MAP.get(x, x))

    return df


def stratified_sample(df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    """
    Balanced stratified sample:
      - Attack classes: equal share up to class size
      - Benign: downsampled to match total attack rows
    """
    attack_df = df[df["Label"] != "Benign"]
    benign_df = df[df["Label"] == "Benign"]

    classes = attack_df["Label"].unique()
    per_class = total_rows // (len(classes) + 1)  # +1 slot for Benign

    parts = []
    for cls in sorted(classes):
        cls_df = attack_df[attack_df["Label"] == cls]
        n = min(len(cls_df), per_class)
        parts.append(cls_df.sample(n, random_state=RANDOM_SEED))

    attack_sampled = pd.concat(parts)
    benign_n = min(len(benign_df), len(attack_sampled))
    benign_sampled = benign_df.sample(benign_n, random_state=RANDOM_SEED)

    result = pd.concat([attack_sampled, benign_sampled]).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)
    return result


def row_to_instruction(row: pd.Series, available_cols: list) -> dict:
    """Convert one flow row to a chat-style instruction dict."""
    # Build a compact feature summary
    cols = [c for c in available_cols if c in row.index]
    feat_lines = []
    for c in cols[:30]:  # top 30 features for prompt length
        val = row[c]
        if isinstance(val, float):
            feat_lines.append(f"  {c}: {val:.4g}")
        else:
            feat_lines.append(f"  {c}: {val}")

    label = row["Label"]
    description = ATTACK_DESCRIPTIONS.get(label, f"Network flow labelled as {label}.")

    user_msg = (
        "Classify the following network flow and explain your reasoning.\n\n"
        "Network flow features:\n"
        + "\n".join(feat_lines)
    )
    assistant_msg = f"Classification: {label}\n\nExplanation: {description}"

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a network intrusion detection expert. "
                    "Given CICFlowMeter network flow features, classify the traffic "
                    "into one of: Benign, BruteForce, DoS, WebAttack, Infiltration, Bot, DDoS. "
                    "Always state the classification first, then briefly explain the key indicators."
                ),
            },
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def write_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  [OK] {path.name}: {len(records):,} records")


def main():
    parser = argparse.ArgumentParser(description="Prepare SLM dataset from CIC-IDS2018 CSVs")
    parser.add_argument("--rows",    type=int,  default=50_000, help="Target sample rows (default 50000)")
    parser.add_argument("--dry-run", action="store_true",       help="Show stats only, do not write files")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir",  type=Path, default=OUT_DIR)
    args = parser.parse_args()

    csvs = sorted(args.data_dir.glob("*.csv"))
    if not csvs:
        print(f"[ERROR] No CSV files in {args.data_dir}. Run download_dataset.py first.")
        sys.exit(1)

    print("=" * 60)
    print(f"CSE-CIC-IDS2018 SLM Dataset Preparation")
    print(f"  Source files : {len(csvs)}")
    print(f"  Target rows  : {args.rows:,}")
    print(f"  Output dir   : {args.out_dir}")
    print("=" * 60)

    # --- Load + clean all files ---
    frames = []
    for csv_path in csvs:
        print(f"\nLoading {csv_path.name}  ({csv_path.stat().st_size/1e6:.0f} MB)...")
        df = load_and_clean(csv_path)
        if df.empty:
            continue
        dist = df["Label"].value_counts() if "Label" in df.columns else {}
        print(f"  Rows: {len(df):,}   Labels: {dict(list(dist.items())[:5])}")
        frames.append(df)

    if not frames:
        print("[ERROR] No data loaded.")
        sys.exit(1)

    full = pd.concat(frames, ignore_index=True)
    print(f"\n{'='*60}")
    print(f"Combined: {len(full):,} rows")
    print("Label distribution (full):")
    for lbl, cnt in full["Label"].value_counts().items():
        print(f"  {lbl:<20} {cnt:>9,}  ({100*cnt/len(full):.1f}%)")

    # --- Stratified sample ---
    sampled = stratified_sample(full, args.rows)
    print(f"\nStratified sample: {len(sampled):,} rows")
    print("Label distribution (sample):")
    for lbl, cnt in sampled["Label"].value_counts().items():
        print(f"  {lbl:<20} {cnt:>7,}  ({100*cnt/len(sampled):.1f}%)")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # --- Write cleaned CSV (for ML baselines) ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "cleaned_dataset.csv"
    sampled.to_csv(out_csv, index=False)
    print(f"\n  [OK] {out_csv.name}: {len(sampled):,} rows, {out_csv.stat().st_size/1e6:.1f} MB")

    # --- Train/test split ---
    train_df, test_df = train_test_split(
        sampled, test_size=0.2, random_state=RANDOM_SEED, stratify=sampled["Label"]
    )
    print(f"  Train: {len(train_df):,}   Test: {len(test_df):,}")

    # --- Build instruction JSONL ---
    available = [c for c in FEATURE_COLS if c in sampled.columns]
    print(f"  Feature columns available: {len(available)} / {len(FEATURE_COLS)}")

    print("\nBuilding instruction JSONL...")
    train_records = [row_to_instruction(row, available) for _, row in train_df.iterrows()]
    test_records  = [row_to_instruction(row, available) for _, row in test_df.iterrows()]

    write_jsonl(train_records, args.out_dir / "slm_train.jsonl")
    write_jsonl(test_records,  args.out_dir / "slm_test.jsonl")

    print(f"\n{'='*60}")
    print("Done. Next steps:")
    print("  1. ollama pull qwen3.5:4b")
    print("  2. python scripts/finetune_qwen.py")
    print("  3. python scripts/evaluate_approaches.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
