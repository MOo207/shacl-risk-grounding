#!/usr/bin/env python3
"""
CSE-CIC-IDS2018 Dataset Validator
====================================
Run after download_dataset.py to verify files are complete and clean.

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py data/raw/CSE-CIC-IDS2018/
    python scripts/validate_dataset.py data/raw/CSE-CIC-IDS2018/Thursday-15-02-2018_TrafficForML_CaptureFlowID.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "CSE-CIC-IDS2018"
EXPECTED_COLS = 80

LABEL_NORM = {
    "Infilteration": "Infiltration",
    "infilteration": "Infiltration",
    "BENIGN":        "Benign",
}


def find_csvs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            print(f"[ERROR] No CSV files in {path}")
            sys.exit(1)
        return files
    print(f"[ERROR] Not found: {path}")
    sys.exit(1)


def check_file(csv_path: Path):
    size_mb = csv_path.stat().st_size / 1e6
    print(f"\n{'='*60}")
    print(f"File  : {csv_path.name}")
    print(f"Size  : {size_mb:.1f} MB")

    # Load with encoding fallback
    df = None
    for enc in ("utf-8", "latin-1", "ISO-8859-1"):
        try:
            df = pd.read_csv(csv_path, encoding=enc, low_memory=False, nrows=300_000)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        print("[ERROR] Failed to decode file")
        return

    print(f"Rows  : {len(df):,} (sampled up to 300K)")

    # Column name leading spaces
    raw_cols = list(df.columns)
    df.columns = [c.strip() for c in raw_cols]
    spaces = sum(1 for a, b in zip(raw_cols, df.columns) if a != b)
    if spaces:
        print(f"[WARN]  {spaces} column name(s) had leading spaces — stripped")
    else:
        print(f"[OK]    Column names clean (no leading spaces)")

    # Column count
    n = len(df.columns)
    tag = "[OK]   " if n == EXPECTED_COLS else "[WARN] "
    print(f"{tag} {n} columns (expected {EXPECTED_COLS})")

    # Label column
    if "Label" not in df.columns:
        cands = [c for c in df.columns if "label" in c.lower()]
        print(f"[ERROR] 'Label' column missing. Candidates: {cands}")
        return
    else:
        print(f"[OK]    Label column present")

    # Normalise labels
    df["Label"] = df["Label"].str.strip().map(lambda x: LABEL_NORM.get(x, x))
    dist = df["Label"].value_counts()
    print(f"\nLabel distribution ({len(dist)} unique):")
    for lbl, cnt in dist.items():
        print(f"  {lbl:<45} {cnt:>8,}  ({100*cnt/len(df):.2f}%)")

    # Inf/NaN
    num = df.select_dtypes(include=[np.number])
    n_inf = (np.isinf(num)).sum().sum()
    n_nan = num.isna().sum().sum()
    print(f"\n[{'WARN' if n_inf else 'OK  '}]  Inf values : {n_inf:,}")
    print(f"[{'WARN' if n_nan else 'OK  '}]  NaN values : {n_nan:,}")

    # Negative IAT
    iat_cols = [c for c in df.columns if "iat" in c.lower()]
    n_neg = sum((pd.to_numeric(df[c], errors="coerce") < 0).sum() for c in iat_cols)
    print(f"[{'WARN' if n_neg else 'OK  '}]  Negative IAT values : {n_neg:,}")

    # Duplicates
    n_dup = df.duplicated().sum()
    print(f"[{'WARN' if n_dup else 'OK  '}]  Duplicate rows : {n_dup:,}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR
    csvs = find_csvs(path)
    print(f"Validating {len(csvs)} file(s) in {path}")

    for f in csvs:
        check_file(f)

    print(f"\n{'='*60}")
    print("Validation complete.")
    print("Fix issues automatically:")
    print("  python scripts/prepare_slm_dataset.py  # cleans + samples 50K rows")
    print("  (loader applies same fixes at runtime for full pipeline)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
