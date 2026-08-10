#!/usr/bin/env python3
"""
NSL-KDD Dataset Preparation
==========================
Reads KDDTrain+.txt and KDDTest+.txt, cleans them, and produces:

  1. data/processed/nslkdd_cleaned.csv     -- merged clean dataset for ML baselines
  2. data/processed/nslkdd_train.jsonl     -- training set
  3. data/processed/nslkdd_test.jsonl      -- test set

NSL-KDD has 5 classes:
  - Normal
  - DoS (Denial of Service)
  - Probe (surveillance and probing)
  - R2L (remote-to-local)
  - U2R (user-to-root)

Source: https://www.unb.ca/cic/datasets/nsl.html
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

DATA_DIR   = Path(__file__).parent.parent / "data" / "raw" / "NSL-KDD"
OUT_DIR    = Path(__file__).parent.parent / "data" / "processed"
RANDOM_SEED = 42

# NSL-KDD label mapping (42 types -> 5 classes)
ATTACK_MAPPING = {
    'normal': 'Normal',
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
    'processtable': 'DoS', 'mailbomb': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L',
    'snmpguess': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',
}

# Column names for NSL-KDD (41 features + label + difficulty)
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load NSL-KDD data from text file."""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, names=COLUMN_NAMES, header=None)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess NSL-KDD data."""
    print("Cleaning data...")

    # Check label distribution before mapping
    print(f"  Original labels: {df['label'].value_counts().to_dict()}")

    # Drop difficulty column (not useful for classification)
    if 'difficulty' in df.columns:
        df = df.drop('difficulty', axis=1)

    # Map 42 attack types to 5 classes
    df['label'] = df['label'].map(ATTACK_MAPPING).fillna('Normal')
    print(f"  Mapped to 5 classes: {df['label'].value_counts().to_dict()}")

    # One-hot encode categorical columns
    categorical_cols = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    print(f"  One-hot encoded {categorical_cols}, new shape: {df.shape}")

    # Convert all columns to numeric
    label_col = df['label']
    df = df.drop('label', axis=1)
    df = df.apply(pd.to_numeric, errors='coerce')
    df['label'] = label_col

    # Fill NaN with 0
    df = df.fillna(0)
    print(f"  Filled NaN values with 0")

    # Drop duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"  Dropped {initial_len - len(df)} duplicate rows")

    return df


def prepare_for_ml(df: pd.DataFrame, train_path: str, test_path: str) -> None:
    """Prepare train/test splits for ML models."""
    print("\nPreparing ML datasets...")

    # Split by class (stratified)
    train_dfs = []
    test_dfs = []

    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        label_train, label_test = train_test_split(
            label_df, test_size=0.2, random_state=RANDOM_SEED
        )
        train_dfs.append(label_train)
        test_dfs.append(label_test)

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    # Shuffle
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"  Train: {len(train_df)} rows")
    print(f"  Test: {len(test_df)} rows")
    print(f"  Train distribution: {train_df['label'].value_counts().to_dict()}")
    print(f"  Test distribution: {test_df['label'].value_counts().to_dict()}")

    # Save
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\n  Saved {train_path}")
    print(f"  Saved {test_path}")


def prepare_for_slm(df: pd.DataFrame, train_path: str, test_path: str) -> None:
    """Prepare instruction-tuning datasets for SLM."""
    print("\nPreparing SLM datasets...")

    # Limit to reasonable sample size for SLM evaluation
    SAMPLE_SIZE = 500  # total samples (100 per class ideally)

    train_dfs = []
    test_dfs = []

    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        total_samples = len(label_df)

        # Calculate samples for this class (4/5 train, 1/5 test)
        train_samples = min(SAMPLE_SIZE // 5 * 4, total_samples // 5 * 4)
        test_samples = min(SAMPLE_SIZE // 5, total_samples // 5)

        if train_samples > 0:
            train_dfs.append(label_df.sample(n=train_samples, random_state=RANDOM_SEED))
        if test_samples > 0:
            test_dfs.append(label_df.sample(n=test_samples, random_state=RANDOM_SEED + 1))

    train_sample = pd.concat(train_dfs, ignore_index=True)
    test_sample = pd.concat(test_dfs, ignore_index=True)

    print(f"  SLM Train: {len(train_sample)} rows")
    print(f"  SLM Test: {len(test_sample)} rows")

    def create_jsonl(df_subset, output_path):
        with open(output_path, 'w') as f:
            for _, row in df_subset.iterrows():
                # Select top 20 features for brevity
                features = row.drop('label').head(20).to_dict()
                entry = {
                    "system": "You are a network intrusion detection expert. Classify the following network flow.",
                    "user": json.dumps(features, indent=2),
                    "assistant": f"{row['label']}"
                }
                f.write(json.dumps(entry) + '\n')
        print(f"  Saved {output_path}")

    create_jsonl(train_sample, train_path)
    create_jsonl(test_sample, test_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare NSL-KDD dataset for ML and SLM")
    parser.add_argument('--dry-run', action='store_true', help='Show stats only, no output')
    args = parser.parse_args()

    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load train and test data
    train_df = load_data(DATA_DIR / "KDDTrain+.txt")
    test_df = load_data(DATA_DIR / "KDDTest+.txt")

    # Merge
    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"\nTotal dataset: {len(df)} rows")

    # Clean
    df = clean_data(df)

    if args.dry_run:
        print("\nDry run - no files written")
        return

    # Prepare for ML
    prepare_for_ml(
        df,
        OUT_DIR / "nslkdd_train.csv",
        OUT_DIR / "nslkdd_test.csv"
    )

    # Prepare for SLM
    prepare_for_slm(
        df,
        OUT_DIR / "nslkdd_slm_train.jsonl",
        OUT_DIR / "nslkdd_slm_test.jsonl"
    )

    # Save full cleaned dataset
    df.to_csv(OUT_DIR / "nslkdd_cleaned.csv", index=False)
    print(f"\nSaved cleaned dataset: {OUT_DIR / 'nslkdd_cleaned.csv'}")

    print("\n✓ NSL-KDD preparation complete!")


if __name__ == "__main__":
    main()
