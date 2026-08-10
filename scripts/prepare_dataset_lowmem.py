#!/usr/bin/env python3
"""Memory-efficient dataset preparation. Stratified sampling per-file with rare class preservation."""
import json, sys, warnings, gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'CSE-CIC-IDS2018'
OUT_DIR  = Path(__file__).parent.parent / 'data' / 'processed'
SEED = 42

LABEL_MAP = {
    'Benign': 'Benign', 'BENIGN': 'Benign',
    'FTP-BruteForce': 'BruteForce', 'SSH-Bruteforce': 'BruteForce',
    'Brute Force -Web': 'BruteForce', 'Brute Force -XSS': 'BruteForce',
    'DoS attacks-Hulk': 'DoS', 'DoS attacks-SlowHTTPTest': 'DoS',
    'DoS attacks-GoldenEye': 'DoS', 'DoS attacks-Slowloris': 'DoS',
    'SQL Injection': 'WebAttack', 'XSS': 'WebAttack',
    'Infiltration': 'Infiltration', 'Infilteration': 'Infiltration',
    'Bot': 'Bot',
    'DDOS attack-HOIC': 'DDoS', 'DDoS attacks-LOIC-HTTP': 'DDoS',
    'DDOS attack-LOIC-UDP': 'DDoS',
}

def process_csv(csv_path):
    """Process one CSV in chunks. Keep ALL rare class rows, sample majority classes."""
    rare_rows = []   # Keep all rows from rare classes
    majority_reservoir = []  # Sample from majority classes
    majority_count = 0
    
    for chunk in pd.read_csv(csv_path, chunksize=20000, low_memory=False, on_bad_lines='skip'):
        chunk.columns = chunk.columns.str.strip()
        if 'Label' not in chunk.columns:
            continue
        chunk = chunk[chunk['Label'] != 'Label']
        if chunk.empty:
            continue
        
        for col in chunk.columns:
            if col != 'Label':
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        chunk = chunk.fillna(0)
        chunk['Label'] = chunk['Label'].str.strip().map(LABEL_MAP).fillna(chunk['Label'].str.strip())
        chunk = chunk.drop_duplicates()
        
        # Separate rare and majority
        for label in chunk['Label'].unique():
            label_rows = chunk[chunk['Label'] == label]
            if label in ('WebAttack', 'Bot', 'Infiltration', 'BruteForce'):
                rare_rows.append(label_rows)
            else:
                # Reservoir sample for majority (Benign, DoS, DDoS)
                majority_count += len(label_rows)
                n = min(50, len(label_rows))
                majority_reservoir.append(label_rows.sample(n=n, random_state=SEED))
        del chunk
    
    gc.collect()
    
    parts = []
    if rare_rows:
        parts.append(pd.concat(rare_rows, ignore_index=True))
        del rare_rows
    if majority_reservoir:
        parts.append(pd.concat(majority_reservoir, ignore_index=True))
        del majority_reservoir
    gc.collect()
    
    if not parts:
        return pd.DataFrame()
    
    result = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return result

def main():
    csvs = sorted(DATA_DIR.glob('*.csv'))
    if not csvs:
        print(f'No CSVs in {DATA_DIR}')
        sys.exit(1)

    print(f'Processing {len(csvs)} files...')
    all_data = []
    
    for csv_path in csvs:
        print(f'\n{csv_path.name} ({csv_path.stat().st_size/1e6:.0f}MB)...')
        df = process_csv(csv_path)
        if not df.empty:
            print(f'  Collected: {len(df)} rows')
            for lbl, cnt in df['Label'].value_counts().items():
                print(f'    {lbl}: {cnt}')
            all_data.append(df)
        del df
        gc.collect()

    full = pd.concat(all_data, ignore_index=True).drop_duplicates()
    del all_data
    gc.collect()
    
    print(f'\nCombined: {len(full)} rows')
    
    # Stratified sample: ensure 50+ rows per class, balance the rest
    TARGET = 25000
    classes = full['Label'].unique()
    MIN_PER_CLASS = 50
    
    parts = []
    for cls in classes:
        cls_df = full[full['Label'] == cls]
        # Ensure at least MIN_PER_CLASS (or all if fewer)
        n = max(MIN_PER_CLASS, TARGET // len(classes))
        n = min(n, len(cls_df))
        parts.append(cls_df.sample(n=n, random_state=SEED))
    
    full = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED)
    del parts
    gc.collect()

    print(f'\nFinal dataset: {len(full)} rows')
    for lbl, cnt in full['Label'].value_counts().items():
        print(f'  {lbl:<20} {cnt:>6} ({100*cnt/len(full):.1f}%)')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / 'cleaned_dataset.csv'
    full.to_csv(out_csv, index=False)
    print(f'\nSaved: {out_csv} ({out_csv.stat().st_size/1e6:.1f}MB)')

    train_df, test_df = train_test_split(full, test_size=0.2, random_state=SEED, stratify=full['Label'])
    
    def to_jsonl(df, path):
        cols = [c for c in df.columns if c != 'Label']
        with open(path, 'w') as f:
            for _, row in df.iterrows():
                top5 = sorted([(c, float(row[c])) for c in cols if row[c] != 0],
                              key=lambda x: abs(x[1]), reverse=True)[:5]
                features = ', '.join(f'{k}={v:.2f}' for k, v in top5)
                f.write(json.dumps({
                    'messages': [
                        {'role': 'system', 'content': 'You are a network intrusion detection expert.'},
                        {'role': 'user', 'content': f'Classify: {features}'},
                        {'role': 'assistant', 'content': f'{row['Label']}'}
                    ]
                }) + '\n')
        print(f'Saved: {path} ({len(df)} records)')

    to_jsonl(train_df, OUT_DIR / 'slm_train.jsonl')
    to_jsonl(test_df, OUT_DIR / 'slm_test.jsonl')
    print('\nT-A1 COMPLETE')

if __name__ == '__main__':
    main()
