"""
Ablation Study — Pre / Post / Tri-Stage with XGBoost (RQ2 enhancement)
======================================================================
Same 5-condition ablation as run_ablation_v2.py but with XGBoost as the
inference engine instead of Random Forest. Measures whether the same
SHACL pre/post stages provide equivalent marginal benefit over a stronger
baseline (XGBoost 95.90% vs RF 87.70%).

Conditions:
  baseline      : raw XGBoost
  pre_only      : pre-inference symbolic features → XGBoost
  post_annotate : XGBoost predictions + GRC annotation (zero cost)
  post_override  : XGBoost predictions + calibrated overrides (all disabled)
  tri_stage      : pre_only → XGBoost → post_annotate

Saves: results/ablation_study_xgb.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PRE_FEATURES = {
    "sym_high_volume":    lambda r: 1 if r.get("Tot Fwd Pkts", 0) > 1000 else 0,
    "sym_short_duration": lambda r: 1 if r.get("Flow Duration", 0) < 10 else 0,
    "sym_suspicious_iat": lambda r: 1 if r.get("Flow IAT Std", 999) < 5 else 0,
    "sym_syn_flood":      lambda r: 1 if r.get("SYN Flag Cnt", 0) > 50 else 0,
    "sym_large_transfer": lambda r: 1 if r.get("TotLen Fwd Pkts", 0) > 100000 else 0,
}


def add_pre_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, fn in PRE_FEATURES.items():
        df[col] = df.apply(fn, axis=1)
    return df


def post_annotate(row, pred: str) -> str:
    return pred  # zero-cost annotation — prediction unchanged


def post_override(row, pred: str, conf: float = 1.0) -> str:
    return pred  # all 4 override rules disabled per calibration


def get_feature_cols(df: pd.DataFrame, extra_cols=None):
    exclude = {"Label", "label", "Timestamp", "Flow ID", "Source IP", "Destination IP",
               "Source Port", "Destination Port", "Protocol"}
    base = [c for c in df.columns if c not in exclude
            and df[c].dtype in (float, int, "float64", "int64")]
    if extra_cols:
        base += [c for c in extra_cols if c in df.columns and c not in base]
    return base


def train_and_eval(X_train, y_train_enc, X_test, y_test_str, le: LabelEncoder,
                   name: str, post_func=None, apply_confidence: bool = False,
                   test_df=None) -> dict:
    clf = XGBClassifier(
        max_depth=6, learning_rate=0.1, n_estimators=500,
        random_state=42, eval_metric="mlogloss",
        use_label_encoder=False, n_jobs=-1, verbosity=0,
    )
    clf.fit(X_train, y_train_enc)
    y_pred_enc = clf.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    if post_func and test_df is not None:
        if apply_confidence:
            y_proba = clf.predict_proba(X_test)
            y_pred_list = list(y_pred)
            for i, (_, row) in enumerate(test_df.iterrows()):
                conf = float(max(y_proba[i]))
                y_pred_list[i] = post_func(row, y_pred_list[i], conf)
            y_pred = np.array(y_pred_list)
        else:
            y_pred_list = list(y_pred)
            for i, (_, row) in enumerate(test_df.iterrows()):
                y_pred_list[i] = post_func(row, y_pred_list[i])
            y_pred = np.array(y_pred_list)

    res = {
        "condition": name,
        "accuracy": float(accuracy_score(y_test_str, y_pred)),
        "f1_macro": float(f1_score(y_test_str, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test_str, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_test_str, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test_str, y_pred, average="macro", zero_division=0)),
        "per_class_report": classification_report(y_test_str, y_pred, output_dict=True, zero_division=0),
        "samples_test": len(y_test_str),
    }
    print(f"  [{name}] accuracy={res['accuracy']:.4f}  f1_macro={res['f1_macro']:.4f}")
    return res


def run_ablation(data_path: Path) -> dict:
    print(f"[XGB-Ablation] Loading {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    print(f"  {len(df):,} rows, {df['Label'].nunique()} classes")

    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["Label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["Label"])

    le = LabelEncoder()
    le.fit(df["Label"])

    train_pre = add_pre_features(train_df)
    test_pre = add_pre_features(test_df)

    base_cols = get_feature_cols(df)
    pre_cols = get_feature_cols(train_pre, list(PRE_FEATURES.keys()))

    y_train = le.transform(train_df["Label"])
    y_test_str = test_df["Label"].values

    print("\n[XGB-Ablation] Running 5 conditions ...")

    c_baseline = train_and_eval(
        train_df[base_cols].values, y_train,
        test_df[base_cols].values, y_test_str, le,
        "baseline",
    )
    c_pre_only = train_and_eval(
        train_pre[pre_cols].values, y_train,
        test_pre[pre_cols].values, y_test_str, le,
        "pre_only",
    )
    c_post_annotate = train_and_eval(
        train_df[base_cols].values, y_train,
        test_df[base_cols].values, y_test_str, le,
        "post_annotate", post_func=post_annotate, test_df=test_df,
    )
    c_post_override = train_and_eval(
        train_df[base_cols].values, y_train,
        test_df[base_cols].values, y_test_str, le,
        "post_override", post_func=post_override,
        apply_confidence=True, test_df=test_df,
    )
    c_tri_stage = train_and_eval(
        train_pre[pre_cols].values, y_train,
        test_pre[pre_cols].values, y_test_str, le,
        "tri_stage", post_func=post_annotate, test_df=test_df,
    )

    results = {
        "model": "XGBoost",
        "dataset": "CSE-CIC-IDS2018",
        "data_source": str(data_path),
        "split_type": "stratified_70_15_15",
        "xgb_params": {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 500},
        "pre_symbolic_features": list(PRE_FEATURES.keys()),
        "baseline": c_baseline,
        "pre_only": c_pre_only,
        "post_annotate": c_post_annotate,
        "post_override": c_post_override,
        "tri_stage": c_tri_stage,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = ROOT / "results" / "ablation_study_xgb.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[XGB-Ablation] Saved to {out_path}")
    for cond in ["baseline", "pre_only", "post_annotate", "post_override", "tri_stage"]:
        r = results[cond]
        print(f"  {cond:<15} acc={r['accuracy']:.4f}  f1_macro={r['f1_macro']:.4f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/cleaned_dataset.csv")
    args = parser.parse_args()
    run_ablation(ROOT / args.data)
