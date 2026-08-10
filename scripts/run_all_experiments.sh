#!/bin/bash
# Master script: runs T-A1 through T-A7 sequentially with verification
set -e
cd /root/repos/INTERSYMBOLIC-GRC

echo '=== INTERSYMBOLIC-GRC Experiment Runner ==='
echo "Started: $(date -u)"

# T-A1: Dataset preparation
if [ ! -f data/processed/cleaned_dataset.csv ]; then
    echo '[T-A1] Running dataset preparation...'
    python3 scripts/prepare_dataset_lowmem.py --rows 25000 --per-file 5000
    [ -f data/processed/cleaned_dataset.csv ] || { echo 'T-A1 FAILED'; exit 1; }
    echo "[T-A1] DONE: $(wc -l < data/processed/cleaned_dataset.csv) rows"
else
    echo "[T-A1] SKIP: already exists ($(wc -l < data/processed/cleaned_dataset.csv) rows)"
fi

# T-A2: RF Baseline
if [ ! -f results/rf_baseline.json ]; then
    echo '[T-A2] Running RF baseline...'
    python3 scripts/run_rf_baseline.py
    [ -f results/rf_baseline.json ] || { echo 'T-A2 FAILED'; exit 1; }
    echo "[T-A2] DONE: $(du -h results/rf_baseline.json | cut -f1)"
else
    echo '[T-A2] SKIP: already exists'
fi

# T-A3: XGBoost Baseline
if [ ! -f results/xgb_baseline.json ]; then
    echo '[T-A3] Running XGBoost baseline...'
    pip install xgboost -q 2>/dev/null
    python3 scripts/run_xgb_baseline.py
    [ -f results/xgb_baseline.json ] || { echo 'T-A3 FAILED'; exit 1; }
    echo "[T-A3] DONE: $(du -h results/xgb_baseline.json | cut -f1)"
else
    echo '[T-A3] SKIP: already exists'
fi

# T-A4: Rule Baseline
if [ ! -f results/rule_baseline.json ]; then
    echo '[T-A4] Running rule baseline...'
    python3 scripts/run_rule_baseline.py
    [ -f results/rule_baseline.json ] || { echo 'T-A4 FAILED'; exit 1; }
    echo "[T-A4] DONE: $(du -h results/rule_baseline.json | cut -f1)"
else
    echo '[T-A4] SKIP: already exists'
fi

# T-A5: Ablation Study (needs rf_model.joblib from T-A2)
if [ ! -f results/ablation_study.json ] || [ results/ablation_study.json -ot results/rf_baseline.json ]; then
    echo '[T-A5] Running ablation study...'
    python3 scripts/run_ablation.py
    [ -f results/ablation_study.json ] || { echo 'T-A5 FAILED'; exit 1; }
    echo "[T-A5] DONE: $(du -h results/ablation_study.json | cut -f1)"
else
    echo '[T-A5] SKIP: already exists (newer than RF baseline)'
fi

# T-A6: SHAP XAI (needs rf_model.joblib from T-A2)
if [ ! -f results/shap_top20_features.json ]; then
    echo '[T-A6] Running SHAP analysis...'
    pip install shap -q 2>/dev/null
    python3 scripts/run_xai_analysis.py
    [ -f results/shap_top20_features.json ] || { echo 'T-A6 FAILED'; exit 1; }
    echo "[T-A6] DONE: $(du -h thesis/figures/fig-shap-beeswarm.png | cut -f1)"
else
    echo '[T-A6] SKIP: already exists'
fi

# T-A7: 4-Layer Explanation (needs T-A5 + T-A6)
if [ ! -f results/intersymbolic_explanations.json ]; then
    echo '[T-A7] Running intersymbolic explanations...'
    python3 scripts/run_intersymbolic_explanation.py --no-ollama
    [ -f results/intersymbolic_explanations.json ] || { echo 'T-A7 FAILED'; exit 1; }
    echo "[T-A7] DONE: $(du -h results/intersymbolic_explanations.json | cut -f1)"
else
    echo '[T-A7] SKIP: already exists'
fi

echo ''
echo '=== ALL EXPERIMENTS COMPLETE ==='
echo "Finished: $(date -u)"
echo ''
echo 'Results:'
ls -lh results/*.json results/*.npy results/*.joblib 2>/dev/null
echo ''
echo 'Figures:'
ls -lh thesis/figures/*.png 2>/dev/null

# Commit everything
git add results/ data/processed/ thesis/figures/
git commit -m "feat: complete empirical results T-A1 through T-A7

T-A1: Dataset preparation (25K rows)
T-A2: RF baseline
T-A3: XGBoost baseline
T-A4: Rule baseline
T-A5: Ablation study
T-A6: SHAP XAI analysis
T-A7: 4-layer intersymbolic explanations

All results verified on disk." || echo 'Nothing to commit'
git push --set-upstream origin HEAD || echo 'Push failed'
