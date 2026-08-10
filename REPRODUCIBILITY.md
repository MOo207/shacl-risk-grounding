# INTERSYMBOLIC-GRC: Reproducibility Package

This document provides step-by-step instructions for reproducing all experiments and results presented in the INTERSYMBOLIC-GRC thesis.

> **Note:** This repository is the experiments replication package (`scripts/` + `results/`). References below to Docker Compose files and the full pipeline infrastructure describe the complete development environment; the raw experiment outputs in `results/` allow verifying every reported number without rerunning the pipeline.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Running Experiments](#running-experiments)
5. [Expected Results](#expected-results)
6. [Troubleshooting](#troubleshooting)
7. [Citation and License](#citation-and-license)

---

## System Requirements

### Hardware
- **Minimum**: 8 CPU cores, 16 GB RAM, 50 GB disk space
- **Recommended**: 16 CPU cores, 32 GB RAM, 100 GB SSD
- **GPU**: Optional (for TensorFlow/PyTorch acceleration)

### Software
- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10/11 with WSL2
- **Docker**: 20.10+ with Docker Compose 2.0+
- **Python**: 3.11 (included in Docker container)

### Network
- Stable internet connection for:
  - Downloading Docker images
  - Downloading the CSE-CIC-IDS2018 dataset (~8GB)
  - Installing Python packages

---

## Quick Start

The fastest way to reproduce the experiments is using Docker Compose:

```bash
# 1. Clone the repository
git clone https://github.com/MOo207/INTERSYMBOLIC-GRC.git
cd INTERSYMBOLIC-GRC

# 2. Download the dataset
python scripts/download_dataset.py
# Follow the instructions to download CSE-CIC-IDS2018

# 3. Start the Docker containers
docker-compose up -d

# 4. Initialize Neo4j and load SHACL shapes
docker exec intersymbolic-pipeline python scripts/init_neo4j.py

# 5. Load the dataset (replace with actual data path)
docker exec intersymbolic-pipeline python pipeline/loaders/cse_cic_ids2018_loader.py \
    --data-dir /app/data/raw/CSE-CIC-IDS2018

# 6. Train the models
docker exec intersymbolic-pipeline python scripts/train_models.py --model all

# 7. Run comparative experiments
docker exec intersymbolic-pipeline python experiments/comparative_experiments.py

# 8. View results
docker exec intersymbolic-pipeline cat outputs/results.json
```

---

## Detailed Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/MOo207/INTERSYMBOLIC-GRC.git
cd INTERSYMBOLIC-GRC
```

### Step 2: Download the Dataset

The CSE-CIC-IDS2018 dataset is required for training and evaluation.

```bash
# Run the download script for instructions
python scripts/download_dataset.py

# This will create the directory structure and provide download links
```

**Manual Download Instructions:**

1. Visit: https://www.unb.ca/cic/datasets/ids-2018.html
2. Download the machine learning CSV files
3. Place them in: `data/raw/CSE-CIC-IDS2018/`

**Expected Files:**
- `Thursday-15-02-2018_TrafficForML_CaptureFlowID.csv`
- `Friday-16-02-2018_TrafficForML_CaptureFlowID.csv`
- (and other daily CSV files)

### Step 3: Start Docker Containers

```bash
# Start all services (Neo4j, Python pipeline, Jupyter)
docker-compose up -d

# Verify containers are running
docker-compose ps

# View logs
docker-compose logs -f
```

**Services:**
- `neo4j`: Graph database with SHACL validation (ports 7474, 7687)
- `intersymbolic-pipeline`: Main Python environment for experiments
- `jupyter`: Jupyter Lab for interactive analysis (port 8888)

### Step 4: Initialize Neo4j

```bash
# Initialize Neo4j database with constraints and indexes
docker exec intersymbolic-pipeline python scripts/init_neo4j.py

# Load SHACL shapes for NFCRM-1:2025 and ISO/IEC 27005
docker exec intersymbolic-pipeline python ontology/load_shacl_shapes.py

# Verify Neo4j is ready
curl http://localhost:7474
```

### Step 5: Load the Dataset

```bash
# Load CSE-CIC-IDS2018 data into Neo4j ARG
docker exec intersymbolic-pipeline python pipeline/loaders/cse_cic_ids2018_loader.py \
    --data-dir /app/data/raw/CSE-CIC-IDS2018 \
    --batch-size 10000 \
    --validate
```

**Parameters:**
- `--data-dir`: Path to dataset files
- `--batch-size`: Number of flows to process per batch (default: 10000)
- `--validate`: Enable SHACL validation (default: False)

### Step 6: Extract Features

```bash
# Extract features from loaded data
docker exec intersymbolic-pipeline python pipeline/features/feature_extraction.py \
    --output-dir /app/outputs/features \
    --extract-all
```

**Feature Types:**
- Flow features (84+ raw features from CSE-CIC-IDS2018)
- Behavioral features (statistical, entropy, ratios)
- Graph features (centrality, structural, path-based)
- Temporal aggregations (14 time windows)

### Step 7: Train Models

```bash
# Train all models
docker exec intersymbolic-pipeline python scripts/train_models.py \
    --model all \
    --output-dir /app/models/trained \
    --epochs 50

# Train individual models
docker exec intersymbolic-pipeline python scripts/train_models.py --model isolation-forest
docker exec intersymbolic-pipeline python scripts/train_models.py --model autoencoder
```

**Models:**
- Isolation Forest (anomaly detection)
- Autoencoder (anomaly detection)
- Ensemble (combined)
- Graph Behavioral Scoring (PageRank, centrality)
- Probabilistic Risk Indicators (Bayesian, Monte Carlo)

**Hyperparameters:**
- `--contamination`: Expected contamination for Isolation Forest (default: 0.1)
- `--encoding-dim`: Encoding dimension for Autoencoder (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--test-size`: Test set proportion (default: 0.2)
- `--seed`: Random seed (default: 42)

---

## Running Experiments

### Comparative Experiments

Run all three approaches (Neuro-Symbolic, Pure-ML, Pure-Rule):

```bash
docker exec intersymbolic-pipeline python experiments/comparative_experiments.py \
    --data-dir /app/data/processed \
    --output-dir /app/outputs/experiments \
    --run-all
```

**Outputs:**
- `results.json`: All technical and GRC metrics
- `significance_tests.json`: Statistical analysis results
- `baselines_comparison.csv`: Comparison table

### Specific Experiments

Run individual baselines:

```bash
# Pure-ML baseline
docker exec intersymbolic-pipeline python baseline/ml_only/run_baseline.py

# Pure-Rule baseline
docker exec intersymbolic-pipeline python baseline/pure_rule/run_baseline.py

# Neuro-Symbolic approach (full pipeline)
docker exec intersymbolic-pipeline python pipeline/orchestrator.py \
    --mode batch \
    --input /app/data/processed \
    --output /app/outputs/results
```

### Evaluation Metrics

**Technical Metrics:**
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, Specificity, FPR, FNR
- Statistical significance (p-values, effect sizes)

**GRC Metrics:**
- NFCRM coverage (percentage of controls mapped)
- ISO/IEC 27005 coverage
- Event-to-Risk traceability
- Risk-to-Control traceability
- Audit log completeness

### Interactive Analysis with Jupyter

```bash
# Start Jupyter Lab (already running on port 8888)
# Access at: http://localhost:8888

# Or restart if needed
docker-compose restart jupyter
```

**Notebooks:**
- `notebooks/exploratory_data_analysis.ipynb`
- `notebooks/model_evaluation.ipynb`
- `notebooks/results_visualization.ipynb`

---

## Actual Results

The following results are sourced directly from the JSON files in `results/`. All numbers are reproducible by running the pipeline scripts.

### Technical Performance

| Approach | Accuracy | F1-Macro | Source |
|----------|----------|----------|--------|
| Tri-stage (full) | 87.98% | 0.888 | `results/ablation_study_v2.json` |
| RF Baseline (pure-ML) | 87.70% | 0.896 | `results/ablation_study_v2.json` |
| XGBoost Baseline | 95.90% | 0.692 | `results/xgb_baseline.json` |
| Rule-only Baseline | 36.9% | 0.093 | `results/rule_baseline.json` |
| SLM Baseline | 0% | 0% | `results/cicids_slm.json` (design error — tabular features) |

Note: The tri-stage framework increases accuracy by +0.28 percentage points over RF baseline, but F1-macro degrades (0.888 vs 0.896, −0.008). All post-inference override rules were disabled by calibration because each one hurt F1-macro.

### GRC Outcomes

| Metric | Tri-stage | Pure-ML | Pure-Rule | Source |
|--------|-----------|---------|-----------|--------|
| NFCRM-1:2025 Coverage | 100% | 0% | 100% | `results/grc_metrics.json` |
| ISO/IEC 27005 Coverage | 0% | 0% | 0% | `results/grc_metrics.json` |
| Audit Artifact Generation | Yes | No | Partial | `results/grc_metrics.json` |

Note: NFCRM-1:2025 coverage is 100% by design (the framework was built to cover all NFCRM clauses). ISO/IEC 27005 coverage is 0% — the framework was not designed to map to ISO 27005.

### Statistical Significance

- Tri-stage vs. RF Baseline: p = 0.779, Cohen's d = −0.009 (negligible, non-significant)
- Source: `results/statistical_tests_v2.json` (bootstrap B=1,000, Bonferroni α=0.0056)
- All pairwise comparisons are non-significant after Bonferroni correction.

### Performance Benchmarks

- **Latency**: 0.34 ms average per event (source: `results/e2e_pipeline_benchmark.json`)
- **Throughput**: 2,944 events/second (source: `results/e2e_pipeline_benchmark.json`)
- **Test set size**: N=2,122 samples (0.013% of full CIC-IDS2018 dataset)

---

## Troubleshooting

### Docker Issues

**Problem**: Docker containers won't start
```bash
# Check Docker status
sudo systemctl status docker

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

**Problem**: Neo4j health check failing
```bash
# Check Neo4j logs
docker-compose logs neo4j

# Restart Neo4j
docker-compose restart neo4j

# Wait for Neo4j to be ready
docker exec intersymbolic-pipeline python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://neo4j:7687', auth=('neo4j', 'password'))
driver.verify_connectivity()
print('Neo4j is ready!')
"
```

### Dataset Issues

**Problem**: Dataset download fails
```bash
# Use a download manager
wget -c [DATASET_URL]

# Or use aria2c for resumable downloads
aria2c -x 16 -s 16 [DATASET_URL]
```

**Problem**: CSV files too large for memory
```bash
# Process in smaller batches
docker exec intersymbolic-pipeline python pipeline/loaders/cse_cic_ids2018_loader.py \
    --data-dir /app/data/raw/CSE-CIC-IDS2018 \
    --batch-size 5000 \
    --limit 100000
```

### Training Issues

**Problem**: Out of memory during training
```bash
# Reduce batch size or dataset size
docker exec intersymbolic-pipeline python scripts/train_models.py \
    --model autoencoder \
    --encoding-dim 16 \
    --epochs 25

# Or use a subset of the data
docker exec intersymbolic-pipeline python scripts/train_models.py \
    --data-dir /app/data/processed \
    --sample-size 50000
```

**Problem**: Training takes too long
```bash
# Use GPU acceleration (if available)
# Edit Dockerfile to install CUDA and cuDNN
# Then rebuild with GPU support

# Or use pre-trained models (if available)
# Skip training and load pre-trained weights
```

### Neo4j Connection Issues

**Problem**: Cannot connect to Neo4j
```bash
# Check if Neo4j is running
docker-compose ps neo4j

# Check Neo4j logs
docker-compose logs neo4j | tail -50

# Test connection manually
docker exec intersymbolic-pipeline cypher-shell -u neo4j -p password
```

**Problem**: SHACL validation errors
```bash
# Check SHACL shapes are loaded
docker exec intersymbolic-pipeline cypher-shell -u neo4j -p password "
CALL n10s.graphconfig.init()
"

# Reload SHACL shapes
docker exec intersymbolic-pipeline python ontology/load_shacl_shapes.py --force
```

---

## Citation and License

### Citation

If you use this code or results in your research, please cite:

```bibtex
@thesis{intersymbolic-grc-2026,
  title={Neuro-Symbolic AI for Governance, Risk, and Compliance: A Standards-Aligned Approach},
  author={[Your Name]},
  year={2026},
  school={[Your Institution]},
  url={https://github.com/MOo207/INTERSYMBOLIC-GRC}
}
```

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Dataset License

The CSE-CIC-IDS2018 dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
- https://creativecommons.org/licenses/by/4.0/

### Third-Party Licenses

This project uses the following third-party software with their respective licenses:
- Neo4j: GPL v3 (Community Edition)
- scikit-learn: BSD 3-Clause
- TensorFlow: Apache 2.0
- PyTorch: BSD 3-Clause

---

## Contact and Support

For questions, issues, or contributions:
- GitHub Issues: https://github.com/MOo207/INTERSYMBOLIC-GRC/issues
- Email: [Your Email]

---

## DOI and Archival

This reproducibility package is archived and assigned a DOI:
- DOI: [To be assigned]
- Zenodo: [To be created]
