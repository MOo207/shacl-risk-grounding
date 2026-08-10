# shacl-risk-grounding

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21869883.svg)](https://doi.org/10.5281/zenodo.21869883)

Reproducibility package for the manuscript:

> **Grounding Machine Learning and LLM-Based Risk Inference in a Shared SHACL Knowledge Layer: An NFCRM-1:2025 Case Study**
> Mohammed Ismail Alamawy, Fayçal Hamdi, Hamad Binsalleeh

This repository contains the implementation and experimental artefacts used to evaluate the proposed knowledge-grounding architecture: the OWL/RDF cybersecurity knowledge model, SHACL constraint profiles, preprocessing and evaluation scripts, machine-learning and LLM workflows, saved validation and evaluation results, and figure-generation code.

## Contents

| Directory | Description |
|-----------|-------------|
| `ontology/` | OWL/RDF cybersecurity knowledge model (Turtle) and SHACL constraint profiles (17 node shapes) |
| `rules/` | Symbolic rule definitions |
| `pipeline/` | Core pipeline package imported by the scripts (risk scoring, NFCRM mapping, feature extraction, inference) |
| `scripts/` | Preprocessing, experiment, and analysis scripts (ablation studies, LLM workflows, statistical tests, SHAP/XAI analysis, figure generation) |
| `results/` | Saved validation and evaluation outputs (JSON): knowledge-layer validation, GRC artefact conformance, schema comparison, CIC-IDS2018 and NSL-KDD evaluations, ablation and control experiments |
| `data/processed/` | Processed datasets used by the experiments |
| `data/external/` | Enrichment sources: CISA KEV snapshot, NVD enrichment, MITRE ATT&CK (enterprise), CMDB assets, NFCRM clause mapping |
| `data/raw/NSL-KDD/` | NSL-KDD train/test files |
| `requirements.txt` | Python dependencies |
| `REPRODUCIBILITY.md` | Notes on reproducing the experiments |

## Reproducing the reported results

Every number reported in the manuscript can be verified directly from the saved outputs in `results/` without rerunning the pipeline. To rerun experiments:

```bash
pip install -r requirements.txt
python scripts/<experiment_script>.py
```

The raw CSE-CIC-IDS2018 dataset (~8 GB) is not redistributed; obtain it from the [official source](https://www.unb.ca/cic/datasets/ids-2018.html). The processed subset used in the experiments is included. The NFCRM-1:2025 framework document is available from the [NCA regulatory documents page](https://nca.gov.sa/en/regulatory-documents/).

## LLM backend

All LLM experiments call Claude models (Haiku 4.5, Sonnet 4.6) through the Claude CLI subprocess wrapper in `scripts/claude_cli_client.py`. No API keys are stored in this repository.

## License

MIT — see [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).

- Concept DOI (all versions, recommended): [10.5281/zenodo.21869883](https://doi.org/10.5281/zenodo.21869883)
- Version DOI (v1.0.0): [10.5281/zenodo.21869884](https://doi.org/10.5281/zenodo.21869884)
