# Feature Extraction Pipeline

## Overview

The Feature Extraction Pipeline provides comprehensive feature extraction for the CSE-CIC-IDS2018 dataset and Asset Relationship Graph (ARG). It extracts 84+ raw flow features, derived behavioral features, graph-based features, and temporal aggregations for ML model training and inference.

## Architecture

```
Raw Flow Data (CSE-CIC-IDS2018)
         ↓
FlowFeatureExtractor (84+ raw features)
         ↓
BehavioralFeatureExtractor (statistical, entropy, ratios, time)
         ↓
GraphFeatureExtractor (centrality, structural, path, temporal)
         ↓
TemporalAggregator (1s to 1mo windows)
         ↓
Feature Vector (ML-ready)
```

## Components

### 1. FlowFeatureExtractor

Extracts 84+ raw flow features from CSE-CIC-IDS2018 dataset.

**Feature Categories:**
- **Basic Flow Features**: flow_id, timestamp, flow_duration
- **Directional Features**: total_fwd_packets, total_backward_packets, total_length_of_fwd_packet, total_length_of_bwd_packet
- **Packet Length Statistics**: fwd_packet_length_max/min/mean/std, bwd_packet_length_max/min/mean/std
- **Flow-Level Statistics**: flow_bytes_s, flow_packets_s, flow_iat_mean/std/max/min
- **Directional IAT**: fwd_iat_total/mean/std/max/min, bwd_iat_total/mean/std/max/min
- **TCP Flags**: fwd_psh_flags, bwd_psh_flags, fwd_urg_flags, bwd_urg_flags
- **Header Features**: fwd_header_length, bwd_header_length, fwd_packets_s, bwd_packets_s
- **Packet Length Features**: min_packet_length, max_packet_length, packet_length_mean/std/variance
- **Flag Counts**: fin_flag_count, syn_flag_count, rst_flag_count, psh_flag_count, ack_flag_count, urg_flag_count, cwe_flag_count, ece_flag_count
- **Ratio Features**: down_up_ratio, average_packet_size
- **Segment Features**: avg_fwd_segment_size, avg_bwd_segment_size
- **Bulk Features**: fwd_avg_bytes_bulk, fwd_avg_packets_bulk, fwd_avg_bulk_rate, bwd_avg_bytes_bulk, bwd_avg_packets_bulk, bwd_avg_bulk_rate
- **Subflow Features**: subflow_fwd_packets, subflow_fwd_bytes, subflow_bwd_packets, subflow_bwd_bytes
- **Window Features**: init_win_bytes_fwd, init_win_bytes_bwd, act_data_pkt_fwd, act_data_pkt_bwd, min_seg_size_fwd
- **Active/Idle Features**: active_mean/std/max/min, idle_mean/std/max/min

**Total Features:** 84+ raw flow features

### 2. BehavioralFeatureExtractor

Extracts derived and behavioral features from flow data.

**Feature Categories:**
- **Statistical Features**: count, sum, mean, std, min, max (with optional grouping)
- **Entropy Features**: packet_size_entropy, flow_iat_entropy, fwd_packet_size_entropy, bwd_packet_size_entropy
- **Ratio Features**: fwd_bwd_packet_ratio, fwd_bwd_byte_ratio, fwd_bwd_pkt_size_ratio, fwd_bwd_iat_ratio
- **Time Features**: hour, day_of_week, is_weekend, is_business_hour, is_night

**Total Features:** 10-20+ behavioral features

### 3. GraphFeatureExtractor

Extracts graph-based features from Asset Relationship Graph (ARG).

**Feature Categories:**
- **Centrality Features**: pagerank, degree, betweenness, closeness
- **Structural Features**: clustering_coefficient, component_id, community_id
- **Path Features**: shortest_path_length, diameter
- **Temporal Features**: temporal_degree, temporal_pagerank

**Requirements:**
- Neo4j database with ARG
- Neo4j Graph Data Science (GDS) plugin installed

**Total Features:** 8-10+ graph features

### 4. TemporalAggregator

Aggregates flow features over temporal windows.

**Supported Time Windows:**
- Short-term: 1s, 10s, 30s
- Medium-term: 1m, 5m, 15m, 30m, 1h, 6h, 12h
- Long-term: 1d, 1w, 2w, 1mo

**Aggregation Functions:**
- count, sum, mean, std, min, max, entropy, rate

**Usage:**
```python
aggregator = TemporalAggregator(timestamp_col='timestamp')
agg_df = aggregator.aggregate(df, window='1h', functions=['mean', 'count'])
```

## Installation

### Requirements

```bash
pip install pandas numpy scikit-learn networkx neo4j
```

### Neo4j Setup (for Graph Features)

1. Install Neo4j Database
2. Install Neo4j GDS Plugin
3. Create graph projection:

```cypher
CALL gds.graph.project('myGraph', 'Asset', 'CONNECTED')
```

## Usage

### Basic Flow Feature Extraction

```python
from pipeline.features import FlowFeatureExtractor

# Initialize extractor
extractor = FlowFeatureExtractor('/path/to/cse-cic-ids2018')

# Load data (example)
import pandas as pd
df = pd.read_csv('flow_records.csv')

# Extract raw features
features_df = extractor.extract_raw_features(df)

print(f"Extracted {len(features_df.columns)} features")
```

### Behavioral Feature Extraction

```python
from pipeline.features import BehavioralFeatureExtractor

# Initialize extractor
extractor = BehavioralFeatureExtractor()

# Extract statistical features
features_df = extractor.extract_statistical_features(df)

# Extract entropy features
features_df = extractor.extract_entropy_features(features_df)

# Extract ratio features
features_df = extractor.extract_ratio_features(features_df)

# Extract time features
features_df = extractor.extract_time_features(features_df)
```

### Graph Feature Extraction

```python
from pipeline.features import GraphFeatureExtractor
from neo4j import GraphDatabase

# Connect to Neo4j
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))

# Initialize extractor
extractor = GraphFeatureExtractor(driver)

# Extract centrality features
centrality_df = extractor.extract_centrality_features()

# Extract structural features
structural_df = extractor.extract_structural_features()

# Extract path features
path_df = extractor.extract_path_features()

# Extract temporal features
temporal_df = extractor.extract_temporal_features(time_window='1h')
```

### Temporal Aggregation

```python
from pipeline.features import TemporalAggregator

# Initialize aggregator
aggregator = TemporalAggregator(timestamp_col='timestamp')

# Aggregate over 1-hour window
agg_df = aggregator.aggregate(
    df,
    window='1h',
    functions=['mean', 'count', 'sum', 'std']
)

# Aggregate over multiple windows
for window in ['1h', '1d', '1w']:
    agg_df = aggregator.aggregate(df, window=window)
    agg_df.to_csv(f'features_aggregated_{window}.csv')
```

### Full Pipeline

```python
from pipeline.features import FeatureExtractionPipeline

# Initialize pipeline
pipeline = FeatureExtractionPipeline(
    dataset_path='/path/to/cse-cic-ids2018',
    neo4j_driver=driver,  # Optional
    timestamp_col='timestamp'
)

# Extract all features
features_df = pipeline.extract_all_features(
    df,
    extract_graph_features=True,  # Extract graph features if Neo4j available
    time_windows=['1h', '1d', '1w']  # Temporal aggregation windows
)

print(f"Total features: {len(features_df.columns)}")
print(f"Feature shape: {features_df.shape}")

# Save features
features_df.to_csv('extracted_features.csv', index=False)
```

## Feature Catalog

### Raw Flow Features (84+)

| Category | Feature Count | Example Features |
|----------|---------------|------------------|
| Basic Flow | 3 | flow_id, timestamp, flow_duration |
| Directional | 4 | total_fwd_packets, total_backward_packets, total_length_of_fwd_packet, total_length_of_bwd_packet |
| Packet Length Stats | 8 | fwd_packet_length_max/min/mean/std, bwd_packet_length_max/min/mean/std |
| Flow-Level Stats | 6 | flow_bytes_s, flow_packets_s, flow_iat_mean/std/max/min |
| Directional IAT | 10 | fwd_iat_total/mean/std/max/min, bwd_iat_total/mean/std/max/min |
| TCP Flags | 4 | fwd_psh_flags, bwd_psh_flags, fwd_urg_flags, bwd_urg_flags |
| Header Features | 4 | fwd_header_length, bwd_header_length, fwd_packets_s, bwd_packets_s |
| Packet Length Features | 5 | min_packet_length, max_packet_length, packet_length_mean/std/variance |
| Flag Counts | 8 | fin_flag_count, syn_flag_count, rst_flag_count, psh_flag_count, ack_flag_count, urg_flag_count, cwe_flag_count, ece_flag_count |
| Ratio Features | 2 | down_up_ratio, average_packet_size |
| Segment Features | 2 | avg_fwd_segment_size, avg_bwd_segment_size |
| Bulk Features | 6 | fwd_avg_bytes_bulk, fwd_avg_packets_bulk, fwd_avg_bulk_rate, bwd_avg_bytes_bulk, bwd_avg_packets_bulk, bwd_avg_bulk_rate |
| Subflow Features | 4 | subflow_fwd_packets, subflow_fwd_bytes, subflow_bwd_packets, subflow_bwd_bytes |
| Window Features | 5 | init_win_bytes_fwd, init_win_bytes_bwd, act_data_pkt_fwd, act_data_pkt_bwd, min_seg_size_fwd |
| Active/Idle Features | 8 | active_mean/std/max/min, idle_mean/std/max/min |

**Total Raw Flow Features:** 84+

### Behavioral Features (10-20+)

| Category | Feature Count | Example Features |
|----------|---------------|------------------|
| Statistical | 6-12 | count, sum, mean, std, min, max (per numeric column) |
| Entropy | 4 | packet_size_entropy, flow_iat_entropy, fwd_packet_size_entropy, bwd_packet_size_entropy |
| Ratio | 4 | fwd_bwd_packet_ratio, fwd_bwd_byte_ratio, fwd_bwd_pkt_size_ratio, fwd_bwd_iat_ratio |
| Time | 5 | hour, day_of_week, is_weekend, is_business_hour, is_night |

**Total Behavioral Features:** 10-20+ (depends on grouping)

### Graph Features (8-10+)

| Category | Feature Count | Example Features |
|----------|---------------|------------------|
| Centrality | 4 | pagerank, degree, betweenness, closeness |
| Structural | 3 | clustering_coefficient, component_id, community_id |
| Path | 2 | shortest_path_length, diameter |
| Temporal | 2 | temporal_degree, temporal_pagerank |

**Total Graph Features:** 8-10+ (requires Neo4j GDS)

## Performance Considerations

### Flow Feature Extraction
- **Speed:** ~10,000 records/second
- **Memory:** ~100 MB for 100,000 records
- **Optimization:** Vectorized operations with pandas

### Behavioral Feature Extraction
- **Speed:** ~5,000 records/second
- **Memory:** ~50 MB for 100,000 records
- **Optimization:** Efficient aggregation with groupby

### Graph Feature Extraction
- **Speed:** Depends on graph size (100-1000 ms per query)
- **Memory:** Neo4j GDS manages memory
- **Optimization:** Use graph projections, limit results

### Temporal Aggregation
- **Speed:** ~1,000 records/second per window
- **Memory:** Depends on aggregation functions
- **Optimization:** Batch processing for multiple windows

## Testing

Run the test suite:

```bash
pytest tests/test_feature_extraction.py -v
```

Test coverage:
- FlowFeatureExtractor: 15+ tests
- BehavioralFeatureExtractor: 10+ tests
- GraphFeatureExtractor: 5+ tests
- TemporalAggregator: 8+ tests
- FeatureExtractionPipeline: 5+ tests
- Feature Count: 2+ tests

## Integration with ML Models

### Anomaly Detection

```python
from models.anomaly_detection import IsolationForestModel

# Extract features
pipeline = FeatureExtractionPipeline(dataset_path='...')
features_df = pipeline.extract_all_features(df)

# Select numeric features
numeric_cols = features_df.select_dtypes(include=[np.number]).columns
X = features_df[numeric_cols].fillna(0)

# Train model
model = IsolationForestModel(contamination=0.1)
model.train(X)

# Predict anomalies
anomalies = model.predict(X)
```

### Graph Behavioral Scoring

```python
from models.graph_behavioral_scoring import GraphBehavioralScoring

# Extract graph features
graph_extractor = GraphFeatureExtractor(driver)
centrality_df = graph_extractor.extract_centrality_features()

# Use centrality scores for behavioral scoring
scorer = GraphBehavioralScoring(driver)
pagerank_scores = scorer.compute_pagerank()
```

### Probabilistic Risk Indicators

```python
from models.probabilistic_risk_indicators import ProbabilisticRiskModel

# Extract features with temporal aggregation
aggregator = TemporalAggregator()
agg_df = aggregator.aggregate(df, window='1h', functions=['mean', 'std'])

# Use aggregated features for risk modeling
risk_model = ProbabilisticRiskModel()
risk_scores = risk_model.compute_risk(agg_df)
```

## Troubleshooting

### Missing Features

**Problem:** Some features are missing after extraction.

**Solution:**
- Check that required columns exist in input DataFrame
- Verify column names match expected format (lowercase, underscores)
- Check for NaN values in required columns

### Neo4j Connection Errors

**Problem:** Graph feature extraction fails with connection error.

**Solution:**
- Verify Neo4j is running (`cypher-shell` or Neo4j Browser)
- Check connection string and credentials
- Ensure Neo4j GDS plugin is installed
- Verify graph projection exists (`CALL gds.graph.list()`)

### Out of Memory

**Problem:** Feature extraction runs out of memory for large datasets.

**Solution:**
- Process data in batches (chunking)
- Reduce number of temporal windows
- Disable graph features if not needed
- Increase available memory or use a machine with more RAM

### Slow Performance

**Problem:** Feature extraction is slow.

**Solution:**
- Use vectorized operations (pandas)
- Enable caching for repeated queries
- Reduce temporal window count
- Parallelize processing for multiple windows
- Use Neo4j GDS with optimized graph projections

## Future Enhancements

1. **Streaming Feature Extraction:** Real-time feature extraction for live traffic
2. **Feature Selection:** Automatic feature importance analysis
3. **Feature Engineering:** Automatic creation of new features
4. **Distributed Processing:** Spark/Dask for large-scale datasets
5. **GPU Acceleration:** GPU-accelerated graph algorithms
6. **Incremental Updates:** Update features without recomputing everything
7. **Feature Caching:** Cache extracted features for faster repeated access

## References

- **CSE-CIC-IDS2018 Dataset:** Sharafaldin et al. (2018)
- **Neo4j GDS:** https://neo4j.com/docs/graph-data-science/current/
- **Feature Engineering for IDS:** Literature review in research docs

## License

This module is part of INTERSYMBOLIC-GRC, licensed under [LICENSE](../../LICENSE).
