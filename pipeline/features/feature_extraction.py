"""
Feature Extraction Pipeline for INTERSYMBOLIC-GRC

Implements comprehensive feature extraction for CSE-CIC-IDS2018 dataset:
1. Raw Flow Features (84+ features)
2. Derived & Behavioral Features (statistical aggregations, entropy, ratios)
3. Graph-Based Features (centrality, structural, path-based, temporal)
4. Temporal Aggregation (1s to 1mo windows)

Extracted features are designed for:
- Anomaly detection (Isolation Forest, Autoencoders)
- Graph-based behavioral scoring
- Probabilistic risk indicators
- Pre-inference symbolic rules

Reference:
CSE-CIC-IDS2018 Dataset Features
- Timestamp, Flow Duration, Total Fwd/Bwd Packets, Total Length of Fwd/Bwd Packets
- Fwd/Bwd Packet Length Mean/Max/Min/Std, Flow Bytes/s, Flow Packets/s
- Flow IAT Mean/Std/Max/Min, Fwd/Bwd IAT Total/Mean/Std/Max/Min
- Fwd/Bwd PSH Flags, Fwd/Bwd URG Flags, Fwd/Bwd Header Length
- Fwd/Bwd Packets/s, Min Packet Length, Max Packet Length, Packet Length Mean/Var/Std
- Packet Length Variance, FIN Flag Count, SYN Flag Count, RST Flag Count
- PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count
- ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size
- Avg Bwd Segment Size, Fwd/Bwd Header Length, Fwd Avg Bytes/Bulk
- Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk
- Bwd Avg Packets/Bulk, Bwd Avg Bulk Rate, Subflow Fwd Packets
- Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes
- Init Win Bytes Fwd, Init Win Bytes Bwd, Act Data Pkt Fwd
- Act Data Pkt Bwd, Min Seg Size Fwd, Active Mean/Std/Max/Min
- Idle Mean/Std/Max/Min, Label
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from collections import defaultdict
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowFeatureExtractor:
    """
    Extracts 84+ raw flow features from CSE-CIC-IDS2018 dataset.

    Features are extracted in the following categories:
    - Basic Flow Features: timestamp, flow duration, packet/byte counts
    - Directional Features: forward/backward packet/byte statistics
    - Inter-Arrival Time Features: IAT statistics
    - Flag Features: TCP flags (FIN, SYN, RST, PSH, ACK, URG, CWE, ECE)
    - Header Features: header lengths
    - Statistical Features: means, standard deviations, variances
    - Flow-Level Features: bytes/s, packets/s, average packet size
    - Segment Features: segment sizes, bulk statistics
    - Subflow Features: subflow packet/byte counts
    - Window Features: initial window sizes, active/idle times
    - Aggregate Features: min/max packet sizes, ratios
    """

    # CSE-CIC-IDS2018 feature columns (84 features)
    RAW_FEATURE_COLUMNS = [
        # Basic flow features
        'flow_id', 'timestamp', 'flow_duration',
        'total_fwd_packets', 'total_backward_packets',
        'total_length_of_fwd_packet', 'total_length_of_bwd_packet',

        # Directional packet length statistics
        'fwd_packet_length_max', 'fwd_packet_length_min', 'fwd_packet_length_mean',
        'fwd_packet_length_std', 'bwd_packet_length_max', 'bwd_packet_length_min',
        'bwd_packet_length_mean', 'bwd_packet_length_std',

        # Flow-level statistics
        'flow_bytes_s', 'flow_packets_s', 'flow_iat_mean', 'flow_iat_std',
        'flow_iat_max', 'flow_iat_min',

        # Directional IAT statistics
        'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max',
        'fwd_iat_min', 'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std',
        'bwd_iat_max', 'bwd_iat_min',

        # TCP flags
        'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags',

        # Header features
        'fwd_header_length', 'bwd_header_length', 'fwd_packets_s', 'bwd_packets_s',

        # Packet length features
        'min_packet_length', 'max_packet_length', 'packet_length_mean',
        'packet_length_std', 'packet_length_variance',

        # Flag counts
        'fin_flag_count', 'syn_flag_count', 'rst_flag_count', 'psh_flag_count',
        'ack_flag_count', 'urg_flag_count', 'cwe_flag_count', 'ece_flag_count',

        # Ratio features
        'down_up_ratio', 'average_packet_size',

        # Segment features
        'avg_fwd_segment_size', 'avg_bwd_segment_size',

        # Bulk features
        'fwd_avg_bytes_bulk', 'fwd_avg_packets_bulk', 'fwd_avg_bulk_rate',
        'bwd_avg_bytes_bulk', 'bwd_avg_packets_bulk', 'bwd_avg_bulk_rate',

        # Subflow features
        'subflow_fwd_packets', 'subflow_fwd_bytes', 'subflow_bwd_packets',
        'subflow_bwd_bytes',

        # Window features
        'init_win_bytes_fwd', 'init_win_bytes_bwd', 'act_data_pkt_fwd',
        'act_data_pkt_bwd', 'min_seg_size_fwd',

        # Active/Idle features
        'active_mean', 'active_std', 'active_max', 'active_min',
        'idle_mean', 'idle_std', 'idle_max', 'idle_min',

        # Label
        'label'
    ]

    def __init__(self, dataset_path: str):
        """
        Initialize flow feature extractor.

        Args:
            dataset_path: Path to CSE-CIC-IDS2018 dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.features: List[Dict] = []

    def extract_raw_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract 84+ raw flow features from dataframe.

        Args:
            df: DataFrame with raw flow records

        Returns:
            DataFrame with extracted features
        """
        logger.info(f"Extracting raw flow features from {len(df)} records")

        # Create feature DataFrame
        features_df = df.copy()

        # Ensure required columns exist
        required_cols = [
            'timestamp', 'flow_duration', 'total_fwd_packets', 'total_backward_packets',
            'total_length_of_fwd_packet', 'total_length_of_bwd_packet'
        ]

        for col in required_cols:
            if col not in features_df.columns:
                logger.warning(f"Missing required column: {col}")
                features_df[col] = 0

        # Fill missing values
        features_df = features_df.fillna(0)

        # Normalize column names (lowercase, spaces to underscores)
        features_df.columns = [
            col.strip().lower().replace(' ', '_')
            for col in features_df.columns
        ]

        # Parse timestamp
        if 'timestamp' in features_df.columns:
            features_df['timestamp'] = pd.to_datetime(
                features_df['timestamp'],
                errors='coerce'
            )

        # Extract all 84 features
        features_df = self._extract_basic_flow_features(features_df)
        features_df = self._extract_directional_features(features_df)
        features_df = self._extract_iat_features(features_df)
        features_df = self._extract_flag_features(features_df)
        features_df = self._extract_header_features(features_df)
        features_df = self._extract_packet_length_features(features_df)
        features_df = self._extract_ratio_features(features_df)
        features_df = self._extract_segment_features(features_df)
        features_df = self._extract_bulk_features(features_df)
        features_df = self._extract_subflow_features(features_df)
        features_df = self._extract_window_features(features_df)
        features_df = self._extract_active_idle_features(features_df)

        logger.info(f"Extracted {len(features_df.columns)} features")
        return features_df

    def _extract_basic_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic flow features."""
        # Ensure basic features exist
        if 'flow_id' not in df.columns:
            df['flow_id'] = df.index.astype(str)

        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime('now')

        if 'flow_duration' not in df.columns:
            df['flow_duration'] = 0.0

        return df

    def _extract_directional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract forward/backward directional features."""
        # Ensure packet counts exist
        if 'total_fwd_packets' not in df.columns:
            df['total_fwd_packets'] = df.get('fwd_packets', 0)

        if 'total_backward_packets' not in df.columns:
            df['total_backward_packets'] = df.get('bwd_packets', 0)

        # Ensure byte counts exist
        if 'total_length_of_fwd_packet' not in df.columns:
            df['total_length_of_fwd_packet'] = df.get('fwd_bytes', 0)

        if 'total_length_of_bwd_packet' not in df.columns:
            df['total_length_of_bwd_packet'] = df.get('bwd_bytes', 0)

        return df

    def _extract_iat_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract Inter-Arrival Time (IAT) features."""
        # Flow IAT features
        iat_cols = ['flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min']
        for col in iat_cols:
            if col not in df.columns:
                df[col] = 0.0

        # Forward IAT features
        fwd_iat_cols = [
            'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std',
            'fwd_iat_max', 'fwd_iat_min'
        ]
        for col in fwd_iat_cols:
            if col not in df.columns:
                df[col] = 0.0

        # Backward IAT features
        bwd_iat_cols = [
            'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std',
            'bwd_iat_max', 'bwd_iat_min'
        ]
        for col in bwd_iat_cols:
            if col not in df.columns:
                df[col] = 0.0

        return df

    def _extract_flag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract TCP flag features."""
        # Flag count features
        flag_cols = [
            'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags',
            'fin_flag_count', 'syn_flag_count', 'rst_flag_count', 'psh_flag_count',
            'ack_flag_count', 'urg_flag_count', 'cwe_flag_count', 'ece_flag_count'
        ]
        for col in flag_cols:
            if col not in df.columns:
                df[col] = 0

        return df

    def _extract_header_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract header features."""
        # Header length features
        if 'fwd_header_length' not in df.columns:
            df['fwd_header_length'] = df.get('header_length', 0) / 2

        if 'bwd_header_length' not in df.columns:
            df['bwd_header_length'] = df.get('header_length', 0) / 2

        # Packets per second
        if 'flow_duration' in df.columns:
            df['fwd_packets_s'] = (
                df['total_fwd_packets'] / (df['flow_duration'] + 1)
            )
            df['bwd_packets_s'] = (
                df['total_backward_packets'] / (df['flow_duration'] + 1)
            )

        return df

    def _extract_packet_length_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract packet length features."""
        # Min/max packet length
        if 'min_packet_length' not in df.columns:
            df['min_packet_length'] = df.get('packet_length_min', 0)

        if 'max_packet_length' not in df.columns:
            df['max_packet_length'] = df.get('packet_length_max', 0)

        # Packet length statistics
        if 'packet_length_mean' not in df.columns:
            df['packet_length_mean'] = df.get('avg_packet_size', 0)

        if 'packet_length_std' not in df.columns:
            df['packet_length_std'] = df.get('packet_length_std', 0)

        if 'packet_length_variance' not in df.columns:
            df['packet_length_variance'] = df.get('packet_length_var', 0)

        # Forward packet length statistics
        fwd_pkt_cols = [
            'fwd_packet_length_max', 'fwd_packet_length_min',
            'fwd_packet_length_mean', 'fwd_packet_length_std'
        ]
        for col in fwd_pkt_cols:
            if col not in df.columns:
                df[col] = 0.0

        # Backward packet length statistics
        bwd_pkt_cols = [
            'bwd_packet_length_max', 'bwd_packet_length_min',
            'bwd_packet_length_mean', 'bwd_packet_length_std'
        ]
        for col in bwd_pkt_cols:
            if col not in df.columns:
                df[col] = 0.0

        return df

    def _extract_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract ratio features."""
        # Down/Up ratio
        if 'down_up_ratio' not in df.columns:
            total_fwd = df['total_length_of_fwd_packet'] + 1
            total_bwd = df['total_length_of_bwd_packet'] + 1
            df['down_up_ratio'] = total_bwd / total_fwd

        # Average packet size
        if 'average_packet_size' not in df.columns:
            total_packets = df['total_fwd_packets'] + df['total_backward_packets'] + 1
            total_bytes = (
                df['total_length_of_fwd_packet'] + df['total_length_of_bwd_packet']
            )
            df['average_packet_size'] = total_bytes / total_packets

        # Flow bytes/s and packets/s
        if 'flow_duration' in df.columns:
            if 'flow_bytes_s' not in df.columns:
                total_bytes = (
                    df['total_length_of_fwd_packet'] + df['total_length_of_bwd_packet']
                )
                df['flow_bytes_s'] = total_bytes / (df['flow_duration'] + 1)

            if 'flow_packets_s' not in df.columns:
                total_packets = (
                    df['total_fwd_packets'] + df['total_backward_packets']
                )
                df['flow_packets_s'] = total_packets / (df['flow_duration'] + 1)

        return df

    def _extract_segment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract segment features."""
        # Average segment sizes
        if 'avg_fwd_segment_size' not in df.columns:
            df['avg_fwd_segment_size'] = df.get('fwd_segment_size', 0)

        if 'avg_bwd_segment_size' not in df.columns:
            df['avg_bwd_segment_size'] = df.get('bwd_segment_size', 0)

        return df

    def _extract_bulk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract bulk transfer features."""
        # Bulk features
        bulk_cols = [
            'fwd_avg_bytes_bulk', 'fwd_avg_packets_bulk', 'fwd_avg_bulk_rate',
            'bwd_avg_bytes_bulk', 'bwd_avg_packets_bulk', 'bwd_avg_bulk_rate'
        ]
        for col in bulk_cols:
            if col not in df.columns:
                df[col] = 0.0

        return df

    def _extract_subflow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract subflow features."""
        # Subflow features
        subflow_cols = [
            'subflow_fwd_packets', 'subflow_fwd_bytes',
            'subflow_bwd_packets', 'subflow_bwd_bytes'
        ]
        for col in subflow_cols:
            if col not in df.columns:
                df[col] = 0.0

        return df

    def _extract_window_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract window features."""
        # Initial window sizes
        if 'init_win_bytes_fwd' not in df.columns:
            df['init_win_bytes_fwd'] = df.get('window_size_fwd', 0)

        if 'init_win_bytes_bwd' not in df.columns:
            df['init_win_bytes_bwd'] = df.get('window_size_bwd', 0)

        # Active data packets
        if 'act_data_pkt_fwd' not in df.columns:
            df['act_data_pkt_fwd'] = df.get('active_data_pkt_fwd', 0)

        if 'act_data_pkt_bwd' not in df.columns:
            df['act_data_pkt_bwd'] = df.get('active_data_pkt_bwd', 0)

        # Minimum segment size
        if 'min_seg_size_fwd' not in df.columns:
            df['min_seg_size_fwd'] = df.get('min_segment_size', 0)

        return df

    def _extract_active_idle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract active/idle time features."""
        # Active time features
        active_cols = ['active_mean', 'active_std', 'active_max', 'active_min']
        for col in active_cols:
            if col not in df.columns:
                df[col] = 0.0

        # Idle time features
        idle_cols = ['idle_mean', 'idle_std', 'idle_max', 'idle_min']
        for col in idle_cols:
            if col not in df.columns:
                df[col] = 0.0

        return df


class BehavioralFeatureExtractor:
    """
    Extracts derived and behavioral features from flow data.

    Features include:
    - Statistical aggregations (count, sum, mean, std, min, max)
    - Entropy features (packet size entropy, inter-arrival time entropy)
    - Ratio features (fwd/bwd ratios, normalized features)
    - Time-based features (hour of day, day of week, is_weekend)
    """

    def __init__(self):
        """Initialize behavioral feature extractor."""
        pass

    def extract_statistical_features(
        self,
        df: pd.DataFrame,
        group_by: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract statistical aggregation features.

        Args:
            df: DataFrame with flow features
            group_by: Optional list of columns to group by

        Returns:
            DataFrame with statistical features
        """
        logger.info("Extracting statistical features")

        if group_by is None:
            group_by = []

        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not group_by:
            # Compute statistics across all rows
            stats = {
                'count': len(df),
                'mean': df[numeric_cols].mean(),
                'std': df[numeric_cols].std(),
                'min': df[numeric_cols].min(),
                'max': df[numeric_cols].max(),
                'sum': df[numeric_cols].sum()
            }

            # Add to dataframe
            for stat_name, values in stats.items():
                if stat_name == 'count':
                    # count is a scalar for the whole DataFrame
                    for col in numeric_cols:
                        df[f'{col}_{stat_name}'] = values
                else:
                    for col in numeric_cols:
                        df[f'{col}_{stat_name}'] = values[col]

        else:
            # Group and compute statistics
            grouped = df.groupby(group_by)[numeric_cols].agg([
                ('count', 'count'),
                ('mean', 'mean'),
                ('std', 'std'),
                ('min', 'min'),
                ('max', 'max'),
                ('sum', 'sum')
            ])

            # Flatten column names
            grouped.columns = [
                f'{col}_{agg}' if agg else col
                for col, agg in grouped.columns
            ]

            # Merge back to original dataframe
            df = df.merge(grouped, left_on=group_by, right_index=True, how='left')

        logger.info(f"Extracted {len(df.columns) - len(df.select_dtypes(include=[np.number]).columns)} statistical features")
        return df

    def extract_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract entropy features for packet sizes and IATs.

        Args:
            df: DataFrame with flow features

        Returns:
            DataFrame with entropy features
        """
        logger.info("Extracting entropy features")

        # Packet size entropy
        if 'packet_length_mean' in df.columns:
            df['packet_size_entropy'] = self._compute_entropy(
                df['packet_length_mean']
            )

        # Flow IAT entropy
        if 'flow_iat_mean' in df.columns:
            df['flow_iat_entropy'] = self._compute_entropy(df['flow_iat_mean'])

        # Forward packet size entropy
        if 'fwd_packet_length_mean' in df.columns:
            df['fwd_packet_size_entropy'] = self._compute_entropy(
                df['fwd_packet_length_mean']
            )

        # Backward packet size entropy
        if 'bwd_packet_length_mean' in df.columns:
            df['bwd_packet_size_entropy'] = self._compute_entropy(
                df['bwd_packet_length_mean']
            )

        return df

    def _compute_entropy(self, series: pd.Series) -> np.ndarray:
        """
        Compute Shannon entropy for a series.

        Args:
            series: Pandas series of values

        Returns:
            Array of entropy values
        """
        # Discretize values into bins
        binned = pd.cut(series, bins=20, labels=False)

        # Compute probability distribution
        counts = np.bincount(binned[~np.isnan(binned)].astype(int))
        probs = counts / counts.sum()

        # Compute entropy (avoid log(0))
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = -np.sum(probs * np.log2(probs + 1e-10))

        return entropy

    def extract_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract ratio features between directional metrics.

        Args:
            df: DataFrame with flow features

        Returns:
            DataFrame with ratio features
        """
        logger.info("Extracting ratio features")

        # Forward/Backward packet ratio
        if 'total_fwd_packets' in df.columns and 'total_backward_packets' in df.columns:
            df['fwd_bwd_packet_ratio'] = df['total_fwd_packets'] / (
                df['total_backward_packets'] + 1
            )

        # Forward/Backward byte ratio
        if 'total_length_of_fwd_packet' in df.columns and 'total_length_of_bwd_packet' in df.columns:
            df['fwd_bwd_byte_ratio'] = df['total_length_of_fwd_packet'] / (
                df['total_length_of_bwd_packet'] + 1
            )

        # Packet size ratio
        if 'fwd_packet_length_mean' in df.columns and 'bwd_packet_length_mean' in df.columns:
            df['fwd_bwd_pkt_size_ratio'] = df['fwd_packet_length_mean'] / (
                df['bwd_packet_length_mean'] + 1
            )

        # IAT ratio
        if 'fwd_iat_mean' in df.columns and 'bwd_iat_mean' in df.columns:
            df['fwd_bwd_iat_ratio'] = df['fwd_iat_mean'] / (df['bwd_iat_mean'] + 1)

        return df

    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features.

        Args:
            df: DataFrame with flow features

        Returns:
            DataFrame with time features
        """
        logger.info("Extracting time features")

        if 'timestamp' not in df.columns:
            return df

        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hour'] = (
            (df['hour'] >= 9) & (df['hour'] <= 17)
        ).astype(int)
        df['is_night'] = (
            (df['hour'] >= 22) | (df['hour'] <= 6)
        ).astype(int)

        return df


class GraphFeatureExtractor:
    """
    Extracts graph-based features from Asset Relationship Graph (ARG).

    Features include:
    - Centrality features (PageRank, degree, betweenness, closeness)
    - Structural features (clustering coefficient, connected components)
    - Path-based features (shortest paths, diameter)
    - Temporal features (time-evolving graph metrics)
    """

    def __init__(self, neo4j_driver=None):
        """
        Initialize graph feature extractor.

        Args:
            neo4j_driver: Neo4j driver instance
        """
        self.neo4j_driver = neo4j_driver
        self.graph = None

    def extract_centrality_features(
        self,
        node_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract centrality features for assets.

        Args:
            node_ids: List of node IDs (if None, extract for all)

        Returns:
            DataFrame with centrality features
        """
        logger.info("Extracting centrality features")

        if self.neo4j_driver is None:
            raise ValueError("Neo4j driver not configured")

        # Query to compute PageRank
        pagerank_query = """
        CALL gds.pageRank.stream('myGraph')
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).id AS node_id, score AS pagerank
        """

        # Query to compute degree centrality
        degree_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[]-(neighbor)
        WITH n, count(neighbor) AS degree
        RETURN n.id AS node_id, degree
        """

        # Query to compute betweenness centrality
        betweenness_query = """
        CALL gds.betweenness.stream('myGraph')
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).id AS node_id, score AS betweenness
        """

        # Query to compute closeness centrality
        closeness_query = """
        CALL gds.closeness.stream('myGraph')
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).id AS node_id, score AS closeness
        """

        # Execute queries and merge results
        features_df = self._execute_and_merge_queries([
            pagerank_query, degree_query, betweenness_query, closeness_query
        ])

        # Filter by node IDs if provided
        if node_ids is not None:
            features_df = features_df[features_df['node_id'].isin(node_ids)]

        logger.info(f"Extracted centrality features for {len(features_df)} nodes")
        return features_df

    def extract_structural_features(
        self,
        node_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract structural features for assets.

        Args:
            node_ids: List of node IDs (if None, extract for all)

        Returns:
            DataFrame with structural features
        """
        logger.info("Extracting structural features")

        if self.neo4j_driver is None:
            raise ValueError("Neo4j driver not configured")

        # Query to compute clustering coefficient
        clustering_query = """
        MATCH (n)
        CALL gds.localClusteringCoefficient.stream('myGraph', {nodeProjection: 'Asset'})
        YIELD nodeId, coefficient
        RETURN gds.util.asNode(nodeId).id AS node_id, coefficient AS clustering_coefficient
        """

        # Query to compute connected components
        components_query = """
        CALL gds.wcc.stream('myGraph')
        YIELD nodeId, componentId
        RETURN gds.util.asNode(nodeId).id AS node_id, componentId
        """

        # Query to compute community detection
        community_query = """
        CALL gds.louvain.stream('myGraph')
        YIELD nodeId, communityId
        RETURN gds.util.asNode(nodeId).id AS node_id, communityId
        """

        # Execute queries and merge results
        features_df = self._execute_and_merge_queries([
            clustering_query, components_query, community_query
        ])

        # Filter by node IDs if provided
        if node_ids is not None:
            features_df = features_df[features_df['node_id'].isin(node_ids)]

        logger.info(f"Extracted structural features for {len(features_df)} nodes")
        return features_df

    def extract_path_features(
        self,
        node_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract path-based features for assets.

        Args:
            node_ids: List of node IDs (if None, extract for all)

        Returns:
            DataFrame with path features
        """
        logger.info("Extracting path-based features")

        if self.neo4j_driver is None:
            raise ValueError("Neo4j driver not configured")

        # Query to compute shortest path lengths
        shortest_path_query = """
        MATCH (source:Asset), (target:Asset)
        WHERE id(source) < id(target)
        CALL gds.shortestPath.stream('myGraph', {
            sourceNode: source,
            targetNode: target
        })
        YIELD totalCost
        RETURN gds.util.asNode(source).id AS source_id,
               gds.util.asNode(target).id AS target_id,
               totalCost AS shortest_path_length
        LIMIT 1000
        """

        # Query to compute diameter
        diameter_query = """
        CALL gds.alpha.allShortestPaths.stream('myGraph')
        YIELD nodeId, distance
        WITH max(distance) AS diameter
        RETURN diameter
        """

        features_df = pd.DataFrame()

        # Execute queries and merge results
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(shortest_path_query)
                features_df = pd.DataFrame([dict(record) for record in result])

                # Filter by node IDs if provided
                if node_ids is not None:
                    features_df = features_df[
                        features_df['source_id'].isin(node_ids) |
                        features_df['target_id'].isin(node_ids)
                    ]
        except Exception as e:
            logger.error(f"Error extracting path features: {e}")

        logger.info(f"Extracted path features for {len(features_df)} paths")
        return features_df

    def extract_temporal_features(
        self,
        node_ids: Optional[List[str]] = None,
        time_window: str = '1h'
    ) -> pd.DataFrame:
        """
        Extract temporal graph features.

        Args:
            node_ids: List of node IDs (if None, extract for all)
            time_window: Time window for aggregation (e.g., '1h', '1d', '1w')

        Returns:
            DataFrame with temporal features
        """
        logger.info(f"Extracting temporal features (window: {time_window})")

        if self.neo4j_driver is None:
            raise ValueError("Neo4j driver not configured")

        # Query to compute time-based degree centrality
        temporal_degree_query = f"""
        MATCH (n:Asset)-[r]->(m:Asset)
        WHERE r.timestamp >= datetime() - duration('{time_window}')
        WITH n.id AS node_id, count(r) AS temporal_degree
        RETURN node_id, temporal_degree
        """

        # Query to compute time-based PageRank
        temporal_pagerank_query = f"""
        CALL gds.pageRank.stream('myGraph', {{
            relationshipTypes: ['CONNECTED'],
            relationshipProperties: ['timestamp'],
            minFloat: 0.0,
            maxFloat: 1.0
        }})
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).id AS node_id, score AS temporal_pagerank
        """

        # Execute queries and merge results
        features_df = self._execute_and_merge_queries([
            temporal_degree_query, temporal_pagerank_query
        ])

        # Filter by node IDs if provided
        if node_ids is not None:
            features_df = features_df[features_df['node_id'].isin(node_ids)]

        logger.info(f"Extracted temporal features for {len(features_df)} nodes")
        return features_df

    def _execute_and_merge_queries(
        self,
        queries: List[str]
    ) -> pd.DataFrame:
        """
        Execute multiple Cypher queries and merge results.

        Args:
            queries: List of Cypher queries

        Returns:
            Merged DataFrame
        """
        results = []

        for query in queries:
            try:
                with self.neo4j_driver.session() as session:
                    query_result = session.run(query)
                    df = pd.DataFrame([dict(record) for record in query_result])
                    results.append(df)
            except Exception as e:
                logger.error(f"Error executing query: {e}")
                continue

        # Merge all results on node_id
        if results:
            merged_df = results[0]
            for df in results[1:]:
                merged_df = pd.merge(merged_df, df, on='node_id', how='outer')
            return merged_df
        else:
            return pd.DataFrame()


class TemporalAggregator:
    """
    Aggregates flow features over temporal windows.

    Supports windows from 1 second to 1 month:
    - 1s, 10s, 30s, 1m, 5m, 15m, 30m, 1h, 6h, 12h, 1d, 1w, 2w, 1mo

    Aggregation functions:
    - count, sum, mean, std, min, max, entropy, rate
    """

    def __init__(self, timestamp_col: str = 'timestamp'):
        """
        Initialize temporal aggregator.

        Args:
            timestamp_col: Name of timestamp column
        """
        self.timestamp_col = timestamp_col

    # Supported time windows
    TIME_WINDOWS = {
        '1s': '1s',
        '10s': '10s',
        '30s': '30s',
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1h',
        '6h': '6h',
        '12h': '12h',
        '1d': '1D',
        '1w': '1W',
        '2w': '2W',
        '1mo': '1ME'
    }

    # Aggregation functions
    AGG_FUNCTIONS = ['count', 'sum', 'mean', 'std', 'min', 'max', 'entropy', 'rate']

    def aggregate(
        self,
        df: pd.DataFrame,
        window: str = '1h',
        functions: Optional[List[str]] = None,
        group_by: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Aggregate flow features over temporal window.

        Args:
            df: DataFrame with flow features
            window: Time window (e.g., '1h', '1d', '1w')
            functions: List of aggregation functions
            group_by: Optional list of columns to group by

        Returns:
            Aggregated DataFrame
        """
        if window not in self.TIME_WINDOWS:
            raise ValueError(f"Invalid time window: {window}. "
                           f"Supported: {list(self.TIME_WINDOWS.keys())}")

        if functions is None:
            functions = self.AGG_FUNCTIONS

        logger.info(f"Aggregating features over {window} window "
                   f"(functions: {functions})")

        # Ensure timestamp column exists
        if self.timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_col}' not found")

        # Set timestamp as index and sort
        df_sorted = df.set_index(self.timestamp_col).sort_index()

        # Resample by time window
        resampled = df_sorted.resample(self.TIME_WINDOWS[window])

        # Aggregate features
        agg_df = self._apply_aggregations(resampled, functions)

        # Add timestamp back
        agg_df['timestamp'] = agg_df.index

        logger.info(f"Aggregated {len(agg_df)} time windows")
        return agg_df

    def _apply_aggregations(
        self,
        resampled: pd.core.resample.Resampler,
        functions: List[str]
    ) -> pd.DataFrame:
        """
        Apply aggregation functions to resampled data.

        Args:
            resampled: Resampled DataFrame
            functions: List of aggregation functions

        Returns:
            Aggregated DataFrame
        """
        # Select numeric columns
        numeric_cols = resampled.obj.select_dtypes(include=[np.number]).columns

        # Build aggregation list (pandas 2.x requires a list, not a name→name dict)
        agg_funcs = []
        for func in functions:
            if func in ['count', 'sum', 'mean', 'std', 'min', 'max']:
                agg_funcs.append(func)
            elif func == 'entropy':
                agg_funcs.append(self._compute_entropy_column)
            elif func == 'rate':
                agg_funcs.append(lambda x: x.diff().fillna(0))

        # Apply aggregations
        if agg_funcs:
            agg_df = resampled[numeric_cols].agg(agg_funcs)
        else:
            agg_df = resampled[numeric_cols].mean()

        # Flatten multi-level column index
        if isinstance(agg_df.columns, pd.MultiIndex):
            agg_df.columns = [
                f'{col[0]}_{col[1]}' if col[1] else col[0]
                for col in agg_df.columns
            ]

        return agg_df

    def _compute_entropy_column(self, series: pd.Series) -> float:
        """
        Compute Shannon entropy for a column.

        Args:
            series: Pandas series

        Returns:
            Entropy value
        """
        # Discretize values
        binned = pd.cut(series, bins=20, labels=False, duplicates='drop')

        # Count occurrences
        counts = np.bincount(binned[~np.isnan(binned)].astype(int))

        # Compute probabilities
        probs = counts / counts.sum()

        # Compute entropy
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = -np.sum(probs * np.log2(probs + 1e-10))

        return entropy


class FeatureExtractionPipeline:
    """
    Main pipeline for feature extraction.

    Orchestrates extraction of:
    1. Raw flow features (84+ features)
    2. Behavioral features (statistical, entropy, ratios, time)
    3. Graph features (centrality, structural, path, temporal)
    4. Temporal aggregations
    """

    def __init__(
        self,
        dataset_path: str,
        neo4j_driver=None,
        timestamp_col: str = 'timestamp'
    ):
        """
        Initialize feature extraction pipeline.

        Args:
            dataset_path: Path to CSE-CIC-IDS2018 dataset
            neo4j_driver: Neo4j driver instance
            timestamp_col: Name of timestamp column
        """
        self.flow_extractor = FlowFeatureExtractor(dataset_path)
        self.behavioral_extractor = BehavioralFeatureExtractor()
        self.graph_extractor = GraphFeatureExtractor(neo4j_driver)
        self.temporal_aggregator = TemporalAggregator(timestamp_col)

    def extract_all_features(
        self,
        df: pd.DataFrame,
        extract_graph_features: bool = False,
        time_windows: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract all features from flow data.

        Args:
            df: DataFrame with raw flow records
            extract_graph_features: Whether to extract graph features
            time_windows: List of time windows for temporal aggregation

        Returns:
            DataFrame with all extracted features
        """
        logger.info("Starting comprehensive feature extraction")

        # Extract raw flow features (84+ features)
        df = self.flow_extractor.extract_raw_features(df)

        # Extract behavioral features
        df = self.behavioral_extractor.extract_statistical_features(df)
        df = self.behavioral_extractor.extract_entropy_features(df)
        df = self.behavioral_extractor.extract_ratio_features(df)
        df = self.behavioral_extractor.extract_time_features(df)

        # Extract graph features (optional)
        if extract_graph_features and self.graph_extractor.neo4j_driver:
            # Get node IDs from flow data
            node_ids = df.get('src_ip', df.get('source', None))
            if node_ids is not None:
                centrality_df = self.graph_extractor.extract_centrality_features(
                    node_ids.unique().tolist()
                )
                structural_df = self.graph_extractor.extract_structural_features(
                    node_ids.unique().tolist()
                )
                # Merge graph features
                df = df.merge(centrality_df, left_on='src_ip', right_on='node_id', how='left')
                df = df.merge(structural_df, left_on='src_ip', right_on='node_id', how='left')

        # Temporal aggregation (optional)
        if time_windows:
            for window in time_windows:
                agg_df = self.temporal_aggregator.aggregate(df, window=window)
                # Reset index so 'timestamp' is a plain column, not both index and column
                agg_df = agg_df.reset_index(drop=True)
                # Merge aggregated features back
                df = df.merge(
                    agg_df,
                    left_on=self.temporal_aggregator.timestamp_col,
                    right_on='timestamp',
                    how='left',
                    suffixes=('', f'_{window}')
                )

        logger.info(f"Feature extraction complete. Total features: {len(df.columns)}")
        return df


def main():
    """Example usage of feature extraction pipeline."""
    # This is a placeholder - update paths and configurations as needed
    pipeline = FeatureExtractionPipeline(
        dataset_path=str(Path(__file__).resolve().parents[3] / "data" / "raw" / "CSE-CIC-IDS2018"),
        neo4j_driver=None,  # Add Neo4j driver if available
        timestamp_col="timestamp"
    )

    # Load data (placeholder - replace with actual data loading)
    import pandas as pd
    df = pd.DataFrame()

    # Extract all features
    features_df = pipeline.extract_all_features(
        df,
        extract_graph_features=False,  # Set True if Neo4j available
        time_windows=['1h', '1d', '1w']  # Optional temporal aggregation
    )

    print(f"Extracted features shape: {features_df.shape}")
    print(f"Feature columns: {features_df.columns.tolist()[:20]}")


if __name__ == "__main__":
    main()
