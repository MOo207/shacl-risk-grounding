"""
CSE-CIC-IDS2018 Dataset Loader

Loads and preprocesses the CSE-CIC-IDS2018 AWS network traffic dataset.
This dataset contains network flow records with labeled benign and malicious activity.

Dataset Structure:
- CSV files for each attack scenario (Botnet, DDoS, DoS, Infiltration, Web Attacks)
- Flow features: Timestamp, Source/Dest IPs, Ports, Protocols, Packet/Byte counts, Flags
- Labels: Benign, Attack type, Attack subtype

Reference:
Sharafaldin, I., Lashkari, A. H., Hakak, S. & Ghorbani, A. A. (2018).
"CIC-IDS2018: A Deep Learning Approach for Network Intrusion Detection System".
Proceedings of the International Conference on Computer, Information, and Telecommunication Systems (CICITS).
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Canonical 8-class label normalization map
LABEL_MAP = {
    "Benign": "Benign",        "BENIGN": "Benign",
    "FTP-BruteForce": "BruteForce",   "SSH-Bruteforce": "BruteForce",
    "Brute Force -Web": "BruteForce", "Brute Force -XSS": "BruteForce",
    "DoS attacks-Hulk": "DoS",        "DoS attacks-SlowHTTPTest": "DoS",
    "DoS attacks-GoldenEye": "DoS",   "DoS attacks-Slowloris": "DoS",
    "SQL Injection": "WebAttack",      "XSS": "WebAttack",
    "Infiltration": "Infiltration",    "Infilteration": "Infiltration",
    "infilteration": "Infiltration",   "Bot": "Bot",
    "DDOS attack-HOIC": "DDoS",       "DDoS attacks-LOIC-HTTP": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",   "DDOS-LOIC-HTTP": "DDoS",
    "DDOS-HOIC": "DDoS",
}


def fix_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard CIC-IDS2018 cleaning: strip columns, drop header rows,
    replace Inf/NaN, clip negative IAT, drop duplicates, normalize labels."""
    df.columns = [c.strip() for c in df.columns]          # strip whitespace only — preserve case
    if "Label" in df.columns:
        df = df[df["Label"] != "Label"]                    # drop embedded header rows
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in [c for c in df.columns if "iat" in c.lower()]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)
    df = df.drop_duplicates()
    if "Label" in df.columns:
        df["Label"] = df["Label"].str.strip().map(lambda x: LABEL_MAP.get(x, x))
    return df


class CSECICIDS2018Loader:
    """Loader for CSE-CIC-IDS2018 dataset."""

    # Attack scenarios in CSE-CIC-IDS2018
    ATTACK_SCENARIOS = [
        "Botnet",
        "DDOS",
        "DoS",
        "Infiltration",
        "WebAttacks"
    ]

    # Attack subtypes
    ATTACK_SUBTYPES = {
        "Botnet": ["Botnet"],
        "DDOS": ["DDOS attack-LOIC-HTTP", "DDOS attack-HOIC"],
        "DoS": ["DoS attacks-Hulk", "DoS attacks-SlowHTTPTest", "DoS attacks-GoldenEye", "DoS attacks-Slowloris"],
        "Infiltration": ["Infiltration", "Bruteforce", "SQL Injection"],
        "WebAttacks": ["Brute Force", "XSS", "SQL Injection"]
    }

    def __init__(self, dataset_path: str):
        """
        Initialize the loader.

        Args:
            dataset_path: Path to CSE-CIC-IDS2018 dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.flow_records: List[Dict] = []

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    def load_all_scenarios(self) -> pd.DataFrame:
        """
        Load all attack scenarios from the dataset.

        Returns:
            Combined DataFrame with all flow records
        """
        all_data = []

        for scenario in self.ATTACK_SCENARIOS:
            logger.info(f"Loading scenario: {scenario}")
            scenario_data = self.load_scenario(scenario)
            if scenario_data is not None and not scenario_data.empty:
                all_data.append(scenario_data)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"Loaded {len(combined)} total flow records")
            return combined
        else:
            logger.warning("No data loaded from any scenario")
            return pd.DataFrame()

    def load_scenario(self, scenario: str) -> Optional[pd.DataFrame]:
        """
        Load a specific attack scenario.

        Args:
            scenario: Attack scenario name (e.g., "Botnet", "DDOS")

        Returns:
            DataFrame with flow records for the scenario, or None if not found
        """
        # Try to find CSV files for this scenario
        scenario_dir = self.dataset_path / scenario

        if scenario_dir.exists() and scenario_dir.is_dir():
            # Load all CSV files in the scenario directory
            csv_files = list(scenario_dir.glob("*.csv"))

            if not csv_files:
                logger.warning(f"No CSV files found in {scenario_dir}")
                return None

            all_files_data = []
            for csv_file in csv_files:
                logger.info(f"Loading {csv_file.name}")
                df = self._load_csv_file(csv_file, scenario)
                if df is not None:
                    all_files_data.append(df)

            if all_files_data:
                return pd.concat(all_files_data, ignore_index=True)
            else:
                return None
        else:
            # Try to find CSV file directly with scenario name
            csv_file = self.dataset_path / f"{scenario}.csv"
            if csv_file.exists():
                return self._load_csv_file(csv_file, scenario)
            else:
                logger.warning(f"Scenario not found: {scenario}")
                return None

    def _load_csv_file(self, csv_file: Path, scenario: str) -> Optional[pd.DataFrame]:
        """
        Load a single CSV file.

        Args:
            csv_file: Path to CSV file
            scenario: Attack scenario name

        Returns:
            DataFrame with flow records, or None if loading fails
        """
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'ISO-8859-1']:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.error(f"Failed to decode {csv_file.name} with any encoding")
                return None

            # Apply standard cleaning (strip cols, fix Inf/NaN, normalize labels)
            df = fix_df(df)

            # Add metadata columns
            df['scenario'] = scenario
            df['source_file'] = csv_file.name

            # Parse timestamp if exists (column name preserved as-is after fix_df)
            ts_col = next((c for c in df.columns if c.lower() == 'timestamp'), None)
            if ts_col:
                df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')

            # Derive binary attack flag from normalized Label
            if 'Label' in df.columns:
                df['is_attack'] = df['Label'] != 'Benign'
                df['is_benign'] = df['Label'] == 'Benign'
            else:
                df['is_attack'] = False
                df['is_benign'] = True

            logger.info(f"Loaded {len(df)} records from {csv_file.name}")
            return df

        except Exception as e:
            logger.error(f"Error loading {csv_file.name}: {e}")
            return None

    def get_flow_statistics(self) -> Dict:
        """
        Get statistics about loaded flow records.

        Returns:
            Dictionary with flow statistics
        """
        if not self.flow_records:
            return {}

        df = pd.DataFrame(self.flow_records)

        stats = {
            'total_records': len(df),
            'benign_records': len(df[df['is_benign']]) if 'is_benign' in df.columns else 0,
            'attack_records': len(df[df['is_attack']]) if 'is_attack' in df.columns else 0,
            'unique_source_ips': df['src_ip'].nunique() if 'src_ip' in df.columns else 0,
            'unique_dest_ips': df['dest_ip'].nunique() if 'dest_ip' in df.columns else 0,
            'protocols': df['protocol'].unique().tolist() if 'protocol' in df.columns else [],
            'scenarios': df['scenario'].unique().tolist() if 'scenario' in df.columns else []
        }

        return stats

    def extract_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract flow features for ML training.

        Args:
            df: DataFrame with raw flow records

        Returns:
            DataFrame with extracted features
        """
        features_df = df.copy()

        # Basic flow features
        numeric_columns = features_df.select_dtypes(include=['number']).columns

        # Replace Inf with NaN, then fill with 0
        features_df[numeric_columns] = (
            features_df[numeric_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        # Derived features
        if 'packet_length' in features_df.columns and 'total_fwd_packets' in features_df.columns:
            features_df['avg_packet_size'] = features_df['packet_length'] / (
                features_df['total_fwd_packets'] + features_df.get('total_backward_packets', 1)
            )

        if 'flow_duration' in features_df.columns:
            features_df['packet_rate'] = (
                features_df.get('total_fwd_packets', 0) + features_df.get('total_backward_packets', 0)
            ) / (features_df['flow_duration'] + 1)
            features_df['byte_rate'] = features_df.get('total_length_of_fwd_packet', 0) / (
                features_df['flow_duration'] + 1
            )

        # Flow ratio features
        if 'total_fwd_packets' in features_df.columns and 'total_backward_packets' in features_df.columns:
            features_df['fwd_bwd_packet_ratio'] = features_df['total_fwd_packets'] / (
                features_df['total_backward_packets'] + 1
            )

        if 'total_length_of_fwd_packet' in features_df.columns and 'total_length_of_bwd_packet' in features_df.columns:
            features_df['fwd_bwd_byte_ratio'] = features_df['total_length_of_fwd_packet'] / (
                features_df.get('total_length_of_bwd_packet', 1)
            )

        logger.info(f"Extracted features for {len(features_df)} records")
        return features_df


def main():
    """Example usage of the CSE-CIC-IDS2018 loader."""
    # This is a placeholder - update the path to actual dataset location
    loader = CSECICIDS2018Loader(
        str(Path(__file__).resolve().parents[3] / "data" / "raw" / "CSE-CIC-IDS2018")
    )

    try:
        # Load all scenarios
        df = loader.load_all_scenarios()

        if not df.empty:
            # Get statistics
            stats = loader.get_flow_statistics()
            logger.info(f"Flow statistics: {stats}")

            # Extract features
            features_df = loader.extract_flow_features(df)
            logger.info(f"Features shape: {features_df.shape}")

            # Save to file
            output_path = Path("output/cse_cic_ids2018_processed.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")

    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()
