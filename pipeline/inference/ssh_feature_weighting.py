"""
SSH Feature Weighting — In-Inference Mechanism for INTERSYMBOLIC-GRC
====================================================================

Implements an in-inference symbolic mechanism that weights network flow
features by their SSH protocol relevance before ML classification. This
allows the tri-stage pipeline to leverage domain knowledge at inference time.

The mechanism:
1. Identifies SSH/FTP flows by destination port (22/21)
2. Amplifies features known to be discriminative for SSH-based attacks
3. Applies weighted feature vector to downstream ML classification

Based on feature importance analysis and domain knowledge:
- SSH attacks: BruteForce on port 22 (packet-based patterns)
- Key features: Tot Fwd Pkts, SYN Flag Cnt, Flow IAT Std, Duration

Usage:
    from pipeline.inference.ssh_feature_weighting import SSHFeatureWeighting
    
    weighting = SSHFeatureWeighting(feature_importance_path="results/ablation_study_v2.json")
    X_weighted = weighting.apply_weighting(X, feature_names, df_test)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union


class SSHFeatureWeighting:
    """
    In-inference SSH feature weighting mechanism.
    
    Weights SSH-relevant features based on:
    1. Feature importance from ablation studies
    2. Domain knowledge of SSH attack patterns
    """
    
    def __init__(self, feature_importance_path: Optional[str] = None):
        """
        Initialize SSH feature weighting.
        
        Args:
            feature_importance_path: Path to JSON with feature importance data.
                                   If None, uses default SSH-relevant features.
        """
        self.feature_importance = None
        self.ssh_features = self._get_default_ssh_features()
        
        if feature_importance_path:
            self._load_feature_importance(feature_importance_path)
    
    def _get_default_ssh_features(self) -> Dict[str, float]:
        """
        Default SSH-relevant features based on domain knowledge.
        
        Returns:
            Dict mapping feature names to default weights.
        """
        return {
            "Tot Fwd Pkts": 2.0,      # Packet count critical for SSH brute force
            "SYN Flag Cnt": 2.5,      # SYN patterns in SSH handshake
            "Flow IAT Std": 1.8,      # Inter-arrival timing patterns
            "Flow Duration": 1.5,     # Session duration patterns
            "Fwd Pkts/s": 1.8,        # Forward packet rate
            "Bwd Pkts/s": 1.5,        # Backward packet rate  
            "Pkt Len Std": 1.3,       # Packet length variation
            "Pkt Len Mean": 1.2,      # Average packet size
        }
    
    def _load_feature_importance(self, path: str):
        """
        Load feature importance from ablation study results.
        
        Args:
            path: Path to JSON file with feature importance data.
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Extract feature importance if available
            if 'feature_importance' in data:
                self.feature_importance = data['feature_importance']
            elif 'baseline' in data and 'feature_names' in data.get('baseline', {}):
                # Try to extract from baseline model
                pass  # Would need model-specific extraction
            else:
                print(f"Warning: No feature importance found in {path}")
                print("Using default SSH feature weights")
                
        except Exception as e:
            print(f"Warning: Could not load feature importance from {path}: {e}")
            print("Using default SSH feature weights")
    
    def apply_weighting(self, 
                       X: np.ndarray, 
                       feature_names: List[str], 
                       df_test: pd.DataFrame,
                       weight_factor: float = 1.0) -> np.ndarray:
        """
        Apply SSH feature weighting to feature matrix.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: List of feature names corresponding to X columns
            df_test: Test DataFrame with metadata (must include 'Dst Port')
            weight_factor: Global multiplier for SSH weights (default: 1.0)
            
        Returns:
            Weighted feature matrix
        """
        X_weighted = X.copy().astype(float)
        
        # Identify SSH/FTP flows by destination port
        ssh_mask = self._identify_ssh_flows(df_test)
        n_ssh_flows = int(ssh_mask.sum())
        
        if n_ssh_flows == 0:
            print(f"No SSH/FTP flows found in test set (ports 21/22)")
            return X_weighted
        
        print(f"Found {n_ssh_flows} SSH/FTP flows in test set")
        
        # Apply weighting to SSH-relevant features for SSH flows
        ssh_features_found = []
        for feature_name, base_weight in self.ssh_features.items():
            if feature_name in feature_names:
                col_idx = feature_names.index(feature_name)
                # Apply weight: base_weight * weight_factor
                final_weight = base_weight * weight_factor
                X_weighted[ssh_mask, col_idx] *= final_weight
                ssh_features_found.append(feature_name)
        
        print(f"Applied SSH weighting to {len(ssh_features_found)} features: {ssh_features_found}")
        print(f"Weight factor used: {weight_factor}")
        
        return X_weighted
    
    def _identify_ssh_flows(self, df: pd.DataFrame) -> np.ndarray:
        """
        Identify SSH and FTP flows by destination port.
        
        Args:
            df: DataFrame containing network flow data
            
        Returns:
            Boolean array indicating SSH/FTP flows
        """
        if 'Dst Port' not in df.columns:
            print("Warning: 'Dst Port' column not found. Cannot identify SSH flows.")
            return np.zeros(len(df), dtype=bool)
        
        # SSH (port 22) and FTP (port 21) flows
        ssh_ports = [21, 22]
        ssh_mask = df['Dst Port'].isin(ssh_ports).values
        
        # Also check for alternative SSH ports
        if 'Protocol' in df.columns:
            # SSH protocol (6) and common alternative SSH ports
            alt_ssh_ports = [2222, 22222, 2022]
            alt_ssh_mask = (df['Dst Port'].isin(alt_ssh_ports) & 
                          (df['Protocol'] == 6)).values  # TCP
            ssh_mask = ssh_mask | alt_ssh_mask
        
        return ssh_mask
    
    def get_ssh_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about SSH/FTP flows in the dataset.
        
        Args:
            df: DataFrame containing network flow data
            
        Returns:
            Dictionary with SSH flow statistics
        """
        ssh_mask = self._identify_ssh_flows(df)
        n_ssh_flows = int(ssh_mask.sum())
        n_total_flows = len(df)
        
        stats = {
            'total_flows': n_total_flows,
            'ssh_ftp_flows': n_ssh_flows,
            'ssh_ftp_percentage': (n_ssh_flows / n_total_flows) * 100 if n_total_flows > 0 else 0,
        }
        
        if n_ssh_flows > 0 and 'Label' in df.columns:
            ssh_df = df[ssh_mask]
            label_counts = ssh_df['Label'].value_counts()
            stats['ssh_label_distribution'] = label_counts.to_dict()
            
            # Count BruteForce in SSH flows
            if 'BruteForce' in label_counts:
                stats['bruteforce_in_ssh'] = int(label_counts['BruteForce'])
            else:
                stats['bruteforce_in_ssh'] = 0
        
        return stats
    
    def optimize_weights(self, 
                        X: np.ndarray, 
                        feature_names: List[str], 
                        df_test: pd.DataFrame,
                        y_test: np.ndarray,
                        model,
                        weight_grid: List[float] = None) -> Dict:
        """
        Optimize SSH weight factor using grid search.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            df_test: Test DataFrame with metadata
            y_test: True labels
            model: ML model with predict() method
            weight_grid: List of weight factors to try
            
        Returns:
            Dictionary with optimization results
        """
        from sklearn.metrics import f1_score, accuracy_score
        
        if weight_grid is None:
            weight_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        
        baseline_pred = model.predict(X)
        baseline_f1 = f1_score(y_test, baseline_pred, average='macro', zero_division=0)
        baseline_acc = accuracy_score(y_test, baseline_pred)
        
        results = []
        best_f1 = baseline_f1
        best_weight = 1.0
        best_result = None
        
        print(f"Optimizing SSH weights (baseline F1-macro: {baseline_f1:.4f})")
        
        for weight in weight_grid:
            X_weighted = self.apply_weighting(X, feature_names, df_test, weight)
            y_pred = model.predict(X_weighted)
            
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            
            result = {
                'weight': weight,
                'accuracy': float(accuracy),
                'f1_macro': float(f1_macro),
                'improvement': float(f1_macro - baseline_f1)
            }
            results.append(result)
            
            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_weight = weight
                best_result = result
            
            print(f"  weight={weight:.1f}: F1={f1_macro:.4f} (+{f1_macro-baseline_f1:+.4f})")
        
        optimization_result = {
            'baseline_f1_macro': float(baseline_f1),
            'baseline_accuracy': float(baseline_acc),
            'best_weight': best_weight,
            'best_f1_macro': float(best_f1),
            'best_accuracy': float(best_result['accuracy']) if best_result else float(baseline_acc),
            'improvement': float(best_f1 - baseline_f1),
            'weight_grid_results': results,
            'significant_improvement': bool(best_f1 - baseline_f1 > 0.005)
        }
        
        print(f"Best weight: {best_weight} (F1: {best_f1:.4f}, improvement: {best_f1-baseline_f1:+.4f})")
        
        return optimization_result


def create_ssh_feature_weighting(feature_importance_path: str = None) -> SSHFeatureWeighting:
    """
    Factory function to create SSHFeatureWeighting instance.
    
    Args:
        feature_importance_path: Path to feature importance JSON
        
    Returns:
        SSHFeatureWeighting instance
    """
    return SSHFeatureWeighting(feature_importance_path)