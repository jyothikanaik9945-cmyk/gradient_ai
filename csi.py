import math
import numpy as np
from typing import Dict, List

class CSIComputer:
    def __init__(self):
        # Rebalanced weights for new feature set
        self.weights = {
            "negativity": 0.35,
            "uncertainty": 0.15,
            "typing": 0.35,
            "repetition": 0.15
        }

    def compute_raw(self, features: Dict[str, float]) -> float:
        linear = (
            self.weights["negativity"] * features["negativity"] +
            self.weights["uncertainty"] * features["uncertainty"] +
            self.weights["typing"] * features["typing_irregularity"] +
            self.weights["repetition"] * features["repetition"]
        )
        # Sigmoid compression to keep CSI bounded [0, 1]
        return 1 / (1 + math.exp(-5 * (linear - 0.5)))

    def compute_z_score(self, current_csi: float, history: List[Dict]) -> float:
        """Task 2: Baseline Modeling. z = (csi - mean) / std"""
        if len(history) < 3:
            return 0.0 # Not enough baseline data
            
        historical_csi = [h["csi"] for h in history]
        mean_csi = np.mean(historical_csi)
        std_csi = np.std(historical_csi) + 1e-5 # Prevent division by zero
        
        return float((current_csi - mean_csi) / std_csi)