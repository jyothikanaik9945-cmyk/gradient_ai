import numpy as np

class BaselineManager:

    def compute_baseline(self, history: list) -> dict:
        csi_values = [h["csi"] for h in history]

        return {
            "mean": float(np.mean(csi_values)),
            "std": float(np.std(csi_values) + 1e-6)
        }

    def compute_zscore(self, csi: float, baseline: dict) -> float:
        return (csi - baseline["mean"]) / baseline["std"]