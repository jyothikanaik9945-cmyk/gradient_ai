import numpy as np
from typing import List, Dict, Tuple

class RiskModel:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha # EMA smoothing factor

    def compute(self, current_csi: float, z_score: float, is_masking: bool, history: List[Dict]) -> Tuple[float, List[str]]:
        explanations = []
        
        if not history:
            return current_csi * 0.5, ["Insufficient baseline. Defaulting to safe risk floor."]

        # 1. EMA Trend Calculation
        historical_csi = [h["csi"] for h in history]
        ema = historical_csi[0]
        for past_csi in historical_csi[1:]:
            ema = self.alpha * past_csi + (1 - self.alpha) * ema
        
        current_ema = self.alpha * current_csi + (1 - self.alpha) * ema
        trend_delta = current_ema - ema

        # 2. Risk Scoring Component
        risk = 0.0
        
        if trend_delta > 0.05:
            risk += 0.4
            explanations.append(f"Steady upward trend in cognitive strain detected (+{trend_delta:.2f}).")
            
        if z_score > 1.5: # Anomaly detected
            risk += 0.4
            explanations.append(f"Sudden behavioral spike: CSI is {z_score:.1f} standard deviations above user baseline.")
            
        if is_masking:
            risk += 0.2
            explanations.append("Hidden distress flagged: High emotional positivity contradicted by erratic typing telemetry.")

        # Smooth the final risk against previous risk state to prevent UI whiplash
        prev_risk = history[-1].get("risk", 0.0)
        final_risk = (0.7 * np.clip(risk, 0.0, 1.0)) + (0.3 * prev_risk)
        
        if final_risk < 0.3 and not explanations:
            explanations.append("Behavioral signals remain stable within user baseline.")

        return float(final_risk), explanations