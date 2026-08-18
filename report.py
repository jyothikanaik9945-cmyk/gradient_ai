from typing import List, Dict
import numpy as np

class ReportGenerator:
    def generate_summary(self, history: List[Dict]) -> str:
        if not history:
            return "Insufficient data for clinical summary."

        # Extract temporal arrays
        csi_trend = [h["csi"] for h in history]
        risk_trend = [h.get("risk", 0.0) for h in history]
        masking_events = sum(1 for h in history if h.get("is_masking", False))
        
        # Calculate session metrics
        max_risk = max(risk_trend)
        avg_csi = np.mean(csi_trend)
        trend_delta = csi_trend[-1] - csi_trend[0]
        
        # Build the Markdown Report
        report = []
        report.append("### 🧠 AI Session Observability Report\n")
        
        report.append(f"**Session Duration:** {len(history)} interaction steps")
        report.append(f"**Peak Risk Score:** {max_risk:.2f} " + ("🔴 (CRITICAL)" if max_risk > 0.8 else "🟡 (ELEVATED)"))
        report.append(f"**Masking Flags Triggered:** {masking_events}\n")
        
        report.append("#### Temporal Trend Analysis:")
        if trend_delta > 0.15:
            report.append("- ⚠️ **Deteriorating:** Cognitive Strain Index steadily increased during the session.")
        elif trend_delta < -0.15:
            report.append("- 📉 **Improving:** User responded well to session, strain decreased.")
        else:
            report.append("- ⚖️ **Stable:** Strain remained within a consistent band.")
            
        # Grab the last few unique explanations from the backend
        recent_explanations = set()
        for h in history[-5:]: # Look at the last 5 interactions
            if "explanation" in h:
                for exp in h["explanation"]:
                    recent_explanations.add(exp)

        if recent_explanations:
            report.append("\n#### Key Behavioral Triggers:")
            for exp in recent_explanations:
                if exp != "Behavioral signals remain stable within user baseline.":
                    report.append(f"- {exp}")

        return "\n".join(report)