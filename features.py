import numpy as np
from typing import Dict, Any
from nltk.sentiment import SentimentIntensityAnalyzer
import math

class FeatureExtractor:
    def __init__(self):
        # Initialize VADER for orthogonal sentiment validation
        self.vader = SentimentIntensityAnalyzer()

    def compute_negativity(self, scores: dict, text: str) -> float:
        # 1. Deterministic Crisis Override (Hackathon Fail-Safe)
        crisis_keywords = ["can't do this anymore", "overwhelmed", "give up", "too much", "help"]
        if any(k in text.lower() for k in crisis_keywords):
            return 1.0  # Max out the negativity score immediately

        # 2. Standard ML Probability 
        return (
            0.35 * scores.get("sadness", 0) +
            0.30 * scores.get("fear", 0) +
            0.20 * scores.get("anger", 0) +
            0.15 * scores.get("disgust", 0)
        )
        
    def compute_positivity(self, scores: Dict[str, float]) -> float:
        return scores.get("joy", 0.0) + 0.5 * scores.get("surprise", 0.0)

    def compute_uncertainty(self, text: str) -> float:
        tokens = ["maybe", "idk", "not sure", "guess", "perhaps", "confused"]
        text_lower = text.lower()
        count = sum(text_lower.count(t) for t in tokens)
        # Smooth scaling instead of hard caps
        return 1.0 - math.exp(-0.5 * count)

    def compute_repetition(self, text: str) -> float:
        """Type-Token Ratio (TTR) inverted. High score = high repetition (rumination)."""
        words = text.lower().split()
        if not words: return 0.0
        ttr = len(set(words)) / len(words)
        return 1.0 - ttr

    def typing_irregularity(self, metrics: Dict[str, float]) -> float:
        # Avoid zero division with epsilons
        speed = metrics.get("speed", 1.0)
        backspaces = metrics.get("backspaces", 0.0)
        latency = metrics.get("latency", 0.0)

        # Non-linear scaling: penalize heavy backspacing and high latency exponentially
        b_score = 1.0 - math.exp(-0.2 * backspaces)
        l_score = 1.0 - math.exp(-0.5 * latency)
        s_score = 1.0 / (speed + 0.1)

        return (0.4 * b_score) + (0.4 * l_score) + (0.2 * min(s_score, 1.0))

    def detect_masking(self, positivity: float, typing_irreg: float) -> bool:
        """Flags when text is highly positive but physical typing is highly erratic."""
        return positivity > 0.7 and typing_irreg > 0.6