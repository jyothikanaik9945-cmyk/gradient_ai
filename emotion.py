from transformers import pipeline
from functools import lru_cache

class EmotionEngine:
    def __init__(self):
        self.model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )

    # Add this decorator! It stores the last 128 inferences in RAM.
    @lru_cache(maxsize=128)
    def _cached_predict(self, text: str) -> tuple:
        # Tuples are hashable, dicts are not, so we return a tuple of tuples for the cache
        outputs = self.model(text)[0]
        return tuple((item['label'], item['score']) for item in outputs)

    def predict(self, text: str) -> dict:
        # Convert the cached tuple back into your expected dictionary format
        cached_outputs = self._cached_predict(text)
        scores = {label: score for label, score in cached_outputs}
        
        return {
            "scores": scores,
            "confidence": max(scores.values()) if scores else 0.0
        }