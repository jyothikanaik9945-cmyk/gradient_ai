import os
import warnings
import logging
from transformers import logging as hf_logging

# 1. Silence all the ML warnings so the backend server logs stay perfectly clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning)
hf_logging.set_verbosity_error()
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# 2. Import your master pipeline
from pipeline import MentalStatePipeline

# 3. THE SINGLETON INSTANCE
# This ensures the AI model only loads into RAM *once* when FastAPI starts.
print("🧠 IMPACT AI: Loading ML Models into memory... (This takes a few seconds)")
_global_engine = MentalStatePipeline()
print("✅ IMPACT AI: ML Engine successfully loaded and ready for requests.")

def analyze_user_state(text: str, typing_metrics: dict, history: list, local_hour: int = None) -> dict:
    """
    The single entry point for the FastAPI backend. 
    It passes the data to the pre-loaded ML pipeline.
    """
    try:
        # Run the engine
        return _global_engine.run(
            text=text,
            typing_metrics=typing_metrics,
            history=history,
            local_hour=local_hour
        )
    except Exception as e:
        print(f"❌ ML Engine Crash: {str(e)}")
        raise e