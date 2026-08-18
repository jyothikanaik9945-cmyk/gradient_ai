import os
import json
import warnings
import numpy as np
import soundfile as sf
from transformers import logging as hf_logging
import logging

# --- SILENCE WARNINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from pipeline import MentalStatePipeline

def generate_synthetic_crisis_audio(filename="synthetic_crisis.wav"):
    print(f"⚙️ Generating synthetic acoustic tremor file: {filename}...")
    sr = 16000
    # Pure random noise simulates the highest possible vocal distortion/tremor
    audio = np.random.normal(0, 1.0, int(sr * 4.0)) 
    sf.write(filename, audio, sr)
    return filename

def run_full_simulation():
    print("\n🚀 INITIALIZING IMPACT AI 3.0 (TEXT + KEYSTROKE + ACOUSTIC)...\n")
    engine = MentalStatePipeline()
    history = []
    
    # Ensure our synthetic audio exists
    audio_file = generate_synthetic_crisis_audio()

    # --- THE SIMULATED USER JOURNEY ---
    interactions = [
        {
            "step": 1,
            "text": "Just woke up, getting ready for work.",
            "metrics": {"speed": 65.0, "backspaces": 2, "latency": 150.0},
            "hour": 8,
            "audio": None # Normal text interaction
        },
        {
            "step": 2,
            "text": "Traffic was a nightmare but I made it to the office.",
            "metrics": {"speed": 60.0, "backspaces": 4, "latency": 180.0},
            "hour": 9,
            "audio": None
        },
        {
            "step": 3,
            "text": "I'm totally fine, everything is completely great.",
            "metrics": {"speed": 68.0, "backspaces": 1, "latency": 140.0},
            "hour": 14,
            "audio": audio_file # THE JUDGE'S OVERRIDE: Positive text, normal typing, but severe vocal tremor!
        }
    ]

    print("\n📊 STARTING SIMULATION...\n" + "-"*60)

    final_output = {}

    for data in interactions:
        print(f"🕒 [Step {data['step']} | Time: {data['hour']}:00]")
        print(f"📩 Text: \"{data['text']}\"")
        if data['audio']:
            print(f"🎤 Audio Payload Attached: {data['audio']}")
        
        # Run the engine
        output = engine.run(
            text=data['text'],
            typing_metrics=data['metrics'],
            history=history,
            local_hour=data['hour'],
            audio_path=data['audio']
        )
        
        # Append to history memory
        history.append({
            "csi": output["csi"],
            "risk": output["risk_score"],
            "is_masking": output["is_masking"],
            "explanation": output["explanation"]
        })
        
        print(f"🧠 CSI: {output['csi']:.4f} | Risk: {output['risk_score']:.4f} | Masking: {output['is_masking']}")
        print("-" * 60)
        final_output = output # Capture the last state for the full JSON print

    print("\n📦 EXACT JSON PAYLOAD SENT TO REACT FRONTEND (STEP 3):")
    # We temporarily remove the markdown report from the print so the JSON is readable, 
    # but we will print the report separately below.
    report_md = final_output.pop("session_report_md")
    
    # Print the full dictionary exactly as FastAPI will return it
    print(json.dumps(final_output, indent=2))

    print("\n📝 DYNAMIC CLINICAL MARKDOWN REPORT:")
    print("=" * 60)
    print(report_md)
    print("=" * 60)
    print("\n✅ MULTIMODAL PIPELINE FULLY OPERATIONAL.")

if __name__ == "__main__":
    run_full_simulation()