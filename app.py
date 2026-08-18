from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import warnings
from transformers import logging as hf_logging
import logging

# Silence terminal warnings for a clean demo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from pipeline_modified import MentalStatePipeline

app = Flask(__name__)
CORS(app) # This allows your HTML file to talk to Python securely

print("🚀 INITIALIZING IMPACT AI ML ENGINE...")
engine = MentalStatePipeline()
print("✅ Engine Ready. Waiting for dashboard data...")

@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. Grab the typing telemetry from the frontend
    text = request.form.get('text', '')
    speed = float(request.form.get('speed', 0))
    backspaces = int(request.form.get('backspaces', 0))
    latency = float(request.form.get('latency', 0))

    typing_metrics = {
        "speed": speed, 
        "backspaces": backspaces, 
        "latency": latency
    }

    # 2. Grab the audio file if it exists
    audio_file = request.files.get('audio_file')
    audio_path = None

    if audio_file:
        audio_path = "demo_temp.wav"
        audio_file.save(audio_path)

    # 3. RUN YOUR MASTERPIECE
    output = engine.run(
        text=text,
        typing_metrics=typing_metrics,
        history=[], 
        local_hour=14, # Mocking a daytime hour for the demo
        audio_path=audio_path
    )

    # 4. Clean up the temp audio file
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)

    return jsonify(output)

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(port=5000, debug=False)