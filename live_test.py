import time
import os
import warnings
import json
from datetime import datetime
from pynput import keyboard
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import logging as hf_logging
import logging

# --- SILENCE WARNINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from pipeline import MentalStatePipeline

# --- GLOBALS FOR THE KEYLOGGER ---
typed_chars = []
backspace_count = 0
key_times = []
is_typing = True
import sys # Make sure to add this at the top!

def on_press(key):
    global backspace_count, is_typing
    key_times.append(time.time())
    try:
        if key.char is not None:
            typed_chars.append(key.char)
            # Print the character to the terminal immediately
            sys.stdout.write(key.char)
            sys.stdout.flush()
    except AttributeError:
        if key == keyboard.Key.space:
            typed_chars.append(' ')
            sys.stdout.write(' ')
            sys.stdout.flush()
        elif key == keyboard.Key.backspace:
            backspace_count += 1
            if typed_chars:
                typed_chars.pop()
                # Visually erase the character from the terminal screen
                sys.stdout.write('\b \b') 
                sys.stdout.flush()
        elif key == keyboard.Key.enter:
            is_typing = False
            print() # Move to the next line when done
            return False # Stop listener

def record_voice(duration=4, filename="live_audio.wav"):
    fs = 16000
    print(f"\n🔴 RECORDING FOR {duration} SECONDS... SPEAK NOW!")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, myrecording)
    print("✅ Audio captured successfully.\n")
    return filename

def run_live_test():
    print("🚀 INITIALIZING LIVE IMPACT AI HARDWARE TESTER...")
    engine = MentalStatePipeline()
    print("✅ Engine Ready.\n")

    print("=" * 60)
    print("⌨️  STEP 1: TYPING TELEMETRY TEST")
    print("Type a sentence naturally below. Feel free to make mistakes and use backspaces.")
    print("Press ENTER when you are done.")
    print("=" * 60)

    # Start the keylogger
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Keep the main thread alive while the user types
    while is_typing:
        time.sleep(0.1)

    # --- CALCULATE LIVE HARDWARE METRICS ---
    text_result = "".join(typed_chars).strip()
    total_time = key_times[-1] - key_times[0] if len(key_times) > 1 else 1.0
    word_count = len(text_result.split())
    wpm = (word_count / total_time) * 60 if total_time > 0 else 0

    # Calculate inter-keystroke latency in milliseconds
    latencies = [key_times[i] - key_times[i-1] for i in range(1, len(key_times))]
    avg_latency = (sum(latencies) / len(latencies) * 1000) if latencies else 0

    live_metrics = {
        "speed": round(wpm, 2),
        "backspaces": backspace_count,
        "latency": round(avg_latency, 2)
    }

    print(f"\n📩 Captured Text: \"{text_result}\"")
    print(f"📊 Live Telemetry: {live_metrics}")

    # --- AUDIO TEST ---
    print("\n" + "=" * 60)
    print("🎤 STEP 2: VOICE MODULATION TEST")
    print("Get ready. I will record your voice for 4 seconds.")
    print("=" * 60)
    
    time.sleep(2) # Give you a second to clear your throat
    audio_file = record_voice()

    # --- RUN THE ENGINE ---
    print("🧠 RUNNING MULTIMODAL INFERENCE ON YOUR LIVE DATA...")
    output = engine.run(
        text=text_result,
        typing_metrics=live_metrics,
        history=[], 
        local_hour=datetime.now().hour,
        audio_path=audio_file
    )

    print("\n📦 FINAL API PAYLOAD:")
    report_md = output.pop("session_report_md")
    print(json.dumps(output, indent=2))
    
    print("\n📝 CLINICAL REPORT:")
    print("=" * 60)
    print(report_md)
    print("=" * 60)

if __name__ == "__main__":
    run_live_test()