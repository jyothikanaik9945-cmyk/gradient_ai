import librosa
import numpy as np
import warnings

# Silence librosa warnings
warnings.filterwarnings('ignore', category=UserWarning)

class VoiceModulationEngine:
    def __init__(self):
        pass

    def extract_vocal_telemetry(self, audio_file_path: str) -> dict:
        """
        Analyzes the physical modulation of a voice note, filtering out room static.
        """
        try:
            # Load the audio file
            y, sr = librosa.load(audio_file_path, sr=16000)
            
            # --- THE MAGIC FIX: TRIM THE STATIC ---
            # This strips away the background room noise below 20 decibels
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            # Fallback just in case the entire clip was silent
            if len(y_trimmed) == 0:
                y_trimmed = y

            # 1. Speaking Rate (Calculated ONLY on actual speech)
            onsets = librosa.onset.onset_detect(y=y_trimmed, sr=sr)
            duration = librosa.get_duration(y=y_trimmed, sr=sr)
            speaking_rate = len(onsets) / duration if duration > 0 else 0

            # 2. Voice Tremor (Calculated ONLY on actual speech)
            centroids = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]
            voice_volatility = float(np.std(centroids))

            # 3. Energy Variance
            rms_energy = librosa.feature.rms(y=y_trimmed)[0]
            energy_variance = float(np.std(rms_energy))

            # --- HACKATHON DEMO TUNING ---
            normalized_rate = min(speaking_rate / 3.0, 1.0) 
            normalized_volatility = min(voice_volatility / 800.0, 1.0) 
            
            vocal_strain = round((0.7 * normalized_volatility) + (0.3 * normalized_rate), 4)
            
            print(f"   🎤 [Audio Engine] Trimmed Volatility: {voice_volatility:.1f} | Final Vocal Strain: {vocal_strain:.2f}")
            
            return {
                "speaking_rate": round(normalized_rate, 4),
                "voice_volatility": round(normalized_volatility, 4),
                "energy_variance": round(energy_variance, 4),
                "vocal_strain_index": vocal_strain
            }
            
        except Exception as e:
            print(f"Audio processing error: {e}")
            return {"speaking_rate": 0, "voice_volatility": 0, "vocal_strain_index": 0}