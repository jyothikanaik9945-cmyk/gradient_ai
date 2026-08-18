import datetime

from matplotlib import text
from emotion import EmotionEngine
from features import FeatureExtractor
from csi import CSIComputer
from risk_model import RiskModel
from intervention import InterventionEngine
from report import ReportGenerator
from audio_engine import VoiceModulationEngine

class MentalStatePipeline:
    def __init__(self):
        # 1. Initialize the Core Extractors
        self.emotion = EmotionEngine()
        self.features = FeatureExtractor()
        self.audio_engine = VoiceModulationEngine() # NEW: Acoustic Engine
        
        # 2. Initialize the Calculators
        self.csi = CSIComputer()
        self.risk = RiskModel(alpha=0.3)
        
        # 3. Initialize the Action Engines (Tier 2 Features)
        self.intervention = InterventionEngine()
        self.report_gen = ReportGenerator()


    def _build_contextual_response(
        self,
        text: str,
        emotion: dict,
        features: dict,
        csi: float,
        risk_score: float,
        is_masking: bool,
        trigger_sos: bool,
        suggested_intervention: str = None,
    ) -> str:
        """
        Build a short, analysis-driven conversational response.

        This is intentionally rule-based: it uses the signals already produced
        by this project rather than pretending that a separate LLM is generating
        a response.
        """
        scores = emotion.get("scores", {}) or {}
        dominant_emotion = max(scores, key=scores.get) if scores else "neutral"

        emotion_phrases = {
            "joy": "a positive tone",
            "sadness": "a more subdued tone",
            "anger": "a frustrated tone",
            "fear": "a worried tone",
            "surprise": "a noticeable change in tone",
            "disgust": "a negative tone",
            "neutral": "a relatively neutral tone",
        }

        emotion_questions = {
            "joy": "What has been going well for you today?",
            "sadness": "What has been weighing on you most?",
            "anger": "What has been frustrating you most?",
            "fear": "What is making you feel most concerned right now?",
            "surprise": "What happened that felt unexpected?",
            "disgust": "What part of the situation is bothering you most?",
            "neutral": "What would you like to focus on right now?",
        }

        tone = emotion_phrases.get(dominant_emotion, "a changing emotional tone")
        follow_up = emotion_questions.get(dominant_emotion, "What would you like to talk about next?")

        negativity = float(features.get("negativity", 0.0) or 0.0)
        positivity = float(features.get("positivity", 0.0) or 0.0)
        uncertainty = float(features.get("uncertainty", 0.0) or 0.0)
        repetition = float(features.get("repetition", 0.0) or 0.0)
        typing_irregularity = float(features.get("typing_irregularity", 0.0) or 0.0)

        if trigger_sos or risk_score >= 0.70:
            return (
                f"Your message shows {tone}, and the current analysis indicates a "
                f"high level of strain (risk {risk_score:.2f}). "
                f"I'd like to understand what is happening rather than assume. "
                f"{follow_up}"
            )

        if is_masking:
            return (
                f"Your message has {tone}, but the behavioral signals are not fully "
                f"consistent with the text. The analysis flagged possible masking. "
                f"Could you tell me more about how you are actually feeling right now?"
            )

        if risk_score > 0.40:
            if uncertainty > 0.20:
                detail = "There is also some uncertainty in the wording."
            elif negativity > 0.35:
                detail = "The text also contains a stronger negative signal."
            elif typing_irregularity > 0.60:
                detail = "The typing pattern also shows increased irregularity."
            else:
                detail = "The overall strain signal is above the stable range."

            return (
                f"I picked up {tone} in your message. {detail} "
                f"The current risk estimate is {risk_score:.2f}. {follow_up}"
            )

        if dominant_emotion == "joy" or positivity > 0.55:
            return (
                f"Your message has {tone}, and the current signals look relatively "
                f"stable (risk {risk_score:.2f}). {follow_up}"
            )

        if dominant_emotion == "neutral" and negativity < 0.15 and uncertainty < 0.20:
            return (
                f"Your message has {tone}, with the current behavioral signals "
                f"remaining relatively stable (risk {risk_score:.2f}). "
                f"{follow_up}"
            )

        if repetition > 0.40:
            detail = "I also noticed repeated wording in the current message."
        elif uncertainty > 0.20:
            detail = "The wording contains some uncertainty."
        elif negativity > 0.25:
            detail = "There is a noticeable negative signal in the wording."
        else:
            detail = "The current signals are within the lower-risk range."

        return (
            f"I picked up {tone} in what you shared. {detail} "
            f"The current risk estimate is {risk_score:.2f}. {follow_up}"
        )

    def run(self, text: str, typing_metrics: dict, history: list, local_hour: int = None, audio_path: str = None) -> dict:
        """
        Executes the full Multimodal (Text + Keystroke + Acoustic) pipeline.
        """
        # Fallback to server time if the frontend doesn't provide the user's local hour
        if local_hour is None:
            local_hour = datetime.datetime.now().hour

        # We start an explanations array early to catch acoustic anomalies
        explanations = []

        # --- STEP 1: Feature Extraction ---
        emo = self.emotion.predict(text)
        
        feats = {
            "negativity": self.features.compute_negativity(emo["scores"], text),
            "positivity": self.features.compute_positivity(emo["scores"]),
            "uncertainty": self.features.compute_uncertainty(text),
            "repetition": self.features.compute_repetition(text),
            "typing_irregularity": self.features.typing_irregularity(typing_metrics)
        }

        # --- STEP 2: Base Masking Detection ---
        is_masking = self.features.detect_masking(feats["positivity"], feats["typing_irregularity"])

        # --- STEP 3: Vocal Telemetry Processing (The Judge's Request) ---
        vocal_strain = 0.0
        if audio_path:
            voice_metrics = self.audio_engine.extract_vocal_telemetry(audio_path)
            vocal_strain = voice_metrics.get("vocal_strain_index", 0.0)
            
            # THE OVERRIDE: If the user's voice is literally shaking, force a masking flag
            if vocal_strain > 0.75 and not is_masking:
                is_masking = True
                explanations.append("Acoustic anomaly: High vocal tremor detected contradicting semantic input.")

        # --- STEP 4: Cognitive Strain Index (CSI) Calibration ---
        raw_csi = self.csi.compute_raw(feats)
        
        # Acoustic Fusion: Blend the typing/text strain with the vocal strain
        if audio_path:
            raw_csi = (0.6 * raw_csi) + (0.4 * vocal_strain)

        z_score = self.csi.compute_z_score(raw_csi, history)

        # --- STEP 5: Temporal Risk & Interpretability ---
        risk_score, risk_explanations = self.risk.compute(
            current_csi=raw_csi, 
            z_score=z_score, 
            is_masking=is_masking, 
            history=history
        )
        explanations.extend(risk_explanations)

        # --- THE FAIL-SAFE: CRISIS LEXICON OVERRIDE ---
        # If the user explicitly states severe distress, bypass all EMA smoothing!
        crisis_keywords = [
            "kill myself", "killing myself", "suicide", "end it all", 
            "want to die", "can't do this anymore", "no reason to live",
            "i give up", "giving up", "no point anymore" # Added your test phrases!
        ]
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in crisis_keywords):
            risk_score = 1.00  # Instantly max out the risk
            explanations.append("CRITICAL: Severe crisis lexicon detected. Bypassing temporal smoothing.")
            is_masking = False # They aren't masking, they are explicitly crying for help

        # --- STEP 6: Intervention & Action Triggers ---
        action_plan = self.intervention.determine_action(
            risk_score=risk_score, 
            is_masking=is_masking, 
            local_hour=local_hour
        )

        # --- STEP 7: AI Clinical Report Generation ---
        current_state = {
            "csi": raw_csi, 
            "risk": risk_score, 
            "is_masking": is_masking,
            "explanation": explanations
        }
        session_report = self.report_gen.generate_summary(history + [current_state])

        # --- STEP 8: The Final Data Payload ---
        return {
            "emotion": emo,
            "features": feats,
            "csi": round(raw_csi, 4),
            "z_score": round(z_score, 4),
            "is_masking": is_masking,
            "risk_score": round(risk_score, 4), 
            "explanation": explanations,
            "trigger_sos": action_plan["trigger_sos"],
            "suggested_intervention": action_plan["suggested_intervention"],
            "session_report_md": session_report,
            "assistant_response": self._build_contextual_response(
                text=text,
                emotion=emo,
                features=feats,
                csi=raw_csi,
                risk_score=risk_score,
                is_masking=is_masking,
                trigger_sos=action_plan["trigger_sos"],
                suggested_intervention=action_plan["suggested_intervention"]
            )
        }