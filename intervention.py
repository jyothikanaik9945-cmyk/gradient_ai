class InterventionEngine:
    def __init__(self):
        # Thresholds for triggering UI interventions
        self.CRITICAL_THRESHOLD = 0.70
        self.MODERATE_THRESHOLD = 0.60
        self.MILD_THRESHOLD = 0.40

    def determine_action(self, risk_score: float, is_masking: bool, local_hour: int) -> dict:
        trigger_sos = False
        suggested_intervention = None

        # 1. Check for Critical Risk / SOS
        if risk_score >= self.CRITICAL_THRESHOLD:
            trigger_sos = True
            suggested_intervention = "CRISIS_HOTLINE_MODAL"
            
        # 2. Check for Masking (High priority intervention)
        elif is_masking:
            suggested_intervention = "GROUNDING_EXERCISE_54321"
            
        # 3. Check for Moderate Risk
        elif risk_score >= self.MODERATE_THRESHOLD:
            # If it's late at night, suggest a sleep-focused intervention
            if local_hour <= 5 or local_hour >= 23:
                suggested_intervention = "SLEEP_HYGIENE_PROMPT"
            else:
                suggested_intervention = "BREATHING_BUBBLE_UI"
                
        # 4. Check for Mild Strain
        elif risk_score >= self.MILD_THRESHOLD:
            suggested_intervention = "MINDFUL_CHECKIN"

        return {
            "trigger_sos": trigger_sos,
            "suggested_intervention": suggested_intervention
        }