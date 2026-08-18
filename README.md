# 🧠 Multimodal Behavioral Observability Engine

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00.svg)
![Status](https://img.shields.io/badge/Status-Hackathon_Ready-success.svg)

> **Built for the IMPACT AI 3.0 Hackathon**
> 
> *Moving beyond standard "AI chatbots" to create a clinical-grade, time-series behavioral triage system.*

## 📖 The Problem
Current mental health AI relies entirely on semantic text analysis. If a user types, *"I'm doing absolutely great!"*, standard LLM wrappers will classify them as "Happy." But what if it took them 45 seconds, 20 backspaces, and erratic keystrokes to type that single sentence? Standard AI misses the "Fake Smile."

## 🚀 Our Solution
We built a **Multimodal Temporal Observability Engine**. It does not try to be an "AI Therapist." Instead, it is a frictionless background triage system that ingests both natural language and sub-conscious physical typing telemetry to infer cognitive strain over time. 

It calculates a personalized **Cognitive Strain Index (CSI)**, measures deviations from the user's established baseline using Z-Scores, and triggers UI interventions before a crisis occurs.

---

## ✨ Core Architecture & Pitchable Features

### 1. 🎭 "Digital Masking" Detection (The Heuristic Engine)
Standard sentiment analysis is brittle. We map Hugging Face text-classification (`j-hartmann/emotion-english-distilroberta-base`) against live keystroke telemetry. If semantic positivity is high but physical typing is highly erratic, the system instantly flags hidden distress (`is_masking: True`).

### 2. 📊 Dynamic Baseline Calibration (Zero-Shot Personalization)
Stress looks different for everyone. Instead of hardcoded risk thresholds, our engine calculates a rolling mean and standard deviation of a user's CSI. The system only escalates risk when a user deviates significantly (e.g., `+2.0 SD`) from *their own* historical norm.

### 3. 📉 Temporal Risk Observability
Mental state is a time-series sequence. We use **Exponential Moving Averages (EMA)** to smooth telemetry noise. This prevents UI whiplash from a single typo while successfully catching slow, steady declines in mental stability over days or weeks.

### 4. 🚑 Automated Micro-Intervention Engine
The ML pipeline outputs direct UI action triggers based on current risk levels:
* **Mild/Moderate Risk:** Triggers contextual grounding UI (e.g., 5-4-3-2-1 technique, Breathing Bubbles).
* **Late-Night Strain:** Applies a 1.25x Risk Multiplier and triggers Sleep Hygiene prompts.
* **Severe Spikes:** Bypasses standard UI and triggers an un-dismissible **SOS Crisis Modal**.

### 5. 🔍 Transparent Explainability (XAI)
To bridge the gap between AI and clinical trust, every inference returns a human-readable `explanations` array detailing exactly *why* the AI made its decision, culminating in a dynamically generated Markdown clinical session report.

---

## ⚙️ System Workflow

1. **Patient Interface (React):** User types normally. A custom hook silently harvests `speed`, `backspaces`, and `latency`.
2. **The API Bridge (FastAPI):** Receives the multimodal payload and fetches the user's recent state history.
3. **The ML Engine (Python):** * Runs NLP extraction (DistilRoBERTa + VADER).
   * Calculates Repetition/Rumination indexes.
   * Fuses text + physical data into a raw CSI.
   * Normalizes CSI into a Z-Score against historical memory.
   * Calculates EMA risk trend.
4. **Clinician Dashboard (React):** Renders real-time telemetry, gauge charts, and anomaly alerts.

---

## 🛠️ Installation & Local Setup

### Prerequisites
* Python 3.9+
* pip

### 1. Clone the repository
```bash
git clone https://github.com/SuhaskumarR/gradient_ai.git
cd gradient_ai
