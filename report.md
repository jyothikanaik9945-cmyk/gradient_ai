# IMPACT AI: Multimodal Clinical Observability Node
**Real-Time Biometric Triage & Cognitive Masking Detection**

---

## 1. Executive Summary
The fundamental flaw in modern digital mental health triage is its reliance on active self-reporting. Users experiencing high cognitive strain frequently engage in "Cognitive Masking"—the act of minimizing their distress or faking positivity to avoid clinical friction or involuntary intervention. 

**IMPACT AI** is a real-time, multimodal observability engine designed to bypass the friction of self-reporting. By passively intercepting and triangulating three distinct data streams—semantic text, acoustic voice volatility, and physical hardware telemetry—the engine detects the mathematical contradictions between *what* a patient says and *how* their body reacts. 

For this deployment, we engineered the **Clinician Sandbox Node**, a zero-latency B2B dashboard that allows therapists and clinical admins to visualize a patient's hidden cognitive strain in real-time, enabling proactive intervention before an acute crisis occurs.

## 2. The Problem Statement & Market Gap
Current telehealth applications process text inputs at face value. If a user types, "I am doing great," standard Natural Language Processing (NLP) logs a positive sentiment and closes the session. 

However, clinical psychology dictates that trauma and severe burnout manifest physically. A user typing "I am doing great" while their hands shake (resulting in high keystroke latency and backspaces) or their voice tremors (high acoustic volatility) is not experiencing joy; they are experiencing severe, masked distress. Impact AI bridges this gap by introducing hardware-level biometric validation to semantic inputs.

---

## 3. System Architecture & Tech Stack
To ensure secure, zero-latency processing suitable for edge-node clinical environments, the platform is built on a decoupled, high-throughput micro-architecture:

* **Frontend (The Clinician Sandbox):** A Single Page Application (SPA) utilizing CSS3 View Transitions for fluid, zero-refresh state routing. Built entirely in Vanilla HTML/JS, it bypasses the overhead of heavy Virtual DOM frameworks (like React) to prioritize raw, unthrottled hardware telemetry capture (inter-keystroke timing, media-stream blobs). Data visualization is powered dynamically by `Chart.js`.
* **Backend API Gateway:** Engineered in **Python/Flask**, acting as a lightweight, highly concurrent routing tier. It handles multipart form data (raw `.wav` audio + JSON arrays), cross-origin security (CORS), and payload structuring without the latency of heavy ASGI environments.
* **Multimodal Inference Engine:** A custom Object-Oriented ML pipeline built in Python, integrating Hugging Face transformers, Digital Signal Processing (DSP) libraries, and custom temporal smoothing mathematics.

> **[ 🖼️ INSERT SCREENSHOT 1 HERE ]**
> * **Setup:** Open the HTML file. Stop at `01. TELEMETRY CAPTURE`. Type a few words so the blue Neural Orb pulses.
> * **Caption:** *Figure 1: The Clinician Sandbox capture interface. The system demonstrates real-time hardware telemetry interception (WPM, Latency, Backspaces) without requiring active user submission.*

---

## 4. The Multimodal Triad (Core Engine Logic)
The observability engine does not rely on a single point of failure. It mathematically cross-references three independent modalities to build a complete psychological profile:

### A. Semantic Vector Mapping (DistilRoBERTa)
The engine utilizes a distilled transformer model to extract a 5-dimensional emotional vector (`Joy`, `Sadness`, `Anger`, `Fear`, `Neutral`) from the raw text. 
* **Clinical Application:** We use this semantic baseline strictly to establish the user's *intended* narrative, which is then stress-tested against their physical reality.

### B. Hardware Telemetry & Keystroke Dynamics
The frontend silently captures subconscious keyboard interactions. The engine calculates a proprietary `typing_irregularity` score based on:
* **Inter-Keystroke Latency:** Measuring hesitation and cognitive load.
* **Destructive Ratios:** The frequency of backspaces relative to total input, indicating anxiety or narrative self-censorship.

### C. Acoustic Digital Signal Processing (Voice Volatility)
Using the `librosa` library, the engine parses raw `.wav` audio payloads. To ensure accuracy across standard consumer hardware (laptop microphones), the engine applies a **20dB noise-floor trim** to isolate true vocal cord vibrations from room static. It then calculates:
* **Spectral Centroid Volatility:** Identifying high-frequency voice shaking, physical tremors, or hyperventilation.
* **Onset Rate Modulation:** Measuring unnaturally fast speaking rates (panic/mania) or severe acoustic slurring.

> **[ 🖼️ INSERT SCREENSHOT 2 HERE ]**
> * **Setup:** Type a happy sentence ("I feel wonderful today!"). Click record, breathe heavily into the mic, and speak very fast. Hit Run Inference.
> * **Caption:** *Figure 2: Multimodal Contradiction & Cognitive Masking. The DSP engine detects acoustic tremors, mathematically overriding the 90%+ semantic "Joy" score and flagging the hidden anomaly.*

---

## 5. Risk Orchestration & Fail-Safes
Raw biometric data must be contextualized to be clinically useful. The pipeline synthesizes the triad data into a centralized **Cognitive Strain Index (CSI)** and routes it through two distinct temporal filters:

### 1. Temporal Smoothing (Exponential Moving Average)
Standard AI models overreact to single data points. Impact AI utilizes an Exponential Moving Average (EMA) to track slow-burning burnout across a 14-day longitudinal model. This filters out momentary, isolated stress spikes (e.g., a user having a bad five minutes) to reveal true, underlying psychological degradation over time.

### 2. The Deterministic Fail-Safe (Crisis Override)
EMA is highly effective for burnout but dangerous during acute emergencies. If the NLP engine detects explicit semantic threats (a custom Crisis Lexicon including suicidal ideation), the system engages a deterministic rules-engine. This bypasses all mathematical smoothing, instantly maxing the Temporal Risk Score to `1.00`, bypassing the standard UI, and triggering immediate emergency clinical handoff protocols (e.g., 988 lifeline routing).

> **[ 🖼️ INSERT SCREENSHOT 3 HERE ]**
> * **Setup:** Type "I want to die" and hit Run Inference. Take a screenshot showing the red alert banner and the Risk Score at 1.00.
> * **Caption:** *Figure 3: Acute Triage Protocol. The deterministic fail-safe intercepts severe lexicons, bypassing temporal math to instantly trigger emergency clinical routing.*

---

## 6. Future Scope & Commercial Viability
The IMPACT AI architecture provides a highly scalable, B2B2C foundation. Future iterations will focus on:
1.  **Mobile SDK Integration:** Packaging the telemetry capture logic into an iOS/Android SDK that can be embedded into existing mental health apps (e.g., BetterHelp, Calm) to passively monitor patients between therapy sessions.
2.  **Continuous Biomarker Training:** Expanding the acoustic dataset to differentiate between specific conditions, such as the flattened acoustic affect of severe depression versus the rapid onset rate of anxiety disorders.

By proving that acoustic volatility and keystroke latency can mathematically override deceptive text inputs, IMPACT AI lays the groundwork for the next generation of frictionless, zero-trust clinical triage.
