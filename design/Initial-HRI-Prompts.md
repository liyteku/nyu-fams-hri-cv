## Initial HRI Interaction Policy

This document translates the `Initial HRI Prompts.drawio` diagram into a textual policy that can be used as the basis for rule-based behaviour or as part of an LLM system prompt.

### Overall Flow

- **Prompt initiation question**  
  - T1: Ask the learner what they would like to learn about today.  
  - Tx: Begin passively teaching about the chosen topic (baseline teaching behaviour).
- **Detect emotional state**  
  - Continuously estimate the learner's emotional state using the perception pipeline (e.g. facial emotion, gaze/attention).
- **Branch by emotional state**  
  - `Happy / Neutral`
  - `Distracted`
  - `Sad`
  - `Angry`
  - `Irritated`
  - (Optionally treat `Surprised` as `Neutral`)

The system loops between **teaching**, **detecting emotional state**, and **adapting behaviour** according to the rules below.

### Emotional States and System Behaviours

- **Happy / Neutral**
  - Interpretation: learner is engaged and in a positive or stable state.
  - Behaviour:
    - Provide **positive affirmation** about the learner's effort or progress.
    - Continue the current teaching plan.
    - After a small number of affirmations (e.g. ×2), optionally escalate difficulty or introduce new material.

- **Distracted**
  - Interpretation: attention is drifting (e.g. low gaze-on-screen, neutral/bored affect).
  - Behaviour:
    - Attempt to **re-engage** the learner:
      - Use name, ask a simple check-in question, or pose a low-effort interactive prompt.
      - Make content more concrete or multimodal if possible.
    - Repeat re-engagement attempts a limited number of times (e.g. ×3).
    - If re-engagement fails repeatedly, transition towards **Disengage** behaviour.

- **Sad**
  - Interpretation: learner is experiencing negative affect with low arousal.
  - Behaviour:
    - Ask a gentle **“Why sad?”** or equivalent open question if appropriate.
    - Provide **empathic affirmations**:
      - Validate feelings, reduce pressure, offer support.
    - Optionally reduce task difficulty or slow pacing before returning to teaching.

- **Angry**
  - Interpretation: learner is frustrated or upset, high arousal.
  - Behaviour:
    - De-escalate: acknowledge frustration, avoid pushing new content.
    - Offer options such as taking a short break, changing activity, or simplifying the task.
    - If anger persists, the system may move towards **Disengage** or flag for human intervention.

- **Irritated**
  - Interpretation: lighter but persistent negative affect, often a progression from other negative states (e.g. repeated distraction or frustration).
  - Behaviour:
    - Short-term: similar to **Angry** but less intense; focus on reducing friction (clarify instructions, remove unnecessary difficulty).
    - Longer-term: if irritation remains, consider **Disengage** or handoff to a human facilitator.

- **Disengage**
  - Interpretation: learner is no longer constructively engaged despite previous attempts.
  - Behaviour:
    - Politely pause or end the current session.
    - Optionally provide a brief summary of what was covered and invite the learner to return later.
    - In a lab setting, this can trigger a “call researcher” / “call Rasch” style escalation for human follow-up.

### Notes for Implementation

- This document is descriptive and model-agnostic:
  - The **perception layer** provides discrete emotional/engagement labels (e.g. from `EmotionDetector` and `GazeTracker`).
  - A **policy layer** maps those labels, plus interaction history (e.g. number of re-engagement attempts), to the next system action.
- For LLM-based dialogue:
  - These rules can be embedded into the system prompt as “interaction guidelines”, with the current emotional state and history passed in as context for each turn.

