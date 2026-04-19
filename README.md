# NYU FAMS HRI-CV

Real-time multimodal perception (face emotion, gaze) plus a rule-based **policy layer** for the NYU FAMS Lab, designed for deployment on Nvidia Jetson devices.

## Quick run (webcam demo)

From the project root, with dependencies installed (`pip install -r requirements.txt`) and a trained emotion checkpoint at `outputs/best_model.pth` (or `best_model.pth` in the project root):

```bash
python main.py
```

`main.py` runs **EmotionDetector** + **GazeTracker** and shows **PolicyEngine** tutor instructions on the overlay (see `design/Initial-HRI-Prompts.md`). **StudentFaceRecognition** is not used in this entry point.

Teaching / attention hints elsewhere in the code go through the same policy: `EmotionDetector.get_teaching_recommendation(...)` and `GazeTracker.get_attention_recommendation(...)` call `instruction_from_perception` (optional fused `gaze_data` / `emotion_data`). Robot pose mapping (planned) lives in `src/embodied_policy/emotion_to_position.py`.

## Layout

| Path | Role |
|------|------|
| `src/face/` | Emotion detection, gaze tracking, (optional) face ID |
| `src/body/` | YOLO pose + rule-based body cues |
| `src/embodied_policy/` | `PolicyEngine`, `policy_context_from_perception`, `instruction_from_perception`, `emotion_to_position` (stub) |
| `training/` | Train emotion (RAF-DB) and skeleton action (NTU) models |
| `main.py` | Live camera demo (emotion + gaze + policy) |
