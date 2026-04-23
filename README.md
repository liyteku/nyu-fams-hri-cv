# NYU FAMS HRI-CV

Real-time multimodal perception (face emotion, gaze) plus a rule-based **policy layer** for the NYU FAMS Lab. Suitable for laptops (e.g. Mac) for vision + policy, and for **Raspberry Pi** when driving **PCA9685** upper body over I2C. Jetson-class devices are also a reasonable deployment target.

## Quick run (webcam demo)

From the project root, with dependencies installed (`pip install -r requirements.txt`) and a trained emotion checkpoint at `outputs/best_model.pth` (or `best_model.pth` in the project root):

```bash
python3 main.py
```

By default this is **emotion / gaze → policy prompt only** (overlay text). Lower body (``--serial``) and upper body (``--arm-pca9685``) stay **off** until you pass those flags.

Optional flags (see also **`hardware/README.md`**):

```bash
python3 main.py --list-serials
python3 main.py --serial /dev/cu.usbmodemXXXX
python3 main.py --arm-pca9685
```

`main.py` runs **EmotionDetector** + **GazeTracker**, draws non-overlapping overlays, and shows **PolicyEngine** tutor instructions (see `design/Initial-HRI-Prompts.md`). **StudentFaceRecognition** is not used in this entry point.

Teaching / attention hints elsewhere use the same policy: `EmotionDetector.get_teaching_recommendation(...)` and `GazeTracker.get_attention_recommendation(...)` call `instruction_from_perception` with optional fused `gaze_data` / `emotion_data`.

**Embodied actuation (opt-in):** `main.py` calls `actuation_from_perception` **only** when `--serial` and/or `--arm-pca9685` is set, then maps to `ActuationTarget` (Teensy tokens, PCA9685 presets). Otherwise the repo still exposes `actuation_from_perception` for your own scripts. See `src/embodied_policy/emotion_to_position.py`, `pca9685_arm.py`, and `hardware/`.

## Layout

| Path | Role |
|------|------|
| `src/face/` | Emotion detection, gaze tracking, (optional) face ID |
| `src/body/` | YOLO pose + rule-based body cues |
| `src/embodied_policy/` | `PolicyEngine`, `instruction_from_perception`, `policy_context_from_perception`, `emotion_to_position` (actuation + serial helpers), `pca9685_arm` (Pi PCA9685) |
| `hardware/` | Teensy CAN sketch, Pi PCA9685 handshake script, **README** for wiring and CLI |
| `training/` | Train emotion (RAF-DB) and skeleton action (NTU) models |
| `main.py` | Live camera demo (emotion + gaze + policy + optional serial / PCA9685) |

## Pipeline (conceptual)

```
Webcam → EmotionDetector / GazeTracker
              → instruction_from_perception → tutor text (overlay)   [always in main.py]
              → actuation_from_perception → serial / PCA9685        [only if --serial / --arm-pca9685]
```

`PolicyEngine.evaluate` does **not** branch on body pose yet; `PolicyContext.primary_actions` is reserved for future rules.
