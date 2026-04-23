# Hardware (chassis + upper body)

Firmware and standalone scripts for the embodied stack. **Policy → actuation** lives in Python (`src/embodied_policy/emotion_to_position.py`, `pca9685_arm.py`); this folder holds what runs **on the microcontrollers / Pi**.

## Quick reference (from repo root)

| Goal | Where | Command / action |
|------|--------|-------------------|
| Live demo **policy only** (default) | Laptop or Pi | `python3 main.py` |
| List USB serial ports (Teensy, etc.) | Laptop or Pi | `python3 main.py --list-serials` |
| Live demo + **chassis** (Teensy) | Mac or Pi (USB works) | `python3 main.py --serial /dev/cu.usbmodemXXXX` |
| Live demo + **arms** (PCA9685) | **Raspberry Pi** (I2C) | `python3 main.py --arm-pca9685` |
| Chassis + arms on one machine | Pi with USB + I2C | `python3 main.py --serial /dev/ttyACM0 --arm-pca9685` |

Install Python deps for the camera demo from the project root: `pip install -r requirements.txt` (includes `pyserial` for `--serial`). PCA9685 on the Pi additionally needs:

```bash
pip install adafruit-circuitpython-pca9685 adafruit-blinka
```

---

## Lower body: Teensy + CAN (`teensy_can_basic_movement/`)

### What it does

- USB serial at **115200** baud, newline-terminated commands: `FORWARD`, `BACK`, `LEFT`, `RIGHT`, `STOP` (case-sensitive).
- Sends **CAN 1 Mbps** frames on **CAN1**, extended ID off, **ID `0x200`**, 8 bytes: two `int16` motor currents (A, B), big-endian high byte first.
- Default drive mapping in the sketch: **`SPEED` 2500**; `FORWARD` / `BACK` use **opposite** signs on m1 vs m2; `LEFT` / `RIGHT` use same sign on both (tank-style). Adjust signs in `handleCommand` if your wiring or desired “forward” differs.

### Setup

1. Install the **FlexCAN_T4** library in Arduino / Teensyduino: [FlexCAN_T4](https://github.com/tonton81/FlexCAN_T4).
2. Open `hardware/teensy_can_basic_movement/teensy_can_basic_movement.ino`, select your **Teensy** board and port, upload.

### Linking to `main.py`

1. Plug in the Teensy over USB.
2. Discover the port: `python3 main.py --list-serials`.
3. On **macOS**, prefer **`/dev/cu.usbmodem…`** (not `tty`) for Python.
4. Run:

```bash
python3 main.py --serial /dev/cu.usbmodemXXXX
```

Optional baud override (default matches sketch): `--serial-baud 115200`.

When perception resolves to angry / irritated, the demo sends **`BACK`**; when leaving that chassis command it sends **`STOP`** once. Duplicate commands are not spammed every frame.

---

## Upper body: PCA9685 (`rpi_pca9685_handshake.py` + Python driver)

### Channel map

Aligned with `src/embodied_policy/pca9685_arm.py` and the handshake demo:

| Channel | Joint |
|--------|--------|
| 0 | Left shoulder |
| 1 | Left elbow |
| 2 | Left hand |
| 3 | Left wrist |
| 7 | Left shoulder rotate |
| 4 | Right shoulder |
| 5 | Right elbow |
| 6 | Right hand |

### Standalone handshake (Raspberry Pi)

Requires **Linux** (e.g. Pi), I2C enabled, PCA9685 at **0x40**, 50 Hz servo PWM. On macOS the script exits early with a short message (Blinka has no `board` on Darwin).

```bash
cd /path/to/NYU-FAMS-HRI-CV
pip install adafruit-circuitpython-pca9685 adafruit-blinka
python3 hardware/rpi_pca9685_handshake.py
```

### Live demo with policy-driven arm rest (Raspberry Pi)

From project root, with camera + model paths as in the main README:

```bash
python3 main.py --arm-pca9685
```

Optional I2C address (default `0x40`):

```bash
python3 main.py --arm-pca9685 --pca9685-address 0x40
```

**Note:** `--arm-pca9685` uses **Blinka** (`board`, `busio`). It is intended for the **Pi** (or similar); a typical Mac does not expose the PCA9685 on the same bus without extra USB–I2C hardware and Blinka configuration.

Tune rest pose duty cycles in `src/embodied_policy/pca9685_arm.py` (`REST_DUTY_BY_CHANNEL`).

---

## Architecture (short)

```
Webcam → EmotionDetector / GazeTracker → PolicyEngine
              → instruction (overlay)
              → actuation_from_perception
                    → serial lines → Teensy → CAN motors
                    → arm_xyz_inches → PCA9685UpperBody (Pi only)
```

For design intent of the policy layer, see `design/Initial-HRI-Prompts.md`.
