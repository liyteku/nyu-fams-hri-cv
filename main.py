#!/usr/bin/env python3
"""
Live demo: webcam → EmotionDetector + GazeTracker → **policy prompt** (default).

**Default:** perception → ``instruction_from_perception`` only (tutor text on the
overlay). No lower or upper body I/O unless you opt in.

**Optional lower body (Teensy):** pass ``--serial PORT`` → same perception drives
``actuation_from_perception`` → serial (``BACK``, ``STOP``, …) to
``hardware/teensy_can_basic_movement``.

**Optional upper body (Pi + PCA9685):** ``--arm-pca9685`` (Blinka). See
``src/embodied_policy/pca9685_arm.py``.

Run from project root:
    python3 main.py
    python3 main.py --camera 1
    python3 main.py --list-serials
    python3 main.py --serial /dev/cu.usbmodem1101
    python3 main.py --arm-pca9685

Does not load StudentFaceRecognition (see README).
"""

import argparse
import os
import platform
import sys
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2

from src.face import EmotionDetector, GazeTracker
from src.embodied_policy import (
    PolicyEngine,
    actuation_from_perception,
    actuation_to_serial_lines,
    chassis_command_line,
    instruction_from_perception,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="HRI-CV live demo (emotion + gaze + policy)")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument(
        "--print-every",
        type=int,
        default=15,
        help="Print policy instruction to terminal every N frames (0=disable)",
    )
    parser.add_argument(
        "--serial",
        metavar="PORT",
        default=None,
        help=(
            "Teensy serial port for lower body (default: off — policy overlay only). "
            "macOS: prefer /dev/cu.usbmodem*; Windows: COM5."
        ),
    )
    parser.add_argument(
        "--list-serials",
        action="store_true",
        help="List detected serial ports and exit (no camera / model required).",
    )
    parser.add_argument(
        "--serial-baud",
        type=int,
        default=115200,
        help="Baud rate for --serial (default 115200, matches Teensy sketch).",
    )
    parser.add_argument(
        "--arm-pca9685",
        action="store_true",
        help="Upper body on (default: off). PCA9685 over I2C on Raspberry Pi + Blinka only.",
    )
    parser.add_argument(
        "--pca9685-address",
        type=lambda s: int(s, 0),
        default=0x40,
        metavar="ADDR",
        help="PCA9685 I2C address (default 0x40).",
    )
    args = parser.parse_args()

    if args.list_serials:
        _print_serial_ports()
        return

    detector = EmotionDetector()
    if not detector.is_available:
        print("EmotionDetector: no model loaded (place best_model.pth under outputs/ or project root).")
        sys.exit(1)

    tracker = GazeTracker()
    policy_engine = PolicyEngine()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}")
        sys.exit(1)

    ser = None
    if args.serial:
        try:
            import serial
        except ImportError:
            print("Install pyserial for --serial: pip install pyserial")
            sys.exit(1)
        try:
            ser, opened_as = _open_serial_with_macos_fallback(serial, args.serial, args.serial_baud)
            if opened_as != args.serial:
                print(f"Note: opened {opened_as} (macOS cu/ fallback from {args.serial})")
            print(f"Serial open: {opened_as} @ {args.serial_baud}")
        except OSError as e:
            print(f"Could not open serial {args.serial}: {e}")
            print("Plug in the Teensy/USB cable, then pick a port from:")
            _print_serial_ports()
            print("On macOS, use /dev/cu.usbmodem* with Python (tty may be wrong or busy).")
            sys.exit(1)

    upper_body = None
    if args.arm_pca9685:
        try:
            from src.embodied_policy.pca9685_arm import PCA9685UpperBody

            upper_body = PCA9685UpperBody(i2c_address=args.pca9685_address)
            print(f"PCA9685 upper body @ 0x{args.pca9685_address:02X}")
        except ImportError as e:
            print(
                "PCA9685: install on the Pi: "
                "pip install adafruit-circuitpython-pca9685 adafruit-blinka"
            )
            print(repr(e))
            sys.exit(1)
        except Exception as e:
            print(f"Could not open PCA9685 / I2C: {e}")
            sys.exit(1)

    hw_out = []
    if ser is not None:
        hw_out.append("lower body (serial)")
    if upper_body is not None:
        hw_out.append("upper body (PCA9685)")
    if hw_out:
        print("Running with " + " + ".join(hw_out) + ". Press 'q' to quit.")
    else:
        print("Running (policy / prompt only; no robot hardware). Press 'q' to quit.")
    frame_i = 0
    last_chassis_sent: Optional[str] = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            emotion_data = detector.detect_emotion(frame, enforce_detection=False)
            gaze_data = tracker.detect_gaze(frame)

            frame, y_after_emotion = detector.draw_emotion_overlay(frame, emotion_data)
            frame, y_after_gaze = tracker.draw_gaze_overlay(
                frame, gaze_data, start_y=y_after_emotion
            )

            instruction = instruction_from_perception(
                emotion_data,
                gaze_data,
                policy_engine=policy_engine,
            )
            # Actuation only when at least one hardware path is enabled (default: skip).
            actuation = None
            if ser is not None or upper_body is not None:
                actuation = actuation_from_perception(
                    emotion_data,
                    gaze_data,
                    policy_engine=policy_engine,
                )

            if ser is not None and actuation is not None:
                ch = actuation.chassis
                if ch is not None:
                    if ch != last_chassis_sent:
                        for line in actuation_to_serial_lines(actuation):
                            ser.write(line.encode("ascii"))
                        ser.flush()
                        last_chassis_sent = ch
                elif last_chassis_sent is not None:
                    ser.write(chassis_command_line("STOP").encode("ascii"))
                    ser.flush()
                    last_chassis_sent = None

            if upper_body is not None and actuation is not None:
                upper_body.apply(actuation)

            y = max(y_after_gaze, 28)
            if instruction:
                for line in _wrap_text(instruction, max_chars=72):
                    cv2.putText(
                        frame,
                        line,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (220, 255, 220),
                        1,
                        lineType=cv2.LINE_AA,
                    )
                    y += 18

            if args.print_every and instruction and frame_i % args.print_every == 0:
                print(instruction)

            cv2.imshow("NYU FAMS HRI-CV", frame)
            frame_i += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if upper_body is not None:
            upper_body.release()
        if ser is not None and ser.is_open:
            ser.close()
        cap.release()
        cv2.destroyAllWindows()


def _print_serial_ports() -> None:
    try:
        from serial.tools import list_ports
    except ImportError:
        print("Install pyserial: pip install pyserial")
        sys.exit(1)
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports reported by the OS.")
        return
    for p in ports:
        desc = (p.description or "").replace("\n", " ")
        print(f"  {p.device}\t{desc}")


def _open_serial_with_macos_fallback(serial_mod: object, port: str, baud: int):
    """
    Open pyserial Serial. On macOS, if /dev/tty.usbmodem* fails, try /dev/cu.usbmodem*
    (call-out device; recommended for non-terminal programs).
    """
    Serial = serial_mod.Serial
    try:
        return Serial(port, baud, timeout=0), port
    except OSError:
        if platform.system() == "Darwin" and port.startswith("/dev/tty."):
            alt = "/dev/cu." + port[len("/dev/tty.") :]
            if alt != port:
                return Serial(alt, baud, timeout=0), alt
        raise


def _wrap_text(text: str, max_chars: int = 72) -> list:
    """Split long policy text into lines that fit on screen."""
    words = text.split()
    lines: list = []
    cur: list = []
    n = 0
    for w in words:
        if n + len(w) + (1 if cur else 0) > max_chars and cur:
            lines.append(" ".join(cur))
            cur = [w]
            n = len(w)
        else:
            cur.append(w)
            n += len(w) + (1 if len(cur) > 1 else 0)
    if cur:
        lines.append(" ".join(cur))
    return lines[:6]


if __name__ == "__main__":
    main()
