#!/usr/bin/env python3
"""
NYU-FAMS-HRI-CV — PCA9685 servo handshake (Raspberry Pi + CircuitPython).

Install on the Pi (not in repo ``requirements.txt``)::

    pip install adafruit-circuitpython-pca9685 adafruit-blinka

Run on hardware with I2C enabled and PCA9685 at 0x40. Channel indices match
``src/embodied_policy/pca9685_arm.py`` (live demo: ``python main.py --arm-pca9685`` on Pi).

**Not** for macOS: Blinka does not provide ``board`` / I2C there; run this script on the Pi
(whether or not the PCA9685 is plugged in, macOS will fail at ``import board``).
"""

from __future__ import annotations

import platform
import sys
import time

LEFT_SHOULDER = 0
LEFT_ELBOW = 1
LEFT_HAND = 2
LEFT_WRIST = 3
LEFT_SHOULDER_ROTATE = 7

RIGHT_SHOULDER = 4
RIGHT_ELBOW = 5
RIGHT_HAND = 6


def main() -> None:
    if platform.system() != "Linux":
        print(
            "This script must run on Linux (e.g. Raspberry Pi) with I2C + PCA9685.\n"
            "Adafruit Blinka has no `board` definition on macOS/Windows — the error is\n"
            "not from an unplugged servo board; copy the project to your Pi and run it there."
        )
        sys.exit(2)

    try:
        import board
        import busio
        from adafruit_pca9685 import PCA9685
    except NotImplementedError as e:
        print(
            "Blinka could not identify this machine as a supported single-board computer.\n"
            "Use a Raspberry Pi (or Jetson with Blinka support), enable I2C, then retry."
        )
        print(repr(e))
        sys.exit(2)

    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=0x40)
    pca.frequency = 50

    print("Performing left handshake...")

    pca.channels[LEFT_SHOULDER].duty_cycle = 2730
    pca.channels[LEFT_HAND].duty_cycle = 3640
    pca.channels[LEFT_WRIST].duty_cycle = 3276
    pca.channels[LEFT_SHOULDER_ROTATE].duty_cycle = 1638

    time.sleep(1.5)

    pca.channels[LEFT_SHOULDER].duty_cycle = 1638
    pca.channels[LEFT_HAND].duty_cycle = 1638
    pca.channels[LEFT_WRIST].duty_cycle = 1638
    pca.channels[LEFT_SHOULDER_ROTATE].duty_cycle = 1638

    print("Left done")

    time.sleep(1)

    print("Performing right handshake...")

    pca.channels[RIGHT_SHOULDER].duty_cycle = 3276
    pca.channels[RIGHT_ELBOW].duty_cycle = 2457
    pca.channels[RIGHT_HAND].duty_cycle = 2000

    time.sleep(1.5)

    pca.channels[RIGHT_SHOULDER].duty_cycle = 1638
    pca.channels[RIGHT_ELBOW].duty_cycle = 1638
    pca.channels[RIGHT_HAND].duty_cycle = 1638

    print("Right done")

    pca.channels[LEFT_SHOULDER].duty_cycle = 0
    pca.channels[LEFT_ELBOW].duty_cycle = 0
    pca.channels[LEFT_HAND].duty_cycle = 0
    pca.channels[LEFT_WRIST].duty_cycle = 0
    pca.channels[LEFT_SHOULDER_ROTATE].duty_cycle = 0
    pca.channels[RIGHT_SHOULDER].duty_cycle = 0
    pca.channels[RIGHT_ELBOW].duty_cycle = 0
    pca.channels[RIGHT_HAND].duty_cycle = 0

    try:
        i2c.deinit()
    except Exception:
        pass


if __name__ == "__main__":
    main()
