"""
Upper-body servos on PCA9685 (50 Hz), for Raspberry Pi + CircuitPython Blinka.

Maps ``ActuationTarget.arm_xyz_inches`` from ``emotion_to_position`` to **preset
duty cycles** (no IK yet): any non-``None`` arm target uses the same “rest /
folded” pose until you add joint-space targets.

Install on the Pi (not in root ``requirements.txt``)::

    pip install adafruit-circuitpython-pca9685 adafruit-blinka

Channel indices match ``hardware/rpi_pca9685_handshake.py``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from .emotion_to_position import ActuationTarget

# PCA9685 channel indices (same as hardware/rpi_pca9685_handshake.py).
LEFT_SHOULDER = 0
LEFT_ELBOW = 1
LEFT_HAND = 2
LEFT_WRIST = 3
LEFT_SHOULDER_ROTATE = 7
RIGHT_SHOULDER = 4
RIGHT_ELBOW = 5
RIGHT_HAND = 6

# Neutral-ish duty (≈1.5 ms pulse at 50 Hz on 16-bit scaling); tune per rig.
_NEUT = 1638

# Preset for policy “arm rest” / folded upper body (tune per mechanical build).
REST_DUTY_BY_CHANNEL: Dict[int, int] = {
    LEFT_SHOULDER: _NEUT,
    LEFT_ELBOW: _NEUT,
    LEFT_HAND: _NEUT,
    LEFT_WRIST: _NEUT,
    LEFT_SHOULDER_ROTATE: _NEUT,
    RIGHT_SHOULDER: _NEUT,
    RIGHT_ELBOW: _NEUT,
    RIGHT_HAND: _NEUT,
}


def _arm_close(a: Tuple[float, float, float], b: Tuple[float, float, float], tol: float = 1e-3) -> bool:
    return all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b))


class PCA9685UpperBody:
    """
    Apply ``ActuationTarget`` arm fields to a PCA9685 on the default board I2C bus.
    """

    def __init__(self, *, i2c_address: int = 0x40, frequency_hz: int = 50) -> None:
        import board
        import busio
        from adafruit_pca9685 import PCA9685

        self._i2c = busio.I2C(board.SCL, board.SDA)
        self._pca = PCA9685(self._i2c, address=i2c_address)
        self._pca.frequency = frequency_hz
        self._last_arm: Optional[Tuple[float, float, float]] = None

    def apply(self, target: ActuationTarget) -> None:
        """If ``target.arm_xyz_inches`` is set, drive rest preset (debounced)."""
        arm = target.arm_xyz_inches
        if arm is None:
            return
        if self._last_arm is not None and _arm_close(arm, self._last_arm):
            return
        for ch, duty in REST_DUTY_BY_CHANNEL.items():
            self._pca.channels[ch].duty_cycle = int(duty)
        self._last_arm = arm

    def release(self) -> None:
        """PWM off on used channels; then release I2C."""
        for ch in REST_DUTY_BY_CHANNEL:
            try:
                self._pca.channels[ch].duty_cycle = 0
            except Exception:
                pass
        try:
            self._i2c.deinit()
        except Exception:
            pass
