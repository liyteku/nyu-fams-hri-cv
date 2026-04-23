"""
Emotion / policy branch → robot actuation intent (arm pose or chassis command).

Maps fused policy output to targets understood by your hardware sketches:

- **Teensy + FlexCAN** (``hardware/teensy_can_basic_movement``): serial lines
  ``FORWARD``, ``BACK``, ``LEFT``, ``RIGHT``, ``STOP`` (newline-terminated).
- **Arm rest** ``(0, 0, -27)``: inches from shoulder origin (placeholder; tune
  with real link lengths / calibration).
- **PCA9685 upper body** (``pca9685_arm.py`` + ``main.py --arm-pca9685``): maps
  ``arm_xyz_inches`` to servo duty presets on Raspberry Pi. Demo script:
  ``hardware/rpi_pca9685_handshake.py``.

Prefer ``actuation_from_policy_context`` so branch resolution matches
``policy_engine``; use ``emotion_to_actuation`` only for quick tests with a
raw emotion string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Tuple

from .policy_engine import (
    PolicyBranch,
    PolicyContext,
    PolicyDecision,
    PolicyEngine,
    policy_context_from_perception,
)

# Serial tokens consumed by ``hardware/teensy_can_basic_movement``.
ChassisCommand = Literal["FORWARD", "BACK", "LEFT", "RIGHT", "STOP"]

# Shoulder-frame target position in inches (see module docstring).
ArmTargetInches = Tuple[float, float, float]

# Default “arms folded / rest” when affect is not happy-neutral (tune on hardware).
DEFAULT_ARM_REST_INCHES: ArmTargetInches = (0.0, 0.0, -27.0)


@dataclass(frozen=True)
class ActuationTarget:
    """
    At most one primary intent per evaluation (chassis wins when both set).

    Empty instance: no change / hold current pose on the robot side.
    """

    chassis: Optional[ChassisCommand] = None
    arm_xyz_inches: Optional[ArmTargetInches] = None


def branch_to_actuation(branch: PolicyBranch) -> ActuationTarget:
    """
    Map a resolved policy branch to chassis or arm targets.

    Angry / irritated → chassis ``BACK`` (retreat). Other non–happy-neutral
    branches → arm rest pose. ``HAPPY_NEUTRAL`` → no actuation hint.
    """
    if branch in (PolicyBranch.ANGRY, PolicyBranch.IRRITATED):
        return ActuationTarget(chassis="BACK")
    if branch == PolicyBranch.HAPPY_NEUTRAL:
        return ActuationTarget()
    return ActuationTarget(arm_xyz_inches=DEFAULT_ARM_REST_INCHES)


def actuation_from_policy_decision(decision: PolicyDecision) -> ActuationTarget:
    return branch_to_actuation(decision.branch)


def actuation_from_policy_context(
    ctx: PolicyContext,
    *,
    policy_engine: Optional[PolicyEngine] = None,
) -> ActuationTarget:
    """Resolve ``ctx`` with ``PolicyEngine.evaluate`` then map to hardware intent."""
    eng = policy_engine or PolicyEngine()
    return actuation_from_policy_decision(eng.evaluate(ctx))


def actuation_from_perception(
    emotion_data: Optional[Mapping[str, object]] = None,
    gaze_data: Optional[Mapping[str, object]] = None,
    *,
    policy_engine: Optional[PolicyEngine] = None,
    primary_actions: Optional[Mapping[str, bool]] = None,
    reengage_attempts: int = 0,
    anger_or_irritation_streak: int = 0,
    min_emotion_confidence: Optional[float] = 40.0,
) -> ActuationTarget:
    """
    Convenience: build ``PolicyContext`` like ``instruction_from_perception``,
    then return ``ActuationTarget``.

    When ``emotion_data`` is present and ``min_emotion_confidence`` is not None,
    returns an empty target if ``confidence`` is below the threshold (same gate
    as ``instruction_from_perception``). Omit gate with ``min_emotion_confidence=None``.
    """
    if emotion_data is None and gaze_data is None:
        return ActuationTarget()
    if emotion_data is not None and min_emotion_confidence is not None:
        conf = float(emotion_data.get("confidence", 0))
        if conf < min_emotion_confidence:
            return ActuationTarget()
    ctx = policy_context_from_perception(
        emotion_data,
        gaze_data,
        primary_actions=primary_actions,
        reengage_attempts=reengage_attempts,
        anger_or_irritation_streak=anger_or_irritation_streak,
    )
    return actuation_from_policy_context(ctx, policy_engine=policy_engine)


def _norm_emotion(s: Optional[str]) -> str:
    if not s:
        return ""
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def emotion_to_actuation(
    emotion: str,
    *,
    learning_state: Optional[str] = None,
    policy_engine: Optional[PolicyEngine] = None,
) -> ActuationTarget:
    """
    Map a single emotion label (and optional ``learning_state``) to actuation.

    Uses the same ``PolicyEngine`` resolution as the live pipeline so results
    stay consistent with tutor instructions.
    """
    ls = learning_state.strip().lower() if isinstance(learning_state, str) else None
    em = _norm_emotion(emotion)
    ctx = PolicyContext(learning_state=ls, smoothed_emotion=em or None)
    return actuation_from_policy_context(ctx, policy_engine=policy_engine)


def chassis_command_line(cmd: ChassisCommand) -> str:
    """One serial line for the Teensy sketch (newline-terminated)."""
    return f"{cmd}\n"


def actuation_to_serial_lines(target: ActuationTarget) -> list[str]:
    """
    Serial commands to send for this target (chassis only; arm is not serial
    in the bundled Teensy demo). Extend when arm IK streams over another bus.
    """
    if target.chassis is not None:
        return [chassis_command_line(target.chassis)]
    return []
