"""
Policy engine (rule-based)

Aligned with design/Initial-HRI-Prompts.md. Maps affective branches and optional
perception context to tutor instructions for dialogue or LLM system prompts.

Perception correspondence (see ``policy_context_from_perception``):

- **EmotionDetector** dict keys used: ``learning_state``, ``smoothed_emotion``
  (fallback ``dominant_emotion``). Canonical emotions: angry, disgust, fear,
  happy, sad, surprise, neutral. Learning states: confused, frustrated, neutral
  (from ``_get_learning_state``).
- **GazeTracker** dict: ``smoothed_looking_at_screen`` → ``gaze_distracted`` is
  ``not`` that value when gaze data is present.
- **ActionRecognizer** ``primary_actions`` (hand_raised, standing, …) is stored
  on ``PolicyContext`` for downstream use; ``PolicyEngine.evaluate`` does not
  branch on body yet.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional


class PolicyBranch(str, Enum):
    """High-level branches from Initial-HRI-Prompts.md (+ confused from perception)."""

    HAPPY_NEUTRAL = "happy_neutral"
    DISTRACTED = "distracted"
    SAD = "sad"
    ANGRY = "angry"
    IRRITATED = "irritated"
    CONFUSED = "confused"
    DISENGAGE = "disengage"


# Short tutor-facing instructions (usable as LLM hints or TTS scripts).
_DEFAULT_INSTRUCTIONS: Dict[PolicyBranch, str] = {
    PolicyBranch.HAPPY_NEUTRAL: (
        "Continue the current teaching plan; give brief positive affirmation "
        "about effort or progress; after a couple of affirmations, consider "
        "slightly harder material or a new angle."
    ),
    PolicyBranch.DISTRACTED: (
        "Re-engage the learner: use their name if known, ask a simple check-in, "
        "or a low-effort interactive prompt; make the next step concrete or "
        "multimodal if possible."
    ),
    PolicyBranch.SAD: (
        "Ask a gentle open question if appropriate; respond with empathic "
        "affirmations, validate feelings, reduce pressure; consider easier "
        "pacing or simpler tasks before returning to the main plan."
    ),
    PolicyBranch.ANGRY: (
        "De-escalate: acknowledge frustration; avoid pushing new content; offer "
        "a short break, a different activity, or a simpler step; flag human help "
        "if anger persists."
    ),
    PolicyBranch.IRRITATED: (
        "Reduce friction: clarify instructions, remove unnecessary difficulty, "
        "keep tone calm—similar to angry but lighter; escalate or disengage if "
        "irritation continues."
    ),
    PolicyBranch.CONFUSED: (
        "Explain more slowly; rephrase the idea; use a simpler example or analogy; "
        "check understanding with one easy question before moving on."
    ),
    PolicyBranch.DISENGAGE: (
        "Politely pause or end the session; give a brief summary of what was "
        "covered; invite the learner to return later; in the lab, consider "
        "calling a researcher or facilitator."
    ),
}


@dataclass
class PolicyContext:
    """
    Optional rich input from EmotionDetector / GazeTracker + session history.

    Face (``EmotionDetector`` output dict):

    - ``learning_state``: ``confused`` | ``frustrated`` | ``neutral`` only.
    - ``smoothed_emotion``: same strings as ``EmotionDetector.EMOTIONS`` / RAF map
      (e.g. ``happy``, ``sad``, ``angry``, …). Pass this together with
      ``learning_state`` so ``frustrated`` can mean sad vs angry vs disgust
      (policy resolves by emotion first).

    Gaze (``GazeTracker.detect_gaze`` dict):

    - Set ``gaze_distracted`` to ``not gaze_data["smoothed_looking_at_screen"]``
      when you have a frame result; leave ``None`` if gaze is unavailable.

    Body (``ActionRecognizer.analyse``):

    - ``primary_actions``: optional copy of ``result["primary_actions"]``
      (``hand_raised``, ``standing``, ``leaning_left``, ``leaning_right``).
      Not used inside ``evaluate`` yet; attach for logging or future rules.
    """

    learning_state: Optional[str] = None
    smoothed_emotion: Optional[str] = None
    gaze_distracted: Optional[bool] = None
    primary_actions: Optional[Dict[str, bool]] = None
    reengage_attempts: int = 0
    anger_or_irritation_streak: int = 0


@dataclass(frozen=True)
class PolicyDecision:
    """Structured policy output."""

    instruction: str
    branch: PolicyBranch
    disengage_recommended: bool = False
    human_handoff_recommended: bool = False


class PolicyEngine:
    """
    Maps policy branches (and fused perception) to tutor instructions.

    Simple usage::

        engine = PolicyEngine()
        engine.decide("confused")   # -> str
        engine.evaluate(PolicyContext(...))  # -> PolicyDecision
    """

    def __init__(
        self,
        *,
        max_reengage_attempts: int = 3,
        irritation_threshold_for_handoff: int = 4,
        instructions: Optional[Dict[PolicyBranch, str]] = None,
    ) -> None:
        self.max_reengage_attempts = max(1, int(max_reengage_attempts))
        self.irritation_threshold_for_handoff = max(1, int(irritation_threshold_for_handoff))
        self._instructions: Dict[PolicyBranch, str] = dict(_DEFAULT_INSTRUCTIONS)
        if instructions:
            self._instructions.update(instructions)

    def instruction_for(self, branch: PolicyBranch) -> str:
        return self._instructions[branch]

    def decide(self, state: str) -> str:
        """
        Map a coarse state name to a tutor instruction string.

        Accepted ``state`` values (case-insensitive; spaces/underscores normalized):
        confused, distracted, frustrated, neutral, happy, happy_neutral,
        sad, angry, irritated, disengage, plus aliases (e.g. happiness -> happy_neutral).
        """
        branch = _coerce_simple_state_to_branch(state)
        return self._instructions[branch]

    def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        """
        Fuse perception + history into a branch and instruction.

        Resolution order in ``_resolve_branch_from_context``:

        ``smoothed_emotion`` (angry / sad / disgust) → ``learning_state``
        ``confused`` → ``learning_state`` ``frustrated`` (only if emotion did
        not already pick a branch) → ``gaze_distracted`` → remaining emotions
        (surprise / fear / happy / neutral).

        If the resolved branch is ``distracted`` but re-engagement attempts have
        reached ``max_reengage_attempts``, transition to **disengage** (per design doc).
        """
        branch = _resolve_branch_from_context(ctx)

        if branch == PolicyBranch.DISTRACTED and ctx.reengage_attempts >= self.max_reengage_attempts:
            return self._decision(PolicyBranch.DISENGAGE, disengage_recommended=True)

        human = (
            ctx.anger_or_irritation_streak >= self.irritation_threshold_for_handoff
            and branch in (PolicyBranch.ANGRY, PolicyBranch.IRRITATED)
        )
        if human:
            return self._decision(
                branch,
                human_handoff_recommended=True,
            )

        return self._decision(branch)

    def _decision(
        self,
        branch: PolicyBranch,
        *,
        disengage_recommended: bool = False,
        human_handoff_recommended: bool = False,
    ) -> PolicyDecision:
        return PolicyDecision(
            instruction=self._instructions[branch],
            branch=branch,
            disengage_recommended=disengage_recommended,
            human_handoff_recommended=human_handoff_recommended,
        )


def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def _coerce_simple_state_to_branch(state: str) -> PolicyBranch:
    key = _norm(state)
    aliases = {
        "happy": PolicyBranch.HAPPY_NEUTRAL,
        "happiness": PolicyBranch.HAPPY_NEUTRAL,
        "happy_neutral": PolicyBranch.HAPPY_NEUTRAL,
        "neutral": PolicyBranch.HAPPY_NEUTRAL,
        "engaged": PolicyBranch.HAPPY_NEUTRAL,
        "distracted": PolicyBranch.DISTRACTED,
        "bored": PolicyBranch.DISTRACTED,
        "sad": PolicyBranch.SAD,
        "sadness": PolicyBranch.SAD,
        "angry": PolicyBranch.ANGRY,
        "anger": PolicyBranch.ANGRY,
        "frustrated": PolicyBranch.ANGRY,
        "frustration": PolicyBranch.ANGRY,
        "irritated": PolicyBranch.IRRITATED,
        "irritation": PolicyBranch.IRRITATED,
        "disgust": PolicyBranch.IRRITATED,
        "confused": PolicyBranch.CONFUSED,
        "disengage": PolicyBranch.DISENGAGE,
        "quit": PolicyBranch.DISENGAGE,
    }
    if key in aliases:
        return aliases[key]
    # Default unknown → neutral teaching path
    return PolicyBranch.HAPPY_NEUTRAL


def policy_context_from_perception(
    emotion_data: Optional[Mapping[str, Any]] = None,
    gaze_data: Optional[Mapping[str, Any]] = None,
    *,
    primary_actions: Optional[Mapping[str, bool]] = None,
    reengage_attempts: int = 0,
    anger_or_irritation_streak: int = 0,
) -> PolicyContext:
    """
    Build ``PolicyContext`` from ``EmotionDetector`` / ``GazeTracker`` / body outputs.

    - ``emotion_data``: pass through ``detect_emotion`` result (or None).
    - ``gaze_data``: pass through ``detect_gaze`` result; if None, ``gaze_distracted``
      stays None (unknown), not treated as distracted.
    - ``primary_actions``: pass ``ActionRecognizer.analyse(...)["primary_actions"]``
      if desired.
    """
    ls: Optional[str] = None
    em: Optional[str] = None
    if emotion_data is not None:
        raw_ls = emotion_data.get("learning_state")
        if isinstance(raw_ls, str):
            ls = raw_ls.strip().lower()
        em = emotion_data.get("smoothed_emotion") or emotion_data.get("dominant_emotion")
        if isinstance(em, str):
            em = em.strip().lower()

    gaze_distracted: Optional[bool] = None
    if gaze_data is not None:
        gaze_distracted = not bool(gaze_data.get("smoothed_looking_at_screen", False))

    pa: Optional[Dict[str, bool]] = None
    if primary_actions is not None:
        pa = {k: bool(v) for k, v in dict(primary_actions).items()}

    return PolicyContext(
        learning_state=ls,
        smoothed_emotion=em,
        gaze_distracted=gaze_distracted,
        primary_actions=pa,
        reengage_attempts=reengage_attempts,
        anger_or_irritation_streak=anger_or_irritation_streak,
    )


def instruction_from_perception(
    emotion_data: Optional[Mapping[str, Any]] = None,
    gaze_data: Optional[Mapping[str, Any]] = None,
    *,
    policy_engine: Optional[PolicyEngine] = None,
    min_emotion_confidence: Optional[float] = 40.0,
    **ctx_kwargs: Any,
) -> str:
    """
    Single tutor instruction string from fused perception (``PolicyEngine.evaluate``).

    If ``emotion_data`` is present and ``min_emotion_confidence`` is not None,
    returns ``""`` when ``confidence`` is below that threshold (matches the old
    emotion-only recommendation gate).

    When only ``gaze_data`` is set, emotion confidence is not applied.
    """
    if emotion_data is None and gaze_data is None:
        return ""
    if emotion_data is not None and min_emotion_confidence is not None:
        conf = float(emotion_data.get("confidence", 0))
        if conf < min_emotion_confidence:
            return ""
    eng = policy_engine or PolicyEngine()
    ctx = policy_context_from_perception(emotion_data, gaze_data, **ctx_kwargs)
    return eng.evaluate(ctx).instruction


def _resolve_branch_from_context(ctx: PolicyContext) -> PolicyBranch:
    ls = _norm(ctx.learning_state)
    em = _norm(ctx.smoothed_emotion)

    # Strong negative valence first
    if em in ("angry", "anger"):
        return PolicyBranch.ANGRY
    if em in ("sad", "sadness"):
        return PolicyBranch.SAD
    if em in ("disgust",):
        return PolicyBranch.IRRITATED

    # Learning-centric signal from EmotionDetector (surprise/fear → confused ls first)
    if ls == "confused":
        return PolicyBranch.CONFUSED
    # ``frustrated`` ls lumps angry/disgust/sad; those emotions are handled above.
    if ls == "frustrated":
        return PolicyBranch.ANGRY

    # Attention (design: distracted = low gaze / drifting attention)
    if ctx.gaze_distracted is True:
        return PolicyBranch.DISTRACTED

    # Rare path: emotion set without learning_state (e.g. tests); aligns with EMOTIONS
    if em in ("surprise", "fear"):
        return PolicyBranch.CONFUSED

    if em in ("happy", "happiness", "neutral", ""):
        return PolicyBranch.HAPPY_NEUTRAL

    return PolicyBranch.HAPPY_NEUTRAL
