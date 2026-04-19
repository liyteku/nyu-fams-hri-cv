"""
Emotion (or affect) → robot pose mapping (planned).

Intended pipeline (not implemented yet):
  - Input: discrete emotion / affect label, or optionally a :class:`PolicyBranch`
    from ``src.embodied_policy.policy_engine`` after perception is fused.
  - Output: target poses for manipulator arm(s) and mobile chassis (joint angles,
    base twist, etc.) for an embodied tutor or social robot.

This module is the **actuation / kinematics intent** layer: it turns affect into
*where the hardware should move*. It does **not** decide dialogue or teaching
strategy—that is ``PolicyEngine``'s job (tutor instructions, LLM hints).

Relationship to ``policy_engine.py``:
  - **No overlap** if this file only maps *already-resolved* policy outputs
    (e.g. ``PolicyDecision.branch`` or a small enum) → joint/base commands.
  - **Overlap** only if you duplicate the same emotion→branch rules here; avoid
    that by calling ``PolicyEngine.evaluate`` / ``instruction_from_perception``
    first, then map ``PolicyBranch`` → positions in *this* file.

# Input = Emotion (or PolicyBranch) / optional continuous scores
# Output = Position of arm and chassis (joint targets, pose, etc.)
"""
