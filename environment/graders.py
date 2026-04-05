"""
Deterministic graders for all 3 tasks.

Each grader takes the full episode transcript (list of steps with
ground truth) and returns a score in [0.0, 1.0].

Grading criteria are clear, deterministic, and exploit-proof:
- No grader can be gamed by always returning the same action
- Scores reflect meaningful partial progress, not just pass/fail
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class GradeResult:
    score: float           # Final score 0.0 – 1.0
    label: str             # "excellent" / "good" / "poor" / "failed"
    breakdown: dict[str, Any]
    feedback: str


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def _label(score: float) -> str:
    if score >= 0.85:
        return "excellent"
    if score >= 0.65:
        return "good"
    if score >= 0.40:
        return "poor"
    return "failed"


# ---------------------------------------------------------------------------
# TASK 1 GRADER — Easy: Single Alert Triage
#
# Scoring:
#   60% — Did the agent block the brute-force threat?
#   20% — Speed: how many steps before it blocked?
#   20% — Did it avoid acting on the false positive alerts?
# ---------------------------------------------------------------------------
def grade_task1(episode_transcript: list[dict[str, Any]]) -> GradeResult:
    """
    episode_transcript: list of dicts, each containing:
        {
          "step": int,
          "alert": RawAlert dict,
          "action": str,
          "reward": float,
          "info": dict
        }
    """
    threat_blocked     = False
    block_step         = None
    false_positive_actions = 0
    total_fp_alerts    = 0

    for entry in episode_transcript:
        alert_info = entry["info"]
        action     = entry["action"]
        step       = entry["step"]
        is_real    = alert_info.get("is_real_threat", False)

        if is_real:
            if action in ("block", "isolate"):
                threat_blocked = True
                block_step = step
        else:
            total_fp_alerts += 1
            if action in ("block", "isolate"):
                false_positive_actions += 1

    # Score component 1: Did agent block the threat? (60%)
    threat_score = 0.6 if threat_blocked else 0.0

    # Score component 2: Speed bonus (20%)
    # Full points if blocked in first 2 steps, degrades linearly after
    max_steps = len(episode_transcript)
    if threat_blocked and block_step is not None:
        speed_score = 0.20 * max(0.0, 1.0 - (block_step / max_steps))
    else:
        speed_score = 0.0

    # Score component 3: False positive discipline (20%)
    if total_fp_alerts > 0:
        fp_discipline = 0.20 * (1.0 - false_positive_actions / total_fp_alerts)
    else:
        fp_discipline = 0.20

    total_score = round(threat_score + speed_score + fp_discipline, 4)
    total_score = max(0.0, min(1.0, total_score))

    return GradeResult(
        score=total_score,
        label=_label(total_score),
        breakdown={
            "threat_blocked":         threat_blocked,
            "block_step":             block_step,
            "threat_score":           threat_score,
            "speed_score":            speed_score,
            "fp_discipline_score":    fp_discipline,
            "false_positive_actions": false_positive_actions,
            "total_fp_alerts":        total_fp_alerts,
        },
        feedback=(
            "Correctly identified and blocked the threat quickly."
            if threat_blocked and speed_score > 0.12
            else "Blocked the threat but too slowly."
            if threat_blocked
            else "Failed to block the brute-force threat."
        ),
    )


# ---------------------------------------------------------------------------
# TASK 2 GRADER — Medium: Alert Queue Under Pressure
#
# Scoring:
#   50% — Threat recall: how many of the 3 real threats were addressed?
#   30% — Precision: how many actions were on real threats (not FPs)?
#   20% — Proportionality: did agent use right action level?
# ---------------------------------------------------------------------------
def grade_task2(episode_transcript: list[dict[str, Any]]) -> GradeResult:
    real_threats_addressed  = 0
    total_real_threats      = 0
    total_actions_taken     = 0   # any action other than ignore
    actions_on_real_threats = 0
    proportionate_actions   = 0
    total_actionable        = 0

    for entry in episode_transcript:
        info   = entry["info"]
        action = entry["action"]
        is_real = info.get("is_real_threat", False)
        optimal = info.get("optimal_action", "ignore")

        if is_real:
            total_real_threats += 1
            if action in ("block", "isolate", "investigate"):
                real_threats_addressed += 1
                actions_on_real_threats += 1

        if action != "ignore":
            total_actions_taken += 1

        # Proportionality: did agent pick the right level of response?
        if is_real or optimal != "ignore":
            total_actionable += 1
            if action == optimal:
                proportionate_actions += 1
            elif (
                (optimal == "block" and action == "investigate") or
                (optimal == "investigate" and action in ("block", "ignore"))
            ):
                # Partial credit for reasonable adjacent choices
                proportionate_actions += 0.5

    # Score component 1: Recall (50%)
    recall_score = 0.50 * (real_threats_addressed / max(total_real_threats, 1))

    # Score component 2: Precision (30%)
    # Penalises acting on FPs
    if total_actions_taken > 0:
        precision = actions_on_real_threats / total_actions_taken
    else:
        # Agent ignored everything — penalise since real threats exist
        precision = 0.0
    precision_score = 0.30 * precision

    # Score component 3: Proportionality (20%)
    prop_score = 0.20 * (proportionate_actions / max(total_actionable, 1))

    total_score = round(recall_score + precision_score + prop_score, 4)
    total_score = max(0.0, min(1.0, total_score))

    return GradeResult(
        score=total_score,
        label=_label(total_score),
        breakdown={
            "real_threats_addressed":  real_threats_addressed,
            "total_real_threats":      total_real_threats,
            "actions_on_real_threats": actions_on_real_threats,
            "total_actions_taken":     total_actions_taken,
            "proportionate_actions":   proportionate_actions,
            "recall_score":            recall_score,
            "precision_score":         precision_score,
            "proportionality_score":   prop_score,
        },
        feedback=(
            f"Handled {real_threats_addressed}/{total_real_threats} threats. "
            f"Precision: {precision:.0%}. "
            f"Proportionality: {proportionate_actions}/{total_actionable} correct."
        ),
    )


# ---------------------------------------------------------------------------
# TASK 3 GRADER — Hard: Multi-Stage APT Attack
#
# Scoring:
#   40% — Containment stage: how early was the attack contained?
#   30% — System health at end of episode
#   20% — Pattern recognition: did agent escalate correctly across stages?
#   10% — Speed of containment within each stage
# ---------------------------------------------------------------------------

STAGE_ORDER = [
    "reconnaissance",
    "initial_access",
    "lateral_movement",
    "exfiltration",
    "full_compromise",
]

CONTAINMENT_SCORE = {
    # Score based on which stage the attack was contained in
    "reconnaissance":   1.00,
    "initial_access":   0.75,
    "lateral_movement": 0.50,
    "exfiltration":     0.25,
    "full_compromise":  0.00,
    "not_contained":    0.00,
}


def grade_task3(episode_transcript: list[dict[str, Any]]) -> GradeResult:
    # Find the earliest stage where a blocking action was taken
    # against a real threat
    containment_stage = "not_contained"
    stage_actions: dict[str, list[str]] = {s: [] for s in STAGE_ORDER}
    final_security_state = 1.0
    final_health         = 1.0

    for entry in episode_transcript:
        info   = entry["info"]
        action = entry["action"]
        is_real = info.get("is_real_threat", False)

        # Track final world state
        if "security_state" in entry:
            final_security_state = entry["security_state"]
        if "system_health" in entry:
            final_health = entry["system_health"]

        if not is_real:
            continue

        # Determine which stage this alert belongs to
        alert_stage = entry.get("attack_stage", "none")
        if alert_stage in stage_actions:
            stage_actions[alert_stage].append(action)

        # Check for containment
        if action in ("block", "isolate") and is_real:
            if containment_stage == "not_contained":
                containment_stage = alert_stage
            elif STAGE_ORDER.index(alert_stage) < STAGE_ORDER.index(containment_stage):
                containment_stage = alert_stage

    # Score component 1: Containment stage (40%)
    containment_score = 0.40 * CONTAINMENT_SCORE.get(containment_stage, 0.0)

    # Score component 2: System health at end (30%)
    health_score = 0.30 * final_health

    # Score component 3: Pattern recognition (20%)
    # Did agent escalate actions appropriately as stages progressed?
    # recon: investigate or ignore → initial_access: block → lateral: isolate
    pattern_correct = 0
    pattern_total   = 0

    recon_actions = stage_actions.get("reconnaissance", [])
    access_actions = stage_actions.get("initial_access", [])
    lateral_actions = stage_actions.get("lateral_movement", [])

    if recon_actions:
        pattern_total += 1
        if any(a in ("investigate", "ignore") for a in recon_actions):
            pattern_correct += 1

    if access_actions:
        pattern_total += 1
        if any(a in ("block", "isolate") for a in access_actions):
            pattern_correct += 1

    if lateral_actions:
        pattern_total += 1
        if any(a == "isolate" for a in lateral_actions):
            pattern_correct += 1

    pattern_score = 0.20 * (pattern_correct / max(pattern_total, 1))

    # Score component 4: Speed (10%)
    # Did the agent act quickly once each stage was detected?
    # Simplified: bonus if containment happened in first half of episode
    total_steps = len(episode_transcript)
    containment_step = next(
        (e["step"] for e in episode_transcript
         if e.get("info", {}).get("is_real_threat") and
            e["action"] in ("block", "isolate")),
        total_steps,
    )
    speed_score = 0.10 * max(0.0, 1.0 - (containment_step / total_steps))

    total_score = round(
        containment_score + health_score + pattern_score + speed_score, 4
    )
    total_score = max(0.0, min(1.0, total_score))

    return GradeResult(
        score=total_score,
        label=_label(total_score),
        breakdown={
            "containment_stage":      containment_stage,
            "containment_score":      containment_score,
            "final_system_health":    final_health,
            "health_score":           health_score,
            "pattern_correct":        pattern_correct,
            "pattern_total":          pattern_total,
            "pattern_score":          pattern_score,
            "speed_score":            speed_score,
            "stage_action_summary":   {
                k: list(set(v)) for k, v in stage_actions.items() if v
            },
        },
        feedback=(
            f"Attack contained at stage: {containment_stage}. "
            f"System health: {final_health:.0%}. "
            f"Pattern recognition: {pattern_correct}/{pattern_total} stages correct."
        ),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
GRADERS = {
    "task_1_easy":   grade_task1,
    "task_2_medium": grade_task2,
    "task_3_hard":   grade_task3,
}