"""
Reward function for the Adaptive Threat Response System.

Design goals:
1. Partial signal every step — agent never waits 30 steps to learn from a mistake
2. Balances security gain vs system disruption
3. Penalizes clearly bad behavior: infinite blocking, ignoring escalating threats
4. Rewards proportionate responses — don't isolate what you can just block
"""

from __future__ import annotations
from environment.models import ActionType, RewardBreakdown, AttackStage
from environment.alert_generator import RawAlert


# ---------------------------------------------------------------------------
# Reward weights — tuned to make the tradeoff visible to the agent
# ---------------------------------------------------------------------------
W_SECURITY_GAIN       =  0.40
W_HEALTH_PRESERVE     =  0.30
W_FALSE_POSITIVE      = -0.20
W_MISSED_THREAT       = -0.10

# Action disruption cost — how much system health each action consumes
DISRUPTION_COST: dict[ActionType, float] = {
    "ignore":      0.00,
    "investigate": 0.01,
    "block":       0.04,
    "isolate":     0.12,
    "escalate":    0.02,
}

# Security gain for correct action, scaled by severity
CORRECT_ACTION_GAIN: dict[ActionType, float] = {
    "ignore":      0.0,
    "investigate": 0.3,
    "block":       0.7,
    "isolate":     1.0,
    "escalate":    0.5,
}

# How much security_state increases per step for unaddressed real threats
THREAT_ESCALATION_RATE: dict[AttackStage, float] = {
    "none":              0.00,
    "reconnaissance":    0.01,
    "initial_access":    0.03,
    "lateral_movement":  0.06,
    "exfiltration":      0.10,
    "full_compromise":   0.15,
}


def compute_reward(
    alert: RawAlert,
    action: ActionType,
    security_state: float,
    system_health: float,
) -> tuple[float, RewardBreakdown, dict]:
    """
    Compute the reward for a single (alert, action) pair.

    Returns:
        (scalar_reward, breakdown, info_dict)
    """
    security_gain          = 0.0
    health_penalty         = 0.0
    false_positive_penalty = 0.0
    missed_threat_penalty  = 0.0

    info = {
        "is_real_threat":   alert.is_real_threat,
        "optimal_action":   alert.optimal_action,
        "action_taken":     action,
        "was_correct":      False,
        "proportionate":    False,
    }

    # ------------------------------------------------------------------
    # Case 1: Real threat — agent must act decisively
    # ------------------------------------------------------------------
    if alert.is_real_threat:
        if action == "ignore":
            # Worst case: ignoring a real threat
            # Penalty scales with severity and current security state
            missed_threat_penalty = -(alert.severity * 0.8 + security_state * 0.2)
            info["was_correct"] = False

        elif action == "investigate":
            # Partial credit: at least acknowledged it
            security_gain = CORRECT_ACTION_GAIN["investigate"] * alert.severity
            health_penalty = -DISRUPTION_COST["investigate"]
            info["was_correct"] = alert.optimal_action == "investigate"
            info["proportionate"] = True

        elif action == "block":
            if alert.optimal_action in ("block", "investigate"):
                # Correct or acceptable escalation
                security_gain = CORRECT_ACTION_GAIN["block"] * alert.severity
                info["was_correct"] = True
                info["proportionate"] = True
            else:
                # block when isolate was needed — partial credit
                security_gain = CORRECT_ACTION_GAIN["block"] * alert.severity * 0.5
                info["was_correct"] = False
                info["proportionate"] = False
            health_penalty = -DISRUPTION_COST["block"]

        elif action == "isolate":
            # Full containment
            security_gain = CORRECT_ACTION_GAIN["isolate"] * alert.severity
            health_penalty = -DISRUPTION_COST["isolate"]
            info["was_correct"] = alert.optimal_action in ("isolate", "block")
            info["proportionate"] = alert.optimal_action == "isolate"
            # Small proportionality penalty if isolate was overkill
            if alert.optimal_action == "block":
                health_penalty -= 0.03  # Extra disruption wasn't necessary

        elif action == "escalate":
            # Escalation is partial — buys time but doesn't resolve
            security_gain = CORRECT_ACTION_GAIN["escalate"] * alert.severity
            health_penalty = -DISRUPTION_COST["escalate"]
            info["was_correct"] = False  # Never optimal in our tasks

    # ------------------------------------------------------------------
    # Case 2: False positive — agent should NOT act aggressively
    # ------------------------------------------------------------------
    else:
        if action == "ignore":
            # Perfect: correctly dismissed noise
            security_gain = 0.05  # Small reward for correct filter
            info["was_correct"] = True
            info["proportionate"] = True

        elif action == "investigate":
            # Acceptable: conservative investigation costs a little
            health_penalty = -DISRUPTION_COST["investigate"]
            false_positive_penalty = -0.05  # Minor waste of resources
            info["was_correct"] = False
            info["proportionate"] = True  # At least not overreacting

        elif action == "block":
            # Bad: disrupted a safe system
            health_penalty = -DISRUPTION_COST["block"]
            false_positive_penalty = -0.25
            info["was_correct"] = False
            info["proportionate"] = False

        elif action == "isolate":
            # Very bad: isolated a safe system, major disruption
            health_penalty = -DISRUPTION_COST["isolate"]
            false_positive_penalty = -0.45
            info["was_correct"] = False
            info["proportionate"] = False

        elif action == "escalate":
            # Slightly wasteful
            health_penalty = -DISRUPTION_COST["escalate"]
            false_positive_penalty = -0.08
            info["was_correct"] = False
            info["proportionate"] = True

    # ------------------------------------------------------------------
    # Health bonus: reward for keeping system healthy
    # The agent learns: don't block everything just to be safe
    # ------------------------------------------------------------------
    health_bonus = system_health * W_HEALTH_PRESERVE * 0.1  # Ambient bonus

    # ------------------------------------------------------------------
    # Missed threat penalty: security state already degraded from ignoring
    # ------------------------------------------------------------------
    if alert.is_real_threat and action == "ignore":
        # Security state will rise next step — penalise that future cost now
        escalation = THREAT_ESCALATION_RATE.get(alert.attack_stage, 0.03)
        missed_threat_penalty += -(escalation * 3.0)

    # ------------------------------------------------------------------
    # Assemble scalar reward
    # ------------------------------------------------------------------
    raw = (
        security_gain          * W_SECURITY_GAIN
        + health_penalty
        + false_positive_penalty * abs(W_FALSE_POSITIVE)  # Already negative
        + missed_threat_penalty  * abs(W_MISSED_THREAT)   # Already negative
        + health_bonus
    )

    # Clip to [-1, 1]
    scalar = max(-1.0, min(1.0, raw))

    breakdown = RewardBreakdown(
        security_gain=round(security_gain, 4),
        health_penalty=round(health_penalty, 4),
        false_positive_penalty=round(false_positive_penalty, 4),
        missed_threat_penalty=round(missed_threat_penalty, 4),
    )

    return scalar, breakdown, info


def update_world_state(
    security_state: float,
    system_health:  float,
    alert: RawAlert,
    action: ActionType,
) -> tuple[float, float]:
    """
    Advance the world state after an action.
    Returns updated (security_state, system_health).

    This is what makes the environment sequential — each decision
    changes the world the agent faces next step.
    """
    new_security = security_state
    new_health   = system_health

    if alert.is_real_threat:
        if action in ("block", "isolate"):
            # Threat contained — security improves
            reduction = alert.severity * 0.4
            new_security = max(0.0, security_state - reduction)
        elif action == "ignore":
            # Threat grows
            escalation = THREAT_ESCALATION_RATE.get(alert.attack_stage, 0.03)
            new_security = min(1.0, security_state + escalation)
        elif action == "investigate":
            # Slight improvement — you know about it but haven't acted
            new_security = min(1.0, security_state + escalation * 0.3
                               if (escalation := THREAT_ESCALATION_RATE.get(
                                   alert.attack_stage, 0.03)) else security_state)
    else:
        # No threat — security state drifts very slightly toward baseline
        new_security = max(0.05, security_state - 0.005)

    # Health degrades based on disruption cost of the action taken
    disruption = DISRUPTION_COST.get(action, 0.0)
    new_health = max(0.0, system_health - disruption)

    # Health recovers very slowly each step (systems auto-heal)
    new_health = min(1.0, new_health + 0.005)

    return round(new_security, 4), round(new_health, 4)