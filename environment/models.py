"""
OpenEnv typed models for the Adaptive Threat Response System.
All three required model types: Observation, Action, Reward.
"""

from __future__ import annotations
from typing import Literal, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Alert types the agent can encounter
# ---------------------------------------------------------------------------
AlertType = Literal[
    "brute_force",
    "malware_detected",
    "data_exfiltration",
    "lateral_movement",
    "privilege_escalation",
    "port_scan",
    "phishing_attempt",
    "anomalous_traffic",
    "insider_threat",
    "ransomware",
]

# ---------------------------------------------------------------------------
# Action types the agent can take
# ---------------------------------------------------------------------------
ActionType = Literal[
    "ignore",        # Do nothing — accept the risk
    "investigate",   # Gather more info — low disruption, slow response
    "block",         # Block the source IP — moderate disruption
    "isolate",       # Isolate the affected system — high disruption
    "escalate",      # Escalate to human analyst — delays action
]

# ---------------------------------------------------------------------------
# Attack stages for multi-stage scenarios (Task 3)
# ---------------------------------------------------------------------------
AttackStage = Literal[
    "none",
    "reconnaissance",
    "initial_access",
    "lateral_movement",
    "exfiltration",
    "full_compromise",
]


# ---------------------------------------------------------------------------
# OBSERVATION — what the agent sees at each step
# ---------------------------------------------------------------------------
class AlertContext(BaseModel):
    """A single past alert in the agent's memory window."""
    alert_id: str
    alert_type: AlertType
    severity: float = Field(ge=0.0, le=1.0)
    action_taken: ActionType | None = None
    step: int


class Observation(BaseModel):
    """
    Everything the agent can see at a given timestep.
    Returned by reset() and step().
    """
    # Current alert to respond to
    alert_id: str = Field(description="Unique ID for this alert")
    alert_type: AlertType = Field(description="Category of the security alert")
    severity: float = Field(ge=0.0, le=1.0, description="Alert severity score")
    source_ip: str = Field(description="IP address generating the alert")
    affected_systems: list[str] = Field(description="Systems involved in the alert")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident the detection system is (not reliability of threat)"
    )

    # World state — agent must balance these
    security_state: float = Field(
        ge=0.0, le=1.0,
        description="Current threat level: 0=safe, 1=full compromise"
    )
    system_health: float = Field(
        ge=0.0, le=1.0,
        description="System availability: 1=fully operational, 0=unusable"
    )

    # Episode metadata
    step_number: int = Field(ge=0, description="Current step in episode")
    max_steps: int = Field(description="Total steps in this episode")
    task_id: str = Field(description="Which task is being evaluated")

    # Memory: last N alerts the agent has already seen + acted on
    context: list[AlertContext] = Field(
        default_factory=list,
        description="Recent alert history for pattern recognition"
    )

    # Attack chain info (visible in medium/hard tasks)
    attack_stage: AttackStage = Field(
        default="none",
        description="Current stage of a multi-stage attack (if detected)"
    )


# ---------------------------------------------------------------------------
# ACTION — what the agent sends back
# ---------------------------------------------------------------------------
class Action(BaseModel):
    """
    The agent's decision in response to an observation.
    Sent to step().
    """
    action_type: ActionType = Field(description="Type of response action")
    target: str = Field(
        description="IP address or system name to act on"
    )
    reasoning: str = Field(
        default="",
        description="Agent's reasoning — used by graders to assess quality"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Agent's confidence in its own decision"
    )


# ---------------------------------------------------------------------------
# REWARD — signal returned after each step
# ---------------------------------------------------------------------------
class RewardBreakdown(BaseModel):
    """Detailed decomposition of the reward signal."""
    security_gain: float = Field(
        description="Reward for correctly addressing a real threat"
    )
    health_penalty: float = Field(
        description="Penalty for disrupting system availability"
    )
    false_positive_penalty: float = Field(
        description="Penalty for acting on a harmless alert"
    )
    missed_threat_penalty: float = Field(
        description="Penalty for ignoring a real threat"
    )
    escalation_delay_penalty: float = Field(
        default=0.0,
        description="Penalty for slow response on critical alerts"
    )


class Reward(BaseModel):
    """
    Full reward signal returned by step().
    value is the scalar reward; breakdown explains the components.
    """
    value: float = Field(
        ge=-1.0, le=1.0,
        description="Scalar reward for this step, clipped to [-1, 1]"
    )
    breakdown: RewardBreakdown
    done: bool = Field(description="Whether the episode has ended")
    info: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra diagnostic info: was_real_threat, optimal_action, etc."
    )


# ---------------------------------------------------------------------------
# STATE — full internal state returned by state()
# ---------------------------------------------------------------------------
class EnvironmentState(BaseModel):
    """
    Full internal state of the environment.
    Returned by state() — used by validators and debuggers.
    Not visible to the agent during normal operation.
    """
    task_id: str
    episode_id: str
    step_number: int
    max_steps: int

    security_state: float
    system_health: float
    attack_stage: AttackStage

    # Ground truth — hidden from agent during step/reset
    active_threats: list[str] = Field(description="IPs with real active threats")
    blocked_ips: list[str] = Field(description="Currently blocked IPs")
    isolated_systems: list[str] = Field(description="Currently isolated systems")

    # Episode tracking
    total_reward: float
    alerts_seen: int
    true_positives: int   # Agent correctly acted on real threats
    false_positives: int  # Agent acted on harmless alerts
    false_negatives: int  # Agent ignored real threats

    # Full alert history with ground truth
    alert_history: list[dict[str, Any]]

    episode_done: bool


# ---------------------------------------------------------------------------
# TASK METADATA — returned by /tasks endpoint
# ---------------------------------------------------------------------------
class TaskInfo(BaseModel):
    """Describes a single task for the /tasks endpoint."""
    task_id: str
    name: str
    description: str
    difficulty: Literal["easy", "medium", "hard"]
    max_steps: int
    action_space: list[ActionType]
    observation_fields: list[str]
    grader_description: str
    expected_score_random_agent: float
    expected_score_optimal_agent: float