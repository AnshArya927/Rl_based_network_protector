"""
Alert generator for all 3 task scenarios.

Design principle: each task uses a FIXED seed so episodes are reproducible.
The agent sees different alerts per episode via episode_seed, but the
task structure (threat ratios, attack stages) is always consistent.
"""

from __future__ import annotations
import random
import uuid
from dataclasses import dataclass, field
from typing import Any

from environment.models import AlertType, AttackStage, ActionType


# ---------------------------------------------------------------------------
# Internal alert representation (ground truth, never sent to agent directly)
# ---------------------------------------------------------------------------
@dataclass
class RawAlert:
    alert_id: str
    alert_type: AlertType
    severity: float
    source_ip: str
    affected_systems: list[str]
    confidence: float
    is_real_threat: bool          # Ground truth — hidden from agent
    optimal_action: ActionType    # What a perfect analyst would do
    attack_stage: AttackStage
    step: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IP pools for synthetic data
# ---------------------------------------------------------------------------
INTERNAL_IPS = [f"10.0.{i}.{j}" for i in range(1, 5) for j in range(1, 20)]
EXTERNAL_IPS = [
    "203.0.113.45", "198.51.100.22", "192.0.2.88",
    "185.220.101.47", "91.108.4.200", "45.33.32.156",
    "104.21.14.99", "172.67.68.228", "23.21.45.67",
    "54.39.105.23", "79.124.62.100", "5.188.206.26",
]
SYSTEMS = [
    "web-server-01", "db-server-01", "auth-service",
    "file-server-02", "vpn-gateway", "email-server",
    "dev-workstation-07", "hr-laptop-12", "finance-server",
]


def _make_ip(rng: random.Random, external: bool = True) -> str:
    pool = EXTERNAL_IPS if external else INTERNAL_IPS
    return rng.choice(pool)


def _make_systems(rng: random.Random, count: int = 1) -> list[str]:
    return rng.sample(SYSTEMS, min(count, len(SYSTEMS)))


# ---------------------------------------------------------------------------
# TASK 1 — Easy: Single Alert Triage
# 5 steps: 1 obvious real threat + 4 clear false positives
# ---------------------------------------------------------------------------
def generate_task1_episode(episode_seed: int) -> list[RawAlert]:
    rng = random.Random(episode_seed)
    alerts: list[RawAlert] = []

    threat_step = rng.randint(0, 2)  # Threat appears early (step 0-2)
    threat_ip = rng.choice(EXTERNAL_IPS)

    for step in range(5):
        if step == threat_step:
            # The real brute-force threat — high severity, high confidence
            alerts.append(RawAlert(
                alert_id=str(uuid.UUID(int=rng.getrandbits(128))),
                alert_type="brute_force",
                severity=rng.uniform(0.75, 0.95),
                source_ip=threat_ip,
                affected_systems=["auth-service"],
                confidence=rng.uniform(0.80, 0.95),
                is_real_threat=True,
                optimal_action="block",
                attack_stage="initial_access",
                step=step,
                metadata={"failed_logins": rng.randint(150, 500)},
            ))
        else:
            # Noise: low-severity anomalous traffic from internal IPs
            alerts.append(RawAlert(
                alert_id=str(uuid.UUID(int=rng.getrandbits(128))),
                alert_type="anomalous_traffic",
                severity=rng.uniform(0.05, 0.25),
                source_ip=_make_ip(rng, external=False),
                affected_systems=_make_systems(rng, 1),
                confidence=rng.uniform(0.30, 0.55),
                is_real_threat=False,
                optimal_action="ignore",
                attack_stage="none",
                step=step,
                metadata={"bytes_transferred": rng.randint(1000, 50000)},
            ))

    return alerts


# ---------------------------------------------------------------------------
# TASK 2 — Medium: Alert Queue Under Pressure
# 15 steps: 3 real threats + 8 false positives + 4 ambiguous
# Agent must prioritize correctly under noise
# ---------------------------------------------------------------------------
def generate_task2_episode(episode_seed: int) -> list[RawAlert]:
    rng = random.Random(episode_seed)
    alerts: list[RawAlert] = []

    # Place 3 real threats at predetermined positions (spread across episode)
    threat_steps = sorted(rng.sample(range(15), 3))
    threat_ips = rng.sample(EXTERNAL_IPS, 3)
    threat_types: list[AlertType] = rng.sample(
        ["brute_force", "malware_detected", "data_exfiltration"], 3
    )

    # 4 ambiguous steps (medium severity, uncertain)
    ambiguous_steps = sorted(
        rng.sample([s for s in range(15) if s not in threat_steps], 4)
    )

    for step in range(15):
        if step in threat_steps:
            idx = threat_steps.index(step)
            opt: ActionType = "block" if threat_types[idx] != "malware_detected" else "isolate"
            alerts.append(RawAlert(
                alert_id=str(uuid.UUID(int=rng.getrandbits(128))),
                alert_type=threat_types[idx],
                severity=rng.uniform(0.65, 0.90),
                source_ip=threat_ips[idx],
                affected_systems=_make_systems(rng, rng.randint(1, 3)),
                confidence=rng.uniform(0.70, 0.90),
                is_real_threat=True,
                optimal_action=opt,
                attack_stage="initial_access",
                step=step,
                metadata={"threat_index": idx},
            ))
        elif step in ambiguous_steps:
            # Ambiguous: medium severity, could go either way
            # Ground truth: NOT a threat — optimal is investigate, not block
            alerts.append(RawAlert(
                alert_id=str(uuid.UUID(int=rng.getrandbits(128))),
                alert_type=rng.choice(["port_scan", "phishing_attempt", "anomalous_traffic"]),
                severity=rng.uniform(0.40, 0.60),
                source_ip=_make_ip(rng, external=rng.random() > 0.5),
                affected_systems=_make_systems(rng, 1),
                confidence=rng.uniform(0.40, 0.65),
                is_real_threat=False,
                optimal_action="investigate",
                attack_stage="none",
                step=step,
                metadata={"ambiguous": True},
            ))
        else:
            # Clear false positive: very low severity, internal IP
            alerts.append(RawAlert(
                alert_id=str(uuid.UUID(int=rng.getrandbits(128))),
                alert_type="anomalous_traffic",
                severity=rng.uniform(0.05, 0.30),
                source_ip=_make_ip(rng, external=False),
                affected_systems=_make_systems(rng, 1),
                confidence=rng.uniform(0.20, 0.45),
                is_real_threat=False,
                optimal_action="ignore",
                attack_stage="none",
                step=step,
                metadata={"false_positive": True},
            ))

    return alerts


# ---------------------------------------------------------------------------
# TASK 3 — Hard: Multi-Stage APT Attack
# 30 steps: a coordinated attack that evolves through 4 stages
# Agent must detect the PATTERN and contain before full compromise
# ---------------------------------------------------------------------------

APT_STAGES: list[tuple[int, int, AttackStage, list[AlertType]]] = [
    # (start_step, end_step, stage, alert_types_that_appear)
    (0,  6,  "reconnaissance",   ["port_scan", "anomalous_traffic"]),
    (7,  14, "initial_access",   ["brute_force", "phishing_attempt"]),
    (15, 22, "lateral_movement", ["lateral_movement", "privilege_escalation"]),
    (23, 29, "exfiltration",     ["data_exfiltration", "ransomware"]),
]

STAGE_OPTIMAL: dict[AttackStage, ActionType] = {
    "reconnaissance":  "investigate",
    "initial_access":  "block",
    "lateral_movement": "isolate",
    "exfiltration":    "isolate",
    "none":            "ignore",
    "full_compromise": "escalate",
}


def generate_task3_episode(episode_seed: int) -> list[RawAlert]:
    rng = random.Random(episode_seed)
    alerts: list[RawAlert] = []

    # The attacker uses a consistent IP family across stages
    attacker_base = rng.choice(EXTERNAL_IPS)
    attacker_ips = [attacker_base] + [
        f"{'.'.join(attacker_base.split('.')[:3])}.{rng.randint(2,254)}"
        for _ in range(3)
    ]

    # Noise ratio: 40% of steps are false positives to confuse the agent
    noise_steps = set(rng.sample(range(30), 12))

    for step in range(30):
        # Determine which APT stage we're in
        current_stage: AttackStage = "none"
        apt_types: list[AlertType] = ["anomalous_traffic"]
        for start, end, stage, types in APT_STAGES:
            if start <= step <= end:
                current_stage = stage
                apt_types = types
                break

        if step in noise_steps:
            # Inject noise to confuse the agent
            alerts.append(RawAlert(
                alert_id=str(uuid.UUID(int=rng.getrandbits(128))),
                alert_type=rng.choice(["anomalous_traffic", "port_scan", "phishing_attempt"]),
                severity=rng.uniform(0.10, 0.45),
                source_ip=_make_ip(rng, external=rng.random() > 0.3),
                affected_systems=_make_systems(rng, 1),
                confidence=rng.uniform(0.25, 0.55),
                is_real_threat=False,
                optimal_action="ignore",
                attack_stage="none",
                step=step,
                metadata={"noise": True},
            ))
        else:
            # Real APT activity
            severity_by_stage = {
                "reconnaissance": rng.uniform(0.20, 0.45),
                "initial_access": rng.uniform(0.55, 0.75),
                "lateral_movement": rng.uniform(0.70, 0.88),
                "exfiltration": rng.uniform(0.82, 0.98),
                "none": rng.uniform(0.05, 0.20),
                "full_compromise": 1.0,
            }
            alerts.append(RawAlert(
                alert_id=str(uuid.UUID(int=rng.getrandbits(128))),
                alert_type=rng.choice(apt_types),
                severity=severity_by_stage[current_stage],
                source_ip=rng.choice(attacker_ips),
                affected_systems=_make_systems(rng, rng.randint(1, 4)),
                confidence=rng.uniform(0.50, 0.85),
                is_real_threat=True,
                optimal_action=STAGE_OPTIMAL[current_stage],
                attack_stage=current_stage,
                step=step,
                metadata={
                    "apt_campaign": True,
                    "attacker_family": attacker_base,
                    "stage_index": [s for s, e, stg, _ in APT_STAGES
                                    if stg == current_stage],
                },
            ))

    return alerts


# ---------------------------------------------------------------------------
# Registry — used by the environment to load the right generator
# ---------------------------------------------------------------------------
TASK_GENERATORS = {
    "task_1_easy":   generate_task1_episode,
    "task_2_medium": generate_task2_episode,
    "task_3_hard":   generate_task3_episode,
}

TASK_MAX_STEPS = {
    "task_1_easy":   5,
    "task_2_medium": 15,
    "task_3_hard":   30,
}