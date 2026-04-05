"""
FastAPI server — Adaptive Threat Response System
Exposes the SOCEnvironment via HTTP for agents and validators.

Validation script checks:
  POST /reset  must return HTTP 200 with empty body {}

OpenEnv spec requires:
  POST /reset   → Observation
  POST /step    → {observation, reward, done, info}
  GET  /state   → EnvironmentState

Additional:
  GET  /health  → 200 (ping check)
  GET  /tasks   → task metadata list
  POST /grade   → score after episode ends
"""

from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from environment.env import SOCEnvironment
from environment.models import (
    Observation, Action, Reward, EnvironmentState, TaskInfo
)

app = FastAPI(
    title="Adaptive Threat Response System",
    description=(
        "OpenEnv environment simulating a Security Operations Center analyst. "
        "An AI agent learns to triage security alerts, balancing threat containment "
        "against system availability over sequential decision episodes."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance
env = SOCEnvironment()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    task_id: str = "task_1_easy"      # default so empty {} body works
    episode_seed: int | None = None


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


class GradeResponse(BaseModel):
    task_id: str
    score: float
    label: str
    breakdown: dict[str, Any]
    feedback: str


# ---------------------------------------------------------------------------
# Health / ping — validator hits this first, must be 200
# ---------------------------------------------------------------------------
@app.get("/")
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "environment": "Adaptive Threat Response System",
        "version": "1.0.0",
        "openenv": True,
        "tasks": ["task_1_easy", "task_2_medium", "task_3_hard"],
    }


# ---------------------------------------------------------------------------
# OpenEnv core endpoints
# ---------------------------------------------------------------------------
@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest = ResetRequest()) -> Observation:
    """
    Start a new episode.
    Validation script calls this with empty body {} — must return 200.
    """
    try:
        return env.reset(
            task_id=request.task_id,
            episode_seed=request.episode_seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    """Take one action. Returns next observation, reward, done, info."""
    try:
        obs, reward, done, info = env.step(request.action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=EnvironmentState)
def state() -> EnvironmentState:
    """Full internal state — for openenv validate and debugging."""
    return env.state()


# ---------------------------------------------------------------------------
# Additional endpoints
# ---------------------------------------------------------------------------


@app.get("/tasks", response_model=list[TaskInfo])
def list_tasks() -> list[TaskInfo]:
    """List all available tasks with metadata."""
    return [
        TaskInfo(
            task_id="task_1_easy",
            name="Single Alert Triage",
            description=(
                "One high-severity brute-force attack hidden among 4 noise alerts. "
                "Agent must identify and block it quickly without over-reacting."
            ),
            difficulty="easy",
            max_steps=5,
            action_space=["ignore", "investigate", "block", "isolate", "escalate"],
            observation_fields=[
                "alert_id", "alert_type", "severity", "source_ip",
                "affected_systems", "confidence", "security_state",
                "system_health", "step_number", "max_steps", "context",
            ],
            grader_description=(
                "60% threat blocked, 20% speed of block, 20% false-positive discipline."
            ),
            expected_score_random_agent=0.15,
            expected_score_optimal_agent=0.95,
        ),
        TaskInfo(
            task_id="task_2_medium",
            name="Alert Queue Under Pressure",
            description=(
                "15 steps: 3 real threats hidden in 8 false positives + 4 ambiguous alerts. "
                "Agent must prioritize and use proportionate responses."
            ),
            difficulty="medium",
            max_steps=15,
            action_space=["ignore", "investigate", "block", "isolate", "escalate"],
            observation_fields=[
                "alert_id", "alert_type", "severity", "source_ip",
                "affected_systems", "confidence", "security_state",
                "system_health", "step_number", "max_steps", "context",
            ],
            grader_description=(
                "50% recall, 30% precision, 20% response proportionality."
            ),
            expected_score_random_agent=0.20,
            expected_score_optimal_agent=0.90,
        ),
        TaskInfo(
            task_id="task_3_hard",
            name="Multi-Stage APT Attack",
            description=(
                "30 steps: a coordinated attack through 4 stages (recon → access → "
                "lateral movement → exfiltration). Agent must detect the pattern and "
                "contain before full compromise. 12 noise alerts provide distraction."
            ),
            difficulty="hard",
            max_steps=30,
            action_space=["ignore", "investigate", "block", "isolate", "escalate"],
            observation_fields=[
                "alert_id", "alert_type", "severity", "source_ip",
                "affected_systems", "confidence", "security_state",
                "system_health", "step_number", "max_steps", "context",
                "attack_stage",
            ],
            grader_description=(
                "40% containment stage, 30% final system health, "
                "20% pattern recognition, 10% speed."
            ),
            expected_score_random_agent=0.10,
            expected_score_optimal_agent=0.85,
        ),
    ]


@app.post("/grade", response_model=GradeResponse)
def grade() -> GradeResponse:
    """Grade the most recently completed episode. Call after done=True."""
    try:
        result = env.grade_episode()
        return GradeResponse(
            task_id=env._task_id,
            score=result.score,
            label=result.label,
            breakdown=result.breakdown,
            feedback=result.feedback,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks/{task_id}", response_model=TaskInfo)
def get_task(task_id: str) -> TaskInfo:
    """Get metadata for a specific task."""
    all_tasks = {t.task_id: t for t in list_tasks()}
    if task_id not in all_tasks:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    return all_tasks[task_id]