"""
SOCEnvironment — core OpenEnv environment.

The agent interacts via step() / reset() / state().
Manages episode lifecycle, world state, alert serving,
reward computation, and episode transcript for grading.
"""

from __future__ import annotations
import uuid
import time
from typing import Any

from environment.models import (
    Observation, Action, Reward, EnvironmentState,
    AlertContext, AttackStage,
)
from environment.alert_generator import (
    RawAlert, TASK_GENERATORS, TASK_MAX_STEPS,
)
from environment.reward import compute_reward, update_world_state
from environment.graders import GRADERS, GradeResult


class SOCEnvironment:
    """
    Adaptive Threat Response System — OpenEnv compatible environment.

    An AI agent learns by calling:
        obs          = env.reset(task_id)
        obs, rew, done, info = env.step(action)   # repeat until done
        score        = env.grade_episode()
    """

    CONTEXT_WINDOW = 5  # how many past alerts the agent can see

    def __init__(self) -> None:
        self._task_id:         str               = "task_1_easy"
        self._episode_id:      str               = ""
        self._alerts:          list[RawAlert]    = []
        self._step:            int               = 0
        self._max_steps:       int               = 5
        self._episode_done:    bool              = True

        self._security_state:  float             = 0.1
        self._system_health:   float             = 1.0
        self._attack_stage:    AttackStage       = "none"

        self._blocked_ips:     list[str]         = []
        self._isolated_sys:    list[str]         = []
        self._active_threats:  list[str]         = []
        self._total_reward:    float             = 0.0
        self._true_positives:  int               = 0
        self._false_positives: int               = 0
        self._false_negatives: int               = 0

        self._transcript:      list[dict]        = []
        self._context:         list[AlertContext] = []

    # ------------------------------------------------------------------
    # reset() — start a new episode
    # ------------------------------------------------------------------
    def reset(
        self,
        task_id: str = "task_1_easy",
        episode_seed: int | None = None,
    ) -> Observation:
        """Start a new episode. Returns the first observation."""
        if task_id not in TASK_GENERATORS:
            raise ValueError(
                f"Unknown task '{task_id}'. "
                f"Valid: {list(TASK_GENERATORS.keys())}"
            )

        seed = episode_seed if episode_seed is not None \
            else int(time.time() * 1000) % 100_000

        self._task_id      = task_id
        self._episode_id   = str(uuid.uuid4())
        self._alerts       = TASK_GENERATORS[task_id](seed)
        self._step         = 0
        self._max_steps    = TASK_MAX_STEPS[task_id]
        self._episode_done = False

        self._security_state = 0.10
        self._system_health  = 1.00
        self._attack_stage   = "none"

        self._blocked_ips    = []
        self._isolated_sys   = []
        self._active_threats = []
        self._total_reward   = 0.0
        self._true_positives  = 0
        self._false_positives = 0
        self._false_negatives = 0
        self._transcript     = []
        self._context        = []

        return self._build_observation()

    # ------------------------------------------------------------------
    # step() — agent takes an action, world advances one timestep
    # ------------------------------------------------------------------
    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """
        Apply action, advance world, return (observation, reward, done, info).

        This is the core learning loop. Every call:
          1. Scores the action against the current alert (reward)
          2. Updates world state (security_state, system_health)
          3. Records the step in the episode transcript
          4. Serves the next alert as the next observation
        """
        if self._episode_done:
            raise RuntimeError("Episode is done — call reset() first.")

        current_alert = self._alerts[self._step]

        # 1. Compute reward
        reward_scalar, breakdown, step_info = compute_reward(
            alert=current_alert,
            action=action.action_type,
            security_state=self._security_state,
            system_health=self._system_health,
        )

        # 2. Advance world state
        self._security_state, self._system_health = update_world_state(
            security_state=self._security_state,
            system_health=self._system_health,
            alert=current_alert,
            action=action.action_type,
        )
        self._attack_stage = current_alert.attack_stage

        # 3. Side-effect tracking
        if action.action_type == "block" and action.target not in self._blocked_ips:
            self._blocked_ips.append(action.target)
        if action.action_type == "isolate" and action.target not in self._isolated_sys:
            self._isolated_sys.append(action.target)
        if current_alert.is_real_threat and current_alert.source_ip not in self._active_threats:
            self._active_threats.append(current_alert.source_ip)

        # 4. Accuracy counters
        if current_alert.is_real_threat:
            if action.action_type in ("block", "isolate", "investigate"):
                self._true_positives += 1
            else:
                self._false_negatives += 1
        else:
            if action.action_type in ("block", "isolate"):
                self._false_positives += 1

        # 5. Record transcript (used by graders after episode ends)
        self._transcript.append({
            "step":           self._step,
            "alert_id":       current_alert.alert_id,
            "alert_type":     current_alert.alert_type,
            "attack_stage":   current_alert.attack_stage,
            "action":         action.action_type,
            "action_target":  action.target,
            "reward":         reward_scalar,
            "security_state": self._security_state,
            "system_health":  self._system_health,
            "info":           step_info,
        })

        # 6. Update context window (agent's short-term memory)
        self._context.append(AlertContext(
            alert_id=current_alert.alert_id,
            alert_type=current_alert.alert_type,
            severity=current_alert.severity,
            action_taken=action.action_type,
            step=self._step,
        ))
        if len(self._context) > self.CONTEXT_WINDOW:
            self._context.pop(0)

        self._total_reward += reward_scalar
        self._step += 1

        # 7. Check episode termination
        done = (
            self._step >= self._max_steps
            or self._security_state >= 0.95   # full compromise
            or self._system_health   <= 0.05  # system unusable
        )
        self._episode_done = done

        if done:
            step_info["episode_summary"] = {
                "total_reward":    round(self._total_reward, 4),
                "final_security":  self._security_state,
                "final_health":    self._system_health,
                "true_positives":  self._true_positives,
                "false_positives": self._false_positives,
                "false_negatives": self._false_negatives,
                "steps_taken":     self._step,
            }
            next_obs = self._build_final_observation()
        else:
            next_obs = self._build_observation()

        reward_obj = Reward(
            value=round(reward_scalar, 4),
            breakdown=breakdown,
            done=done,
            info=step_info,
        )

        return next_obs, reward_obj, done, step_info

    # ------------------------------------------------------------------
    # state() — full internal state (for validators, not the agent)
    # ------------------------------------------------------------------
    def state(self) -> EnvironmentState:
        """
        Returns complete ground-truth state.
        Used by openenv validate and debuggers.
        The agent does NOT call this during normal operation.
        """
        return EnvironmentState(
            task_id=self._task_id,
            episode_id=self._episode_id,
            step_number=self._step,
            max_steps=self._max_steps,
            security_state=self._security_state,
            system_health=self._system_health,
            attack_stage=self._attack_stage,
            active_threats=list(self._active_threats),
            blocked_ips=list(self._blocked_ips),
            isolated_systems=list(self._isolated_sys),
            total_reward=round(self._total_reward, 4),
            alerts_seen=self._step,
            true_positives=self._true_positives,
            false_positives=self._false_positives,
            false_negatives=self._false_negatives,
            alert_history=list(self._transcript),
            episode_done=self._episode_done,
        )

    # ------------------------------------------------------------------
    # grade_episode() — run the task grader on the completed transcript
    # ------------------------------------------------------------------
    def grade_episode(self) -> GradeResult:
        """
        Score the completed episode using the task's deterministic grader.
        Must be called after the episode is done.
        """
        if not self._episode_done:
            raise RuntimeError("Episode is still running — cannot grade yet.")
        grader = GRADERS[self._task_id]
        return grader(self._transcript)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_observation(self) -> Observation:
        alert = self._alerts[self._step]
        return Observation(
            alert_id=alert.alert_id,
            alert_type=alert.alert_type,
            severity=alert.severity,
            source_ip=alert.source_ip,
            affected_systems=alert.affected_systems,
            confidence=alert.confidence,
            security_state=self._security_state,
            system_health=self._system_health,
            step_number=self._step,
            max_steps=self._max_steps,
            task_id=self._task_id,
            context=list(self._context),
            attack_stage=alert.attack_stage,
        )

    def _build_final_observation(self) -> Observation:
        """Final observation when episode ends — uses last alert's data."""
        last_alert = self._alerts[min(self._step - 1, len(self._alerts) - 1)]
        return Observation(
            alert_id="EPISODE_END",
            alert_type=last_alert.alert_type,
            severity=0.0,
            source_ip="0.0.0.0",
            affected_systems=[],
            confidence=0.0,
            security_state=self._security_state,
            system_health=self._system_health,
            step_number=self._step,
            max_steps=self._max_steps,
            task_id=self._task_id,
            context=list(self._context),
            attack_stage="none",
        )