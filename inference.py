"""
inference.py — Baseline inference for Adaptive Threat Response System.

Submission checklist:
  [x] Follows sample inference.py strictly
  [x] API_BASE_URL default set, MODEL_NAME default set, HF_TOKEN NO default
  [x] All LLM calls use OpenAI client via these variables
  [x] Stdout follows START/STEP/END structured format exactly
  [x] All risky operations wrapped in try/except — no unhandled exceptions
  [x] Never raises RuntimeError — always exits cleanly with scores

Environment variables:
    API_BASE_URL      Your HF Space URL — where the SOC environment runs
                      Default: http://localhost:7860
    MODEL_NAME        LLM model to use for decisions
                      Default: meta-llama/Llama-3.1-8B-Instruct
    HF_TOKEN          HuggingFace API key — NO DEFAULT, must be set externally
    LOCAL_IMAGE_NAME  Optional — if using from_docker_image()

IMPORTANT: API_BASE_URL is the environment server URL (your HF Space).
           The LLM router URL is hardcoded separately — do not confuse them.
"""

import os
import re
import json
import time
import textwrap
import sys
from typing import Any

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables — exactly per submission checklist
# API_BASE_URL and MODEL_NAME have defaults
# HF_TOKEN has NO default — must be set by caller
# ---------------------------------------------------------------------------
API_BASE_URL     = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME       = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")           # NO default — required
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")   # optional

# LLM router — separate from environment URL, never confused
# This is where LLM calls go. API_BASE_URL is where env calls go.
LLM_BASE_URL = "https://router.huggingface.co/v1"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEMPERATURE  = 0.2
MAX_TOKENS   = 400
TASKS        = ["task_1_easy", "task_2_medium", "task_3_hard"]
ACTION_TYPES = {"ignore", "investigate", "block", "isolate", "escalate"}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI Security Operations Center (SOC) analyst.
    You receive security alerts one at a time and must respond.

    Available actions:
    - ignore       : dismiss alert, no disruption
    - investigate  : gather more info, minimal disruption
    - block        : block source IP, moderate disruption
    - isolate      : isolate affected system, high disruption
    - escalate     : hand to human analyst

    Respond ONLY with valid JSON, no other text:
    {
        "action_type": "<ignore|investigate|block|isolate|escalate>",
        "target": "<source_ip or system name>",
        "reasoning": "<brief reasoning>",
        "confidence": <0.0 to 1.0>
    }
""").strip()


# ---------------------------------------------------------------------------
# SOCEnvClient — HTTP wrapper around the environment API
# Calls API_BASE_URL (the HF Space / local server)
# NOT the LLM router
# ---------------------------------------------------------------------------
class SOCEnvClient:
    """HTTP client for the SOC environment server."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session  = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.done     = False

    def reset(self, task_id: str, episode_seed: int = 42) -> dict[str, Any] | None:
        """Reset environment. Returns observation dict or None on failure."""
        try:
            resp = self.session.post(
                f"{self.base_url}/reset",
                json={"task_id": task_id, "episode_seed": episode_seed},
                timeout=30,
            )
            resp.raise_for_status()
            self.done = False
            return resp.json()
        except Exception as exc:
            print(f"STEP task={task_id} step=0 error=reset_failed detail={exc}")
            return None

    def step(self, action: dict[str, Any]) -> dict[str, Any] | None:
        """Take one step. Returns result dict or None on failure."""
        try:
            resp = self.session.post(
                f"{self.base_url}/step",
                json={"action": action},
                timeout=30,
            )
            resp.raise_for_status()
            result    = resp.json()
            self.done = result.get("done", False)
            return result
        except Exception as exc:
            print(f"STEP error=step_failed detail={exc}")
            self.done = True
            return None

    def grade(self) -> dict[str, Any]:
        """Grade completed episode. Returns grade dict (never raises)."""
        try:
            resp = self.session.post(f"{self.base_url}/grade", timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            print(f"STEP error=grade_failed detail={exc}")
            return {"score": 0.0, "label": "error", "feedback": str(exc)}

    def health(self) -> bool:
        """Check if server is alive. Returns True/False, never raises."""
        try:
            resp = self.session.get(
                f"{self.base_url}/health",
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_user_prompt(
    step: int,
    observation: dict[str, Any],
    history: list[str],
) -> str:
    context = observation.get("context", [])
    ctx_lines = [
        f"  Step {c.get('step','?')}: {c.get('alert_type','?')} "
        f"sev={c.get('severity',0):.2f} -> {c.get('action_taken','none')}"
        for c in context[-4:]
    ]
    ctx_text = "\n".join(ctx_lines) if ctx_lines else "None"

    return textwrap.dedent(f"""
        Task: {observation.get('task_id','?')} | Step: {step}/{observation.get('max_steps',1)}

        CURRENT ALERT:
          Type:     {observation.get('alert_type','?')}
          Severity: {observation.get('severity',0):.2f}
          Source IP:{observation.get('source_ip','?')}
          Systems:  {', '.join(observation.get('affected_systems',[]) or ['none'])}
          Confidence: {observation.get('confidence',0):.2f}
          Stage:    {observation.get('attack_stage','none')}

        WORLD STATE:
          Security: {observation.get('security_state',0):.2f}
          Health:   {observation.get('system_health',1):.2f}

        HISTORY:
        {ctx_text}

        ACTIONS SO FAR:
        {chr(10).join(history[-3:]) if history else 'None'}

        Respond with JSON only.
    """).strip()


# ---------------------------------------------------------------------------
# Response parser — never raises
# ---------------------------------------------------------------------------
def parse_model_action(
    response_text: str,
    observation: dict[str, Any],
) -> dict[str, Any]:
    """Parse LLM response into action dict. Always returns valid action."""
    fallback = {
        "action_type": "ignore",
        "target":      observation.get("source_ip", "0.0.0.0"),
        "reasoning":   "fallback",
        "confidence":  0.1,
    }

    if not response_text:
        return fallback

    # Strip markdown fences
    clean = re.sub(r"```(?:json)?", "", response_text).strip().strip("`")

    # Try JSON parse
    try:
        parsed      = json.loads(clean)
        action_type = str(parsed.get("action_type", "")).lower().strip()
        if action_type not in ACTION_TYPES:
            action_type = "ignore"
        return {
            "action_type": action_type,
            "target":      str(parsed.get("target", observation.get("source_ip", "0.0.0.0"))),
            "reasoning":   str(parsed.get("reasoning", ""))[:300],
            "confidence":  float(parsed.get("confidence", 0.5)),
        }
    except Exception:
        pass

    # Keyword fallback
    for action in ACTION_TYPES:
        if re.search(rf"\b{action}\b", response_text, re.IGNORECASE):
            return {
                "action_type": action,
                "target":      observation.get("source_ip", "0.0.0.0"),
                "reasoning":   "keyword extracted",
                "confidence":  0.3,
            }

    return fallback


# ---------------------------------------------------------------------------
# Single episode — fully wrapped, never raises
# ---------------------------------------------------------------------------
def run_episode(
    env:          SOCEnvClient,
    client:       OpenAI | None,
    task_id:      str,
    episode_seed: int = 42,
) -> dict[str, Any]:
    """
    Run one complete episode.
    Fully wrapped in try/except — never raises, always returns result dict.
    """

    # START log — required structured format
    print(f"START task={task_id} seed={episode_seed} model={MODEL_NAME}")

    observation = env.reset(task_id=task_id, episode_seed=episode_seed)

    if observation is None:
        # Environment not reachable — return zero score, do not crash
        print(f"END task={task_id} score=0.0 label=error total_reward=0.0 steps=0")
        return {
            "task_id":      task_id,
            "score":        0.0,
            "label":        "error",
            "total_reward": 0.0,
            "steps_taken":  0,
            "feedback":     "environment reset failed",
        }

    max_steps     = observation.get("max_steps", 30)
    history:      list[str]   = []
    step_rewards: list[float] = []

    for step in range(1, max_steps + 1):
        if env.done:
            break

        # Build prompt
        user_prompt = build_user_prompt(step, observation, history)
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        # LLM call — wrapped in try/except
        response_text = ""
        if client is not None:
            try:
                completion    = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"STEP task={task_id} step={step} error=llm_failed detail={exc}")
                response_text = ""

        # Parse action
        action = parse_model_action(response_text, observation)

        # Step in environment
        result = env.step(action)

        if result is None:
            # Step failed — end episode
            break

        observation = result.get("observation", observation)
        reward_val  = result.get("reward", {}).get("value", 0.0)
        done        = result.get("done", False)

        step_rewards.append(reward_val)
        history.append(
            f"Step {step}: {action['action_type']} -> reward {reward_val:+.3f}"
        )

        # STEP log — required structured format
        print(
            f"STEP task={task_id} step={step}/{max_steps} "
            f"alert={observation.get('alert_type','?')} "
            f"action={action['action_type']} "
            f"reward={reward_val:+.4f} "
            f"security={observation.get('security_state',0):.3f} "
            f"health={observation.get('system_health',1):.3f} "
            f"done={done}"
        )

        if done:
            break

    # Grade episode
    grade_result  = env.grade()
    total_reward  = sum(step_rewards)

    # END log — required structured format
    print(
        f"END task={task_id} "
        f"score={grade_result.get('score', 0.0):.4f} "
        f"label={grade_result.get('label', 'unknown')} "
        f"total_reward={total_reward:.4f} "
        f"steps={len(step_rewards)}"
    )

    return {
        "task_id":      task_id,
        "score":        grade_result.get("score", 0.0),
        "label":        grade_result.get("label", "unknown"),
        "total_reward": round(total_reward, 4),
        "steps_taken":  len(step_rewards),
        "breakdown":    grade_result.get("breakdown", {}),
        "feedback":     grade_result.get("feedback", ""),
    }


# ---------------------------------------------------------------------------
# Main — never raises, always exits 0
# ---------------------------------------------------------------------------
def main() -> None:
    # START global log
    print(f"START inference API_BASE_URL={API_BASE_URL} MODEL_NAME={MODEL_NAME}")

    # Warn if HF_TOKEN not set but do NOT crash
    # Validator may set it differently — graceful degradation
    if not HF_TOKEN:
        print(
            "STEP warning=HF_TOKEN_not_set "
            "LLM calls will fail, using fallback actions"
        )

    # Initialize environment client
    # If server not reachable, still run and report 0.0 scores
    env       = SOCEnvClient(base_url=API_BASE_URL)
    env_alive = env.health()

    if not env_alive:
        print(
            f"STEP warning=env_not_reachable "
            f"url={API_BASE_URL} "
            f"reporting_zero_scores"
        )

    # Initialize LLM client — only if HF_TOKEN available
    client: OpenAI | None = None
    if HF_TOKEN:
        try:
            client = OpenAI(base_url=LLM_BASE_URL, api_key=HF_TOKEN)
        except Exception as exc:
            print(f"STEP warning=llm_client_init_failed detail={exc}")
            client = None

    # Run all 3 tasks — wrapped individually so one failure doesn't stop others
    results: list[dict[str, Any]] = []

    for task_id in TASKS:
        try:
            result = run_episode(
                env=env,
                client=client,
                task_id=task_id,
                episode_seed=42,   # fixed seed — reproducible scores
            )
        except Exception as exc:
            # Last resort catch — should never reach here but guarantees no crash
            print(f"END task={task_id} score=0.0 label=error total_reward=0.0 steps=0")
            result = {
                "task_id":      task_id,
                "score":        0.0,
                "label":        "error",
                "total_reward": 0.0,
                "steps_taken":  0,
                "feedback":     str(exc),
            }
        results.append(result)

    # Final scores — always printed regardless of what happened above
    print("=" * 60)
    print("FINAL SCORES")
    print("=" * 60)
    for r in results:
        print(
            f"{r['task_id']:20s}  "
            f"score={r['score']:.4f}  "
            f"({r['label']:9s})  "
            f"reward={r['total_reward']:+.4f}  "
            f"steps={r['steps_taken']}"
        )

    avg = sum(r["score"] for r in results) / max(len(results), 1)
    print(f"\nAverage score: {avg:.4f}")

    # END global log
    print("END inference complete")

    # Always exit 0 — no unhandled exceptions
    sys.exit(0)


if __name__ == "__main__":
    main()
