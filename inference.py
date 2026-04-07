"""
inference.py — Baseline inference script for Adaptive Threat Response System.

Submission checklist compliance:
  [x] Follows sample inference.py pattern strictly
  [x] API_BASE_URL has default, MODEL_NAME has default, HF_TOKEN has NO default
  [x] All LLM calls use OpenAI client configured via these variables
  [x] Stdout follows START/STEP/END structured format exactly

Environment variables:
    API_BASE_URL      HF Space URL (default: your active space)
    MODEL_NAME        LLM model identifier (default: active model)
    HF_TOKEN          HuggingFace API key — NO DEFAULT, must be set
    LOCAL_IMAGE_NAME  Optional — if using from_docker_image()
"""

import os
import re
import json
import textwrap
from typing import Any
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables — exactly as checklist requires
# API_BASE_URL and MODEL_NAME have defaults, HF_TOKEN does NOT
# ---------------------------------------------------------------------------
API_BASE_URL     = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME       = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")                  # NO default — required
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")           # optional

# ---------------------------------------------------------------------------
# LLM client setup — OpenAI client only, as required
# ---------------------------------------------------------------------------
LLM_BASE_URL = "https://router.huggingface.co/v1"
TEMPERATURE  = 0.2
MAX_TOKENS   = 400

# ---------------------------------------------------------------------------
# Tasks to run
# ---------------------------------------------------------------------------
TASKS = ["task_1_easy", "task_2_medium", "task_3_hard"]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI Security Operations Center (SOC) analyst.
    You receive security alerts one at a time and must decide how to respond.

    Available actions:
    - ignore       : dismiss the alert, no disruption
    - investigate  : gather more information, minimal disruption
    - block        : block the source IP, moderate disruption
    - isolate      : isolate the affected system, high disruption
    - escalate     : hand off to human analyst

    Goal: block real threats quickly, ignore false positives,
    balance security vs system health, recognize attack patterns.

    Respond ONLY with a valid JSON object — no other text:
    {
        "action_type": "<ignore|investigate|block|isolate|escalate>",
        "target": "<source_ip or system name>",
        "reasoning": "<brief reasoning>",
        "confidence": <0.0 to 1.0>
    }
""").strip()


# ---------------------------------------------------------------------------
# SOCEnvClient — HTTP wrapper around the environment API
# ---------------------------------------------------------------------------
class SOCEnvClient:
    """Thin HTTP wrapper so inference loop treats remote env like a local object."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session  = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.done     = False

    def reset(self, task_id: str, episode_seed: int = 42) -> dict[str, Any]:
        payload = {"task_id": task_id, "episode_seed": episode_seed}
        resp = self.session.post(f"{self.base_url}/reset", json=payload, timeout=30)
        resp.raise_for_status()
        self.done = False
        return resp.json()

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        resp = self.session.post(
            f"{self.base_url}/step", json={"action": action}, timeout=30
        )
        resp.raise_for_status()
        result    = resp.json()
        self.done = result.get("done", False)
        return result

    def grade(self) -> dict[str, Any]:
        resp = self.session.post(f"{self.base_url}/grade", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=10)
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
    history_lines = [
        f"  Step {c.get('step','?')}: {c.get('alert_type')} "
        f"sev={c.get('severity',0):.2f} -> {c.get('action_taken','none')}"
        for c in context[-4:]
    ]
    history_text = "\n".join(history_lines) if history_lines else "None"

    return textwrap.dedent(f"""
        Task: {observation.get('task_id','?')} | Step: {step}/{observation.get('max_steps',1)}

        CURRENT ALERT:
          Type:             {observation.get('alert_type','?')}
          Severity:         {observation.get('severity',0):.2f}
          Source IP:        {observation.get('source_ip','?')}
          Affected systems: {', '.join(observation.get('affected_systems',[]) or ['none'])}
          Confidence:       {observation.get('confidence',0):.2f}
          Attack stage:     {observation.get('attack_stage','none')}

        WORLD STATE:
          Security threat level: {observation.get('security_state',0):.2f}
          System health:         {observation.get('system_health',1):.2f}

        RECENT HISTORY:
        {history_text}

        PREVIOUS ACTIONS:
        {chr(10).join(history[-3:]) if history else 'None'}

        Respond with JSON action only.
    """).strip()


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------
ACTION_TYPES = {"ignore", "investigate", "block", "isolate", "escalate"}

def parse_model_action(
    response_text: str,
    observation: dict[str, Any],
) -> dict[str, Any]:
    fallback = {
        "action_type": "ignore",
        "target":      observation.get("source_ip", "0.0.0.0"),
        "reasoning":   "fallback: parse failed",
        "confidence":  0.1,
    }
    if not response_text:
        return fallback

    clean = re.sub(r"```(?:json)?", "", response_text).strip().strip("`")

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
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

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
# Single episode runner
# ---------------------------------------------------------------------------
def run_episode(
    env: SOCEnvClient,
    client: OpenAI,
    task_id: str,
    episode_seed: int = 42,
) -> dict[str, Any]:
    """Run one complete episode. Returns graded result."""

    # START log — required structured format
    print(f"START task={task_id} seed={episode_seed} model={MODEL_NAME}")

    observation   = env.reset(task_id=task_id, episode_seed=episode_seed)
    max_steps     = observation.get("max_steps", 30)
    history:      list[str]   = []
    step_rewards: list[float] = []

    for step in range(1, max_steps + 1):
        if env.done:
            break

        user_prompt = build_user_prompt(step, observation, history)
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

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

        action = parse_model_action(response_text, observation)

        try:
            result      = env.step(action)
            observation = result["observation"]
            reward_val  = result["reward"]["value"]
            done        = result["done"]
        except Exception as exc:
            print(f"STEP task={task_id} step={step} error=step_failed detail={exc}")
            break

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

    try:
        grade_result = env.grade()
    except Exception as exc:
        grade_result = {"score": 0.0, "label": "error", "feedback": str(exc)}

    total_reward = sum(step_rewards)

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
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Validate HF_TOKEN is set — no default allowed per checklist
    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set. "
            "Export it before running: export HF_TOKEN=hf_..."
        )

    print(f"START inference API_BASE_URL={API_BASE_URL} MODEL_NAME={MODEL_NAME}")

    env = SOCEnvClient(base_url=API_BASE_URL)
    if not env.health():
        raise RuntimeError(
            f"Environment not reachable at {API_BASE_URL}. "
            "Start server: uvicorn api.server:app --port 7860"
        )

    # OpenAI client — using HF router, as required by checklist
    client = OpenAI(base_url=LLM_BASE_URL, api_key=HF_TOKEN)

    results: list[dict[str, Any]] = []

    for task_id in TASKS:
        try:
            result = run_episode(
                env=env,
                client=client,
                task_id=task_id,
                episode_seed=42,   # fixed seed — reproducible scores
            )
            results.append(result)
        except Exception as exc:
            print(f"END task={task_id} score=0.0 label=error total_reward=0.0 steps=0")
            results.append({
                "task_id": task_id, "score": 0.0,
                "label": "error", "total_reward": 0.0,
                "steps_taken": 0, "feedback": str(exc),
            })

    # Final summary
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
    print("END inference complete")


if __name__ == "__main__":
    main()
