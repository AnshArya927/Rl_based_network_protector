---

## Team DETAILS
Team Name - Synaptech

Member - Ansh Arya , Abhishree Sinha

---


## What This Is

A real-world OpenEnv environment where an AI agent learns to act as a Security Operations Center (SOC) analyst. The agent receives security alerts, makes response decisions, and gets reward signals that reflect whether those decisions were correct. Over many episodes the agent learns to balance security posture against system availability.

---

## The Real Problem

In real organizations, security systems fire hundreds of alerts every day. The unsolved problem is not detecting alerts — it is deciding what to do after detection. Every alert needs a decision. Each decision has consequences that change the future state of the world:

- Ignore a real threat → attacker spreads → future alerts get worse
- Block everything → system health degrades → future decisions get harder
- The agent must learn the balance

---

## Action Space

| Action | Description | Disruption |
|---|---|---|
| `ignore` | Dismiss alert | None |
| `investigate` | Gather more info | Minimal |
| `block` | Block source IP | Moderate |
| `isolate` | Isolate system | High |
| `escalate` | Hand to human | Delayed |

```json
{
    "action_type": "block",
    "target": "203.0.113.45",
    "reasoning": "High severity brute force, external IP, 85% confidence",
    "confidence": 0.85
}
```

---

## Observation Space

| Field | Type | Range | Description |
|---|---|---|---|
| `alert_type` | string | 10 types | brute_force, malware_detected, data_exfiltration, lateral_movement, port_scan, etc. |
| `severity` | float | 0–1 | Alert severity score |
| `source_ip` | string | — | IP generating the alert |
| `affected_systems` | list | — | Systems involved |
| `confidence` | float | 0–1 | Detection confidence |
| `security_state` | float | 0–1 | Threat level — 0=safe, 1=compromised |
| `system_health` | float | 0–1 | Availability — 1=operational, 0=unusable |
| `attack_stage` | string | 6 stages | none, reconnaissance, initial_access, lateral_movement, exfiltration, full_compromise |
| `context` | list | last 5 | Recent alerts + actions for pattern recognition |

---

## Reward Function

Signal at **every step** — not binary end-of-episode:

```
reward = (security_gain × 0.4) + (health_preservation × 0.3)
       - (false_positive_penalty × 0.2) - (missed_threat_penalty × 0.1)
```

Clipped to `[-1.0, 1.0]`. Agent learns: do not block everything (health penalty) and do not ignore everything (security penalty).

---

## Tasks

### Task 1 — Easy: Single Alert Triage
5 steps. One real brute-force attack among 4 noise alerts. Block the right IP fast.
Grader: 60% threat blocked + 20% speed + 20% false-positive discipline.
Random: ~0.15 | Optimal: ~0.95

### Task 2 — Medium: Alert Queue Under Pressure
15 steps. 3 real threats among 8 false positives and 4 ambiguous alerts. Prioritize correctly.
Grader: 50% recall + 30% precision + 20% proportionality.
Random: ~0.20 | Optimal: ~0.90

### Task 3 — Hard: Multi-Stage APT Attack
30 steps. Coordinated APT across 4 stages: recon → access → lateral → exfiltration. Contain before full compromise. 12 noise alerts distract throughout.
Grader: 40% containment stage + 30% system health + 20% pattern recognition + 10% speed.
Random: ~0.10 | Optimal: ~0.85

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/reset` | POST | Start episode, returns first Observation |
| `/step` | POST | Take action, returns Observation + Reward + done |
| `/state` | GET | Full internal state for validators |
| `/tasks` | GET | List all 3 tasks |
| `/grade` | POST | Grade completed episode |
| `/docs` | GET | Swagger UI |

---

## Setup

```bash
git clone https://github.com/AnshArya927/Rl_based_network_protector
cd Rl_based_network_protector
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn api.server:app --host 0.0.0.0 --port 7860
```

### Docker
```bash
docker build -t soc-analyst-env .
docker run -p 7860:7860 soc-analyst-env
```

### Run LLM Baseline
```bash
export API_BASE_URL="http://localhost:7860"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token"
python inference.py
```


### Validate
```bash
openenv validate
./validate-submission.sh https://huggingface.co/spaces/AnshArya2810/soc-analyst-env .
```

---

## Baseline Scores

### LLM Agent (with API credit limitations)

| Task | Score | Label |
|---|---|---|
| task_1_easy | 0.2000 | failed |
| task_2_medium | 0.7357 | good |
| task_3_hard | 0.3667 | failed |
| **Average** | **0.4341** | |

**Note:** Some inference steps failed due to Hugging Face API credit limits (HTTP 402), which affected performance on longer tasks.


## Project Structure

```
soc-analyst-env/
├── environment/
│   ├── models.py           # Typed Observation, Action, Reward
│   ├── alert_generator.py  # Synthetic alerts for all 3 tasks
│   ├── reward.py           # Partial reward — signal every step
│   ├── graders.py          # 3 deterministic graders 0.0–1.0
│   └── env.py              # step() reset() state()
├── api/
│   └── server.py           # FastAPI HTTP layer
├── server/                 
│   └── app.py              # OpenEnv validation bridg
├── inference.py            # LLM baseline (required by hackathon)
├── openenv.yaml            # OpenEnv spec metadata
├── Dockerfile              # Builds HF Space Docker image
├── requirements.txt        # Dependencies
├── pyproject.toml          # Python project info (name, version, scripts, dependencies)
├── uv.lock                 # Dependency lock file generated by `uv lock`
├── validate-submission.sh   # OpenEnv submission validation script
└── README.md                # Project description, instructions, and tasks

```

---

## How Agents Learn

```
reset(task_id)  →  Observation (alert + world state)
                        ↓
              Agent picks Action
                        ↓
step(action)    →  Reward + next Observation
                        ↓
         LLM: use history in next prompt
                        ↓
              Repeat until done=True
                        ↓
grade_episode() →  Score 0.0–1.0
```

The environment is the teacher. Reward signals are the curriculum.

---

## Infrastructure

- 2 vCPU, 8 GB RAM, no GPU
- Port 7860
- Pure Python — no PyTorch, no TensorFlow

## License
MIT
