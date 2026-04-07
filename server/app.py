from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from environment.env import Env  # your environment logic

app = FastAPI(title="Adaptive Threat Response System")

# Allow all origins (HF Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Create a singleton environment
env = Env()

@app.get("/health")
def health():
    return {"status": "ok", "environment": env.name, "version": env.version, "openenv": True, "tasks": env.tasks}

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: dict):
    return env.step(action)

@app.get("/state")
def state():
    return env.state()

# This is needed for openenv validate
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()