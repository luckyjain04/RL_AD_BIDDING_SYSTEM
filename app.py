from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from engine.environment import RTBEnv
from engine.agent import DQNAgent
import uvicorn, numpy as np

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

env = RTBEnv()
agent = DQNAgent(env.state_dim, len(env.action_space))
m = {"clicks": 0, "spend": 0.0, "steps": 0}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(name="index.html", request={"request": request})

@app.post("/train_step")
async def train_step():
    global m
    current_intent = float(env.current_state[0])
    state = env.current_state
    action_idx = agent.select_action(state)
    next_state, reward, done, info = env.step(action_idx)
    agent.memory.append((state, action_idx, reward, next_state, done))
    agent.train_step()
    if done: env.reset()
    m["steps"] += 1
    if info["clicked"]: m["clicks"] += 1
    m["spend"] += float(info["cost"])
    return {
        "action": float(env.action_space[action_idx]), "reward": float(reward),
        "intent": current_intent, "epsilon": round(float(agent.epsilon), 3),
        "stats": {"clicks": m["clicks"], "spend": round(m["spend"], 2), "ctr": round((m["clicks"]/m["steps"])*100, 2)}
    }

@app.post("/save")
async def save(): agent.save(); return {"msg": "Saved"}
@app.post("/load")
async def load(): return {"msg": "Loaded" if agent.load() else "Failed"}

if __name__ == "__main__": uvicorn.run(app, host="127.0.0.1", port=8000)