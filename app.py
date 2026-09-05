from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from engine.environment import RTBEnv
from engine.agent import DQNAgent
import uvicorn, numpy as np, traceback

app = FastAPI()

# Ensure folders exist
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

env = RTBEnv()
agent = DQNAgent(env.state_dim, len(env.action_space))
m = {"clicks": 0, "spend": 0.0, "steps": 0, "revenue": 0.0}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Pass 'request' as a keyword argument (Required by latest FastAPI versions)
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

@app.post("/train_step")
async def train_step():
    global m
    try:
        current_intent = float(env.current_state[0])
        state = env.current_state
        action_idx = agent.select_action(state)
        next_state, reward, done, info = env.step(action_idx)
        
        agent.memory.append((state, action_idx, reward, next_state, done))
        agent.train_step()
        
        if done: env.reset()
        
        m["steps"] += 1
        m["spend"] += float(info["cost"])
        if info["clicked"]: 
            m["clicks"] += 1
            if np.random.rand() < 0.10: # 10% Purchase Probability
                m["revenue"] += 150000.0

        return {
            "action": float(env.action_space[action_idx]), "reward": float(reward),
            "intent": current_intent, "epsilon": round(float(agent.epsilon), 3),
            "stats": {
                "clicks": m["clicks"], "spend": round(m["spend"], 2), 
                "revenue": round(m["revenue"], 2),
                "ctr": round((m["clicks"]/m["steps"])*100, 2)
            }
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/save")
async def save(): agent.save(); return {"msg": "Weights Saved"}

@app.post("/load")
async def load(): 
    success = agent.load()
    return {"msg": "Weights Loaded" if success else "Failed to find weights"}

if __name__ == "__main__": 
    uvicorn.run(app, host="127.0.0.1", port=8000)