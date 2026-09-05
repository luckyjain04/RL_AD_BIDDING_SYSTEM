import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x): return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.model = QNetwork(state_dim, action_dim)
        self.target_model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=5000)
        self.gamma, self.epsilon, self.epsilon_decay = 0.95, 1.0, 0.99

    def select_action(self, state):
        if random.random() < self.epsilon: return random.randint(0, self.action_dim - 1)
        state_t = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): return int(torch.argmax(self.model(state_t)).item())

    def train_step(self, batch_size=10):
        if len(self.memory) < batch_size: return
        batch = random.sample(self.memory, batch_size)
        s, a, r, ns, d = zip(*batch)
        s, a, r, ns = torch.tensor(np.array(s)), torch.tensor(np.array(a)), torch.tensor(np.array(r)), torch.tensor(np.array(ns))
        d = torch.tensor(np.array(d), dtype=torch.float32) # Fix for RuntimeError

        current_q = self.model(s).gather(1, a.unsqueeze(1))
        next_q = self.target_model(ns).max(1)[0].detach()
        target_q = r + (1 - d) * self.gamma * next_q
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        self.epsilon = float(max(0.01, self.epsilon * self.epsilon_decay))

    def save(self, f="ad_agent.pth"): torch.save(self.model.state_dict(), f)
    def load(self, f="ad_agent.pth"):
        if os.path.exists(f):
            self.model.load_state_dict(torch.load(f)); self.epsilon = 0.1; return True
        return False