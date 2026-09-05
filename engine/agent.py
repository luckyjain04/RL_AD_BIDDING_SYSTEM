import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from collections import deque


# ============================================================
# Q NETWORK
# ============================================================

class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):

        super(QNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


# ============================================================
# DQN AGENT
# ============================================================

class DQNAgent:

    def __init__(self, state_dim, action_dim):

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Main network
        self.model = QNetwork(
            state_dim,
            action_dim
        )

        # Target network
        self.target_model = QNetwork(
            state_dim,
            action_dim
        )

        # Initially target = main network
        self.target_model.load_state_dict(
            self.model.state_dict()
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        self.memory = deque(
            maxlen=5000
        )

        # ====================================================
        # DQN PARAMETERS
        # ====================================================

        self.gamma = 0.95

        # BABY BRAIN
        self.epsilon = 1.0

        # Minimum exploration
        self.epsilon_min = 0.01

        # Gradual learning
        self.epsilon_decay = 0.99

        # Target network update
        self.target_update_frequency = 50

        self.training_steps = 0

        # Last loss for dashboard
        self.last_loss = 0.0


    # ========================================================
    # ACTION SELECTION
    # ========================================================

    def select_action(self, state):

        # ====================================================
        # EXPLORATION
        # ====================================================

        if random.random() < self.epsilon:

            return random.randint(
                0,
                self.action_dim - 1
            )

        # ====================================================
        # EXPLOITATION
        # ====================================================

        state_t = torch.tensor(
            np.array(state),
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.model(
                state_t
            )

        return int(
            torch.argmax(q_values).item()
        )


    # ========================================================
    # TRAINING STEP
    # ========================================================

    def train_step(
        self,
        batch_size=32
    ):

        # Not enough experiences yet
        if len(self.memory) < batch_size:

            return 0.0


        # ====================================================
        # SAMPLE EXPERIENCE
        # ====================================================

        batch = random.sample(
            self.memory,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        states = torch.tensor(
            np.array(states),
            dtype=torch.float32
        )

        actions = torch.tensor(
            np.array(actions),
            dtype=torch.long
        )

        rewards = torch.tensor(
            np.array(rewards),
            dtype=torch.float32
        )

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32
        )

        dones = torch.tensor(
            np.array(dones),
            dtype=torch.float32
        )


        # ====================================================
        # CURRENT Q VALUES
        # ====================================================

        current_q = self.model(
            states
        ).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)


        # ====================================================
        # TARGET Q VALUES
        # ====================================================

        with torch.no_grad():

            next_q = self.target_model(
                next_states
            ).max(
                1
            )[0]

            target_q = (
                rewards
                +
                (1 - dones)
                * self.gamma
                * next_q
            )


        # ====================================================
        # LOSS
        # ====================================================

        loss = nn.MSELoss()(
            current_q,
            target_q
        )


        # ====================================================
        # BACKPROPAGATION
        # ====================================================

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


        # ====================================================
        # EPSILON DECAY
        # ====================================================

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


        # ====================================================
        # TARGET NETWORK UPDATE
        # ====================================================

        self.training_steps += 1

        if (
            self.training_steps
            % self.target_update_frequency
            == 0
        ):

            self.target_model.load_state_dict(
                self.model.state_dict()
            )


        # Store loss
        self.last_loss = float(
            loss.item()
        )

        return self.last_loss


    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        f="ad_agent.pth"
    ):

        torch.save(
            {
                "model_state_dict":
                    self.model.state_dict(),

                "target_model_state_dict":
                    self.target_model.state_dict(),

                "optimizer_state_dict":
                    self.optimizer.state_dict(),

                "epsilon":
                    self.epsilon,

                "training_steps":
                    self.training_steps
            },
            f
        )


    # ========================================================
    # LOAD
    # ========================================================

    def load(
        self,
        f="ad_agent.pth"
    ):

        if not os.path.exists(f):

            return False


        checkpoint = torch.load(
            f,
            map_location="cpu"
        )


        # New checkpoint format
        if isinstance(
            checkpoint,
            dict
        ) and "model_state_dict" in checkpoint:

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            if "target_model_state_dict" in checkpoint:

                self.target_model.load_state_dict(
                    checkpoint[
                        "target_model_state_dict"
                    ]
                )

            else:

                self.target_model.load_state_dict(
                    self.model.state_dict()
                )


            if "optimizer_state_dict" in checkpoint:

                self.optimizer.load_state_dict(
                    checkpoint[
                        "optimizer_state_dict"
                    ]
                )


            self.epsilon = checkpoint.get(
                "epsilon",
                0.1
            )

            self.training_steps = checkpoint.get(
                "training_steps",
                0
            )

        else:

            # Compatibility with your old .pth
            self.model.load_state_dict(
                checkpoint
            )

            self.target_model.load_state_dict(
                self.model.state_dict()
            )

            self.epsilon = 0.1


        return True