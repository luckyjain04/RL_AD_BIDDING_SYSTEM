# ⚡ Personalized AI Ad Bidding System

A **Reinforcement Learning-based Real-Time Bidding (RTB) system** that uses a **Deep Q-Network (DQN)** to learn intelligent and personalized advertising bidding decisions.

The system simulates real-time advertising auctions, allows an RL agent to learn from auction outcomes, and provides an interactive **Streamlit dashboard** for monitoring model learning, bidding behavior, rewards, CTR, ROI, and auction performance.

---

## 🚀 Overview

In real-time advertising, advertisers must decide **how much to bid for each individual ad impression** within milliseconds.

A bidding strategy that bids too aggressively can waste the advertising budget, while bidding too conservatively can result in missed opportunities.

This project explores the use of **Deep Reinforcement Learning** to dynamically learn bidding decisions based on the current advertising environment.

The DQN agent:

1. Observes the current state
2. Selects a bidding action
3. Participates in a simulated auction
4. Receives a reward based on the outcome
5. Stores the experience in replay memory
6. Updates the neural network
7. Gradually improves its bidding policy

---

## 🎯 Project Objective

The primary objective is to build an intelligent bidding system that can learn **when and how aggressively to bid** instead of relying entirely on manually defined bidding rules.

The project combines:

**Reinforcement Learning + Deep Learning + Real-Time Bidding + Simulation + Interactive Visualization**

---

## ✨ Key Features

- 🧠 **Deep Q-Network (DQN)** for bidding decisions
- 🎯 **Personalized bidding** based on the advertising state
- 💰 **Real-time auction simulation**
- 🔄 **Experience replay** for stable DQN training
- 📉 **Epsilon-greedy exploration**
- ⚡ **Live auction simulation**
- 📊 **Interactive Streamlit dashboard**
- 📈 **Reward and CTR monitoring**
- 💵 **Revenue and ROI tracking**
- 👥 **Bidder-level performance analytics**
- 💾 **DQN model saving and loading**
- 🖼️ **Active campaign visualization**
- 📡 **Real-time auction activity feed**

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │    User / Ad State  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   RTB Environment   │
                         │                     │
                         │  State + Auction    │
                         │      Simulation     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      DQN Agent      │
                         │                     │
                         │   Neural Network    │
                         │   Q-Value Estimator │
                         └──────────┬──────────┘
                                    │
                                    ▼
                              Bid Action
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Ad Auction      │
                         └──────────┬──────────┘
                                    │
                          ┌─────────┴─────────┐
                          ▼                   ▼
                    Auction Result        Click / No Click
                          │                   │
                          └─────────┬─────────┘
                                    ▼
                                  Reward
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Replay Memory     │
                         │                     │
                         │ (s,a,r,s',done)     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    DQN Training     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         Improved Bidding Policy

# 🧠 Reinforcement Learning Approach

The system uses Deep Q-Learning to learn bidding actions.

The DQN estimates the expected future reward for each available action:

Q(state, action)

The agent selects actions using an epsilon-greedy policy, balancing exploration of new actions with exploitation of learned bidding behavior.

🔹 State

The RL environment provides the agent with a representation of the current advertising opportunity.

The state describes the conditions under which the agent must make its bidding decision.

Current Environment
        │
        ▼
     State
        │
        ▼
    DQN Agent
🔹 Action

The DQN selects an action from the available action space.

For dashboard visualization, the actions are represented as simulated bidders:

Action 0 → Bidder-1
Action 1 → Bidder-2
Action 2 → Bidder-3
Action 3 → Bidder-4

The selected action determines the bidding behavior within the simulated auction.

🔹 Reward

After an auction is executed, the environment returns a reward based on the resulting outcome.

The reward provides feedback to the agent about the quality of its decision.

Bid Decision
     │
     ▼
Auction Outcome
     │
     ▼
   Reward
     │
     ▼
DQN Learning

This allows the agent to learn from both successful and unsuccessful bidding decisions.

# 🔄 Experience Replay

The agent stores its experiences in replay memory.

Each experience follows:

(state, action, reward, next_state, done)

During training, random batches of experiences are sampled from memory.

This helps reduce the correlation between consecutive training samples and improves the stability of DQN training.

Environment
     │
     ▼
Experience
     │
     ▼
Replay Memory
     │
     ▼
Random Batch
     │
     ▼
DQN Training

# 📉 Epsilon-Greedy Exploration

The agent uses an epsilon-greedy strategy to balance exploration and exploitation.

At the beginning of training:

Epsilon = 1.0

The agent therefore explores different bidding actions extensively.

As training progresses, epsilon decreases.

High Exploration
       │
       ▼
Auction Experience
       │
       ▼
Replay Memory
       │
       ▼
DQN Training
       │
       ▼
Epsilon Decay
       │
       ▼
More Exploitation
       │
       ▼
Improved Decisions

The dashboard visualizes epsilon decay so that the learning process can be observed in real time.

## 📁 Project Structure
RL_AD/
│
├── dashboard.py
│   └── Main Streamlit dashboard
│
├── engine/
│   ├── __init__.py
│   │
│   ├── agent.py
│   │   └── DQN neural network and RL agent
│   │
│   └── environment.py
│       └── Real-time bidding RL environment
│
├── outputs/
│   └── campaign.jpg
│       └── Active advertising campaign image
│
├── .gitignore
├── requirements.txt
└── README.md

Generated runtime files such as .pkl files and trained model files are excluded from version control using .gitignore.

# 🖥️ Interactive Dashboard

The project includes a Streamlit-based AdBid Intelligence dashboard designed to monitor the DQN agent and simulated RTB environment.

Dashboard Sections
Section	Description
⚡ Global Metrics	Revenue, ROI, CTR, reward and epsilon
🎯 Active Campaign	Displays the current advertising campaign
🧠 DQN Console	Agent status and training information
📈 Reward Graph	Tracks reward progression
📉 Epsilon Decay	Visualizes exploration reduction
🎯 CTR Trend	Tracks click-through performance
🏷️ Live Auction	Runs simulated real-time auctions
📡 Auction Feed	Displays recent auction events
📊 Performance Trends	Visualizes system performance
🧠 Model Intelligence	Displays DQN learning information
👥 Bidder Analytics	Compares bidder performance
📊 Dashboard Metrics

The dashboard monitors important advertising and reinforcement learning metrics.

💰 Revenue

Tracks the revenue generated from successful advertising interactions.

📈 ROI

Measures the return generated relative to advertising spend.

🎯 CTR

Measures the percentage of impressions resulting in clicks.

🧠 Total Reward

Represents the cumulative reward received by the RL agent.

📉 Epsilon

Shows the current exploration rate of the DQN agent.

💵 Bid

Tracks the bidding behavior of the agent during auctions.

🏆 Auction Win Rate

Measures how frequently the bidding strategy wins simulated auctions.

📉 Training Loss

Tracks the DQN optimization process during training.

⚡ Live Auction Simulation

The dashboard includes a live auction environment where bidding decisions can be observed in real time.

Each auction generates information such as:

Auction
   │
   ├── Bid Amount
   ├── Selected Action
   ├── Bidder
   ├── Win / Loss
   ├── Click
   └── Reward

The auction feed allows the bidding process to be monitored as it happens.

# 🔬 Model Learning Process

The training process follows:

1. Initialize Environment
          ↓
2. Initialize DQN Agent
          ↓
3. Observe Current State
          ↓
4. Select Action
          ↓
5. Execute Auction
          ↓
6. Receive Reward
          ↓
7. Store Experience
          ↓
8. Sample Replay Batch
          ↓
9. Train DQN
          ↓
10. Decay Epsilon
          ↓
11. Repeat

Over time, the agent learns to make increasingly informed bidding decisions.

# ▶️ Quick Start
1. Clone the Repository
git clone <your-github-repository-url>
cd RL_AD
2. Create a Virtual Environment
python -m venv .venv
3. Activate the Environment
Windows PowerShell
.venv\Scripts\Activate.ps1
4. Install Dependencies
pip install -r requirements.txt
5. Launch the Dashboard
streamlit run dashboard.py

The dashboard will open in your browser.

# 💾 Model Saving & Loading

The dashboard supports saving and loading the trained DQN model.

A trained model can be stored locally as:

ad_agent.pth

Model files are excluded from GitHub using .gitignore.

This keeps large generated model files out of the repository while allowing the application to train or load models locally.

# 🛠️ Technologies Used
Technology	Purpose
Python	Core development
PyTorch	DQN and neural network
Streamlit	Interactive dashboard
Plotly	Interactive visualizations
NumPy	Numerical computation
Reinforcement Learning	Adaptive bidding
Git	Version control
GitHub	Project hosting

#📈 Key Performance Indicators

The system focuses on the following indicators:

💰 Revenue
📈 ROI
🎯 CTR
🧠 Total Reward
💵 Bid Amount
🏆 Auction Win Rate
📉 Training Loss
📉 Epsilon
👥 Bidder Performance

These metrics provide insight into both business performance and RL model behavior.

# 🎯 Why Reinforcement Learning?

Traditional bidding systems often rely on predefined rules or static bidding strategies.

Reinforcement Learning provides an alternative approach where the agent can learn from its interaction with the environment.

Instead of explicitly defining the optimal bidding strategy, the system allows the DQN agent to discover actions that maximize long-term reward.

Traditional Approach
Traditional Approach
        │
        ▼
Predefined Rules
        │
        ▼
Fixed Decisions
Reinforcement Learning
Reinforcement Learning
        │
        ▼
Environment Feedback
        │
        ▼
Learned Policy
        │
        ▼
Adaptive Decisions
📌 Learning Outcomes

This project demonstrates practical experience with:

Reinforcement Learning
Deep Q-Networks
PyTorch neural networks
Experience Replay
Epsilon-Greedy Exploration
Environment-Agent interaction
Real-Time Bidding concepts
Auction simulation
Performance monitoring
Interactive ML dashboards
Model persistence
Git and GitHub project management
🔮 Future Improvements

Potential extensions include:

Integration with larger real-world advertising datasets
More advanced user personalization
Budget-aware bidding
Dynamic budget allocation
Multi-agent reinforcement learning
Contextual bandit comparison
Advanced auction mechanisms
Online model updating
Real-time data streaming
REST API deployment
Cloud deployment
Production-scale RTB integration

# 📄 License
MIT License