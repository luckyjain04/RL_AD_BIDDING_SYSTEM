# ⚡ Personalized AI Ad Bidding System

A **Reinforcement Learning-based Real-Time Bidding (RTB) system** that uses a **Deep Q-Network (DQN)** to learn intelligent and personalized advertising bidding decisions.

The system simulates real-time ad auctions, trains an RL agent through interaction with the environment, and provides an interactive **Streamlit dashboard** to visualize bidding performance, rewards, CTR, ROI, exploration, and auction activity.

---

## 🚀 Project Highlights

- 🧠 Deep Q-Network (DQN) based bidding agent
- 🎯 Personalized bidding decisions
- 💰 Real-time ad auction simulation
- 📈 Reward-based reinforcement learning
- ⚡ Live auction simulation
- 📊 Interactive Streamlit dashboard
- 📉 Reward and epsilon-decay visualization
- 👥 Bidder performance analytics
- 💾 DQN model saving and loading
- 🖼️ Active advertising campaign visualization

---

## 🏗️ System Architecture

```text
                    USER / AD STATE
                          │
                          ▼
                ┌───────────────────┐
                │   RTB Environment │
                └─────────┬─────────┘
                          │
                          ▼
                   CURRENT STATE
                          │
                          ▼
                ┌───────────────────┐
                │     DQN Agent     │
                │                   │
                │   Neural Network  │
                └─────────┬─────────┘
                          │
                          ▼
                    BID ACTION
                          │
                          ▼
                ┌───────────────────┐
                │    AD AUCTION     │
                └─────────┬─────────┘
                          │
                          ▼
                    AUCTION RESULT
                          │
                          ▼
                      REWARD
                          │
                          ▼
                ┌───────────────────┐
                │  Replay Memory    │
                └─────────┬─────────┘
                          │
                          ▼
                    DQN TRAINING
                          │
                          ▼
                IMPROVED BID POLICY
🧠 Reinforcement Learning

The project uses Deep Q-Learning to learn bidding decisions.

The DQN agent observes the current advertising environment and selects an action using an epsilon-greedy policy.

State

The environment provides the current state representing the advertising opportunity and auction conditions.

Action

The DQN selects a bidding action from the available action space.

The dashboard represents the available actions as simulated bidders:

Action 0 → Bidder-1
Action 1 → Bidder-2
Action 2 → Bidder-3
Action 3 → Bidder-4
Reward

After an auction, the environment returns a reward based on the outcome of the bidding decision.

The experience is stored as:

(state, action, reward, next_state, done)

The DQN learns from these experiences using experience replay.

📉 Epsilon-Greedy Exploration

At the beginning of training, the agent has a high exploration rate.

Epsilon = 1.0

This allows the agent to explore different bidding actions.

As training progresses, epsilon decreases and the agent increasingly relies on its learned policy.

High Exploration
       ↓
Auction Experience
       ↓
Replay Memory
       ↓
DQN Training
       ↓
Epsilon Decay
       ↓
Better Bidding Decisions
📁 Project Structure
RL_AD/
│
├── dashboard.py
│   └── Main Streamlit dashboard
│
├── engine/
│   ├── __init__.py
│   ├── agent.py
│   │   └── DQN neural network and agent
│   │
│   └── environment.py
│       └── Reinforcement learning environment
│
├── outputs/
│   └── campaign.jpg
│       └── Active advertising campaign image
│
├── .gitignore
├── requirements.txt
└── README.md

Generated files such as trained model files and runtime .pkl files are excluded using .gitignore.

🖥️ Streamlit Dashboard

The project includes a futuristic interactive dashboard built using Streamlit.

Dashboard Sections
Section	Description
⚡ Global Metrics	Revenue, ROI, CTR, reward and epsilon
🎯 Active Campaign	Displays the current advertising campaign
🧠 DQN Console	Training status and model information
📈 Reward Graph	Tracks reward during training
📉 Epsilon Decay	Shows exploration reduction
🎯 CTR Trend	Tracks click-through performance
🏷️ Live Auction	Simulates real-time bidding
📡 Auction Feed	Displays recent auction events
📊 Performance Trends	Visualizes training and auction performance
🧠 Model Intelligence	Displays DQN learning information
👥 Bidder Analytics	Compares simulated bidder performance
▶️ Quick Start
1. Clone the Repository
git clone <your-github-repository-url>
cd RL_AD
2. Create a Virtual Environment
python -m venv .venv
3. Activate the Virtual Environment

For Windows PowerShell:

.venv\Scripts\Activate.ps1
4. Install Dependencies
pip install -r requirements.txt
5. Run the Dashboard
streamlit run dashboard.py

The Streamlit dashboard will open in your browser.

💾 Model Saving and Loading

The dashboard provides options to save and load the trained DQN model.

A trained PyTorch model can be saved locally as:

ad_agent.pth

Model files are excluded from GitHub using .gitignore.

📊 Performance Metrics

The dashboard monitors several important RTB and reinforcement learning metrics:

Revenue
ROI
CTR (Click-Through Rate)
Total Reward
Bid Amount
Auction Win Rate
Training Loss
Epsilon
Bidder Performance

These metrics help evaluate how the agent behaves during training and simulated auctions.

🔄 Training Workflow
Initialize Environment
        │
        ▼
Initialize DQN Agent
        │
        ▼
Observe State
        │
        ▼
Select Bid Action
        │
        ▼
Execute Auction
        │
        ▼
Receive Reward
        │
        ▼
Store Experience
        │
        ▼
Sample Replay Batch
        │
        ▼
Train DQN
        │
        ▼
Decay Epsilon
        │
        ▼
Repeat
🛠️ Technologies Used
Technology	Purpose
Python	Core development
PyTorch	DQN and neural network
Streamlit	Interactive dashboard
Plotly	Data visualization
NumPy	Numerical computation
Reinforcement Learning	Adaptive bidding
Git & GitHub	Version control
🎯 Project Objective

Traditional advertising systems often rely on predefined bidding rules.

This project explores how Reinforcement Learning can learn bidding decisions dynamically from auction feedback.

The DQN agent continuously interacts with the simulated advertising environment and learns which bidding actions can produce better rewards.

The project combines:

Artificial Intelligence + Reinforcement Learning + Real-Time Bidding + Data Visualization

🔮 Future Improvements
Integration with larger real-world advertising datasets
More advanced user personalization
Budget-aware bidding
Multi-agent reinforcement learning
Contextual bandit comparison
Advanced auction mechanisms
Online model updating
REST API deployment
Cloud deployment
Real-time advertising data integration
📄 License

MIT License