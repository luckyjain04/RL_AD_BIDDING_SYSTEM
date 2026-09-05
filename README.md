# ⚡ Personalized AI Ad Bidding System

A **Reinforcement Learning-based Real-Time Bidding (RTB) system** that uses a **Deep Q-Network (DQN)** to learn intelligent and personalized advertising bidding decisions.

The system simulates real-time advertising auctions, allows an RL agent to learn from auction outcomes, and provides an interactive **Streamlit dashboard** for monitoring model learning, bidding behavior, rewards, CTR, ROI, and auction performance.

---

## 🚀 Overview

In real-time advertising, advertisers must decide **how much to bid for each individual ad impression** within milliseconds. A bidding strategy that bids too aggressively can waste the advertising budget, while bidding too conservatively can result in missed opportunities.

This project explores the use of **Deep Reinforcement Learning** to dynamically learn bidding decisions based on the current advertising environment.

### Iterative Learning Cycle

1. Observes the current environment state.
2. Selects a bidding action via an epsilon-greedy policy.
3. Participates in a simulated real-time auction.
4. Receives a reward based on the auction outcome and ad interaction (click vs. no click).
5. Stores the experience tuple $(s, a, r, s', \text{done})$ in replay memory.
6. Samples random mini-batches from memory to break sample correlation.
7. Updates the neural network weights via Q-learning gradient steps.
8. Gradually optimizes its policy toward high ROI and CTR.

---

## 🎯 Project Objective

The primary objective is to build an intelligent bidding system that can learn **when and how aggressively to bid** instead of relying entirely on manually defined bidding rules.

The project combines **Reinforcement Learning + Deep Learning + Real-Time Bidding + Simulation + Interactive Visualization**.

---

## ✨ Key Features

* 🧠 **Deep Q-Network (DQN):** Dynamically estimates expected future rewards for discrete bidding actions.
* 🎯 **State-Aware Bidding:** Personalizes bid amounts according to user attributes and environment signals.
* 🔄 **Experience Replay & Epsilon Decay:** Ensures stable offline training and balances exploration vs. exploitation.
* ⚡ **Live Auction Simulation:** Simulates multi-bidder, second-price (or first-price) auction environments in real time.
* 📊 **Streamlit Monitoring Suite:** Provides live visual tracking of revenue, total reward, loss curves, and epsilon decay.
* 👥 **Bidder Analytics:** Tracks win rates and return-on-ad-spend across individual bidding agents.
* 💾 **Model Persistence:** Easily save and reload trained PyTorch PyTorch model weights (`.pth`).

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
                         │ (s, a, r, s', done) │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    DQN Training     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         Improved Bidding Policy

```

---

## 🧠 Reinforcement Learning Formulation

The DQN models the optimal action-value function $Q^*(s, a)$ representing the maximum expected future reward attainable by taking action $a$ in state $s$.

### State Space ($s$)

Represents real-time context vector including user demographic profile, historical click-through rate, budget remaining, time of day, and publisher context.

### Action Space ($a$)

Determines the bid intensity or discrete action selected by the agent:

* **Action 0:** Bidder-1 (Conservative)
* **Action 1:** Bidder-2 (Moderate)
* **Action 2:** Bidder-3 (Aggressive)
* **Action 3:** Bidder-3 (Premium Context)

### Reward Function ($r$)

Calculated post-auction based on win status, impression cost, and user response:


$$r = \begin{cases}  \text{Value} - \text{Cost} & \text{if Auction Won and User Clicks} \\ -\text{Cost} & \text{if Auction Won and No Click} \\ 0 & \text{if Auction Lost} \end{cases}$$

---

## 📁 Project Structure

```text
RL_AD/
│
├── dashboard.py           # Streamlit application entry point
│
├── engine/
│   ├── __init__.py
│   ├── agent.py           # Deep Q-Network & Experience Replay Buffer
│   └── environment.py     # Custom OpenAI Gym/Gymnasium RTB simulator
│
├── outputs/
│   └── campaign.jpg       # Static visual assets for dashboard interface
│
├── .gitignore
├── requirements.txt
└── README.md

```

---

## 🖥️ Interactive Dashboard Features

| Section | Description |
| --- | --- |
| **⚡ Global Metrics** | Live snapshot of Total Revenue, ROI %, CTR %, Cumulative Reward, and Epsilon |
| **🎯 Active Campaign** | Displays visual asset and configuration for current simulated campaign |
| **🧠 DQN Console** | Model health status, current learning rate, and batch update logs |
| **📈 Learning Curves** | Real-time Plotly charts tracking Reward trends, Loss reduction, and Epsilon decay |
| **🏷️ Live Auction** | Step-by-step manual execution of single auctions to observe agent actions |
| **👥 Bidder Analytics** | Multi-agent comparative performance matrix and share-of-voice charts |

---

## ▶️ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/RL_AD.git
cd RL_AD

```

### 2. Set Up Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate

```

### 3. Install Requirements

```bash
pip install -r requirements.txt

```

### 4. Run the Dashboard

```bash
streamlit run dashboard.py

```

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch
* **Interactive UI:** Streamlit
* **Data Visualization:** Plotly, Matplotlib
* **Environment & Mathematics:** NumPy, Pandas, Gymnasium

---

## 🏁 Future Enhancements

* [ ] Transition from discrete actions to **Continuous Action Spaces** using DDPG or SAC.
* [ ] Implement **Budget-Constrained RTB** (Safety Layer to prevent early budget depletion).
* [ ] Support **Multi-Agent Competitive Auctions** where multiple DQN networks bid against each other.
* [ ] Integration with real-world RTB datasets like **iPinYou** or **Yandex Real-Time Bidding**.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.
