````markdown
# ⚡ Personalized AI Ad Bidding System

A **Reinforcement Learning-based Real-Time Bidding (RTB) system** that uses a **Deep Q-Network (DQN)** to learn adaptive advertising bidding decisions.

The project combines **data simulation, feature engineering, machine learning, reinforcement learning, auction simulation, evaluation, and interactive dashboards**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-DQN-EE4C2C?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Key Features

- 🧠 Deep Q-Network (DQN) for adaptive bidding
- 🎯 State-aware advertising decisions
- 🔄 Experience Replay
- 📉 Epsilon-Greedy exploration
- ⚡ Real-time auction simulation
- 📊 Interactive Streamlit monitoring dashboard
- 🌐 Flask web application
- 📈 CTR, ROI, revenue, reward, and win-rate analytics
- 💾 PyTorch model saving and loading
- 📁 Auction and evaluation result generation

---

## 🏗️ System Architecture

```text
Advertising Data
       │
       ▼
Data Simulation
       │
       ▼
Feature Engineering
       │
   ┌───┴────────────┐
   ▼                ▼
ML Models      RL Environment
                    │
                    ▼
                DQN Agent
                    │
                    ▼
              Bidding Engine
                    │
                    ▼
               RTB Auction
                    │
                    ▼
              Reward / Outcome
                    │
                    ▼
                Evaluation
                    │
              ┌─────┴─────┐
              ▼           ▼
         Streamlit      Flask
         Dashboard     Web App
````

---

## 🧠 Reinforcement Learning

The DQN agent learns bidding decisions through repeated interaction with the simulated RTB environment.

### Learning Cycle

```text
State
  ↓
Select Action
  ↓
Generate Bid
  ↓
Auction
  ↓
Reward
  ↓
Replay Memory
  ↓
DQN Training
  ↓
Improved Policy
```

### DQN Components

| Component     | Description                                  |
| ------------- | -------------------------------------------- |
| State         | Advertising and user environment information |
| Action        | Discrete bidding decision                    |
| Reward        | Feedback based on auction outcome            |
| Replay Memory | Stores previous experiences                  |
| Epsilon       | Controls exploration vs. exploitation        |

### Experience Format

```text
(state, action, reward, next_state, done)
```

---

## 📊 Dashboard

The Streamlit dashboard provides monitoring and analysis of:

| Metric      | Purpose                            |
| ----------- | ---------------------------------- |
| 💰 Revenue  | Advertising revenue                |
| 📈 ROI      | Return on advertising spend        |
| 🎯 CTR      | Click-through rate                 |
| 🧠 Reward   | Reinforcement learning performance |
| 🏆 Win Rate | Auction success rate               |
| 📉 Loss     | DQN training performance           |
| 📉 Epsilon  | Exploration level                  |

### Dashboard Modules

* ⚡ Live Auction
* 🧠 DQN Console
* 📈 Performance Trends
* 📊 Bidder Analytics
* 👤 User Personalization
* 🤖 Model Performance

---

## 📁 Project Structure

```text
RL_AD/
│
├── app.py
├── dashboard.py
├── bidding_engine.py
├── data_simulator.py
├── feature_engineering.py
├── model_trainer.py
├── evaluation.py
├── main.py
│
├── engine/
│   ├── agent.py
│   └── environment.py
│
├── data/
│   └── criteo_sample.tsv
│
├── models/
├── outputs/
│   ├── auction_results.csv
│   └── metrics_summary.csv
│
├── static/
│   ├── script.js
│   └── style.css
│
├── templates/
│   └── index.html
│
├── ad_agent.pth
├── requirements.txt
└── README.md
```

---

## ▶️ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/luckyjain04/RL_AD_BIDDING_SYSTEM.git
cd RL_AD_BIDDING_SYSTEM
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate on Windows

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Dashboard

```bash
streamlit run dashboard.py
```

### 6. Run the Flask Web Application

```bash
python app.py
```

---

## 🛠️ Tech Stack

**Python • PyTorch • Streamlit • Flask • Pandas • NumPy • Scikit-learn • Plotly • HTML • CSS • JavaScript**

---

## 🔮 Future Improvements

* Continuous-action bidding using DDPG / SAC
* Budget-constrained reinforcement learning
* Multi-agent competitive bidding
* Larger real-world RTB datasets
* Online learning from streaming auction data
* Cloud deployment

---

## 📌 Learning Outcomes

**Deep Reinforcement Learning • DQN • PyTorch • Experience Replay • Epsilon-Greedy • Real-Time Bidding • Auction Simulation • Feature Engineering • Machine Learning • Model Evaluation • Streamlit • Flask**

---

## 📄 License

This project is licensed under the **MIT License**.

```
```
