import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from engine.environment import RTBEnv
from engine.agent import DQNAgent

st.set_page_config(page_title="AdBid Analytics", layout="wide")

if 'env' not in st.session_state:
    st.session_state.env = RTBEnv()
    st.session_state.agent = DQNAgent(st.session_state.env.state_dim, len(st.session_state.env.action_space))
    st.session_state.df = pd.DataFrame(columns=['Step', 'Reward', 'Bid', 'Interest', 'Clicked'])

st.title("⚡ AdBid Deep Analytics")
st.write("Streamlit Data Science Console for Model Auditing")

if st.button("Start AI Analysis"):
    for _ in range(50): # Run 50 quick simulations
        state = st.session_state.env.current_state
        action_idx = st.session_state.agent.select_action(state)
        next_state, reward, done, info = st.session_state.env.step(action_idx)
        st.session_state.agent.memory.append((state, action_idx, reward, next_state, done))
        st.session_state.agent.train_step()
        
        new_row = pd.DataFrame([{'Step': len(st.session_state.df), 'Reward': reward, 'Bid': st.session_state.env.action_space[action_idx], 'Interest': state[0], 'Clicked': int(info['clicked'])}])
        st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)

if not st.session_state.df.empty:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(st.session_state.df['Clicked'].astype(int), st.session_state.df['Interest'])
        fig, ax = plt.subplots(); ax.plot(fpr, tpr); ax.plot([0,1],[0,1], '--'); st.pyplot(fig)
    with col2:
        st.subheader("Bid Distribution")
        fig2, ax2 = plt.subplots(); st.session_state.df['Bid'].value_counts().plot(kind='bar', ax=ax2); st.pyplot(fig2)