# ============================================================
# ⚡ ADBID INTELLIGENCE
# Deep Q-Network Optimization Console
# ============================================================

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from engine.environment import RTBEnv
from engine.agent import DQNAgent


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AdBid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Orbitron:wght@500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
    }

    .stApp {
        background: #050914;
        color: #E6F1FF;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1450px;
    }

    header {
        background: transparent !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 32px;
        font-weight: 800;
        letter-spacing: 2px;
        color: #64FFDA;
        margin-bottom: 3px;
    }

    .subtitle {
        color: #52709A;
        font-size: 10px;
        letter-spacing: 2px;
        margin-bottom: 20px;
    }


    /* ========================================================
       SECTION TITLES
       ======================================================== */

    .section-title {
        font-family: 'Orbitron', sans-serif;
        color: #64FFDA;
        font-size: 16px;
        letter-spacing: 1px;
        margin-top: 28px;
        margin-bottom: 14px;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetric"] {
        background: #10192C;
        border: 1px solid #243758;
        border-radius: 6px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #5775A5 !important;
        font-size: 9px !important;
        letter-spacing: 1px;
    }

    [data-testid="stMetricValue"] {
        color: #E6F1FF !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 22px !important;
    }


    /* ========================================================
       PANELS
       ======================================================== */

    .panel {
        background: #080F1F;
        border: 1px solid #243758;
        border-radius: 6px;
        padding: 16px;
    }

    .panel-title {
        color: #5274A6;
        font-size: 10px;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }


    /* ========================================================
       CAMPAIGN
       ======================================================== */

    .campaign {
        background: linear-gradient(
            180deg,
            #15233B 0%,
            #101A2C 100%
        );

        border: 1px solid #243758;
        border-radius: 6px;
        padding: 14px;
        overflow: hidden;
    }

    .campaign-image {
        height: 145px;
        border: 1px solid #243758;
        background: #0A1221;
        margin-bottom: 12px;
        overflow: hidden;
        border-radius: 4px;
    }

    .campaign-title {
        color: #E6F1FF;
        font-family: 'Orbitron', sans-serif;
        font-size: 14px;
        letter-spacing: 1px;
        margin-top: 5px;
    }

    .campaign-small {
        color: #56709A;
        font-size: 9px;
        margin-top: 8px;
        letter-spacing: 0.5px;
    }


    /* ========================================================
       STATUS
       ======================================================== */

    .status-running {
        color: #00FF88;
        font-size: 10px;
        font-weight: bold;
    }

    .status-stopped {
        color: #FFB703;
        font-size: 10px;
        font-weight: bold;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        width: 100%;
        background: #2856D9;
        color: white;
        border: none;
        border-radius: 4px;
        min-height: 40px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: #3670F5;
        color: white;
    }


    /* ========================================================
       FEED
       ======================================================== */

    .feed {
        background: #050A15;
        border: 1px solid #243758;
        border-radius: 5px;
        padding: 8px;
        height: 350px;
        overflow-y: auto;
    }

    .feed-line {
        color: #5EA1FF;
        font-size: 8px;
        padding: 8px 3px;
        border-bottom: 1px solid #18253B;
    }


    /* ========================================================
       INFO
       ======================================================== */

    .info-box {
        background: #080F1F;
        border: 1px solid #243758;
        border-radius: 5px;
        padding: 12px;
        color: #7893BA;
        font-size: 9px;
        line-height: 1.7;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "env" not in st.session_state:
    st.session_state.env = RTBEnv()


if "agent" not in st.session_state:

    st.session_state.agent = DQNAgent(
        st.session_state.env.state_dim,
        len(st.session_state.env.action_space)
    )

    # Start from BABY BRAIN
    st.session_state.agent.epsilon = 1.0


if "training_running" not in st.session_state:
    st.session_state.training_running = False


if "auction_running" not in st.session_state:
    st.session_state.auction_running = False


if "training_step" not in st.session_state:
    st.session_state.training_step = 0


if "auction_step" not in st.session_state:
    st.session_state.auction_step = 0


if "training_history" not in st.session_state:

    st.session_state.training_history = pd.DataFrame(
        columns=[
            "Step",
            "Reward",
            "Epsilon",
            "CTR",
            "WinRate",
            "Bid",
            "Loss"
        ]
    )


if "auction_history" not in st.session_state:

    st.session_state.auction_history = pd.DataFrame(
        columns=[
            "Auction",
            "Bid",
            "Reward",
            "Clicked",
            "Won",
            "Revenue",
            "Bidder"
        ]
    )


if "auction_feed" not in st.session_state:
    st.session_state.auction_feed = []


# ============================================================
# REFERENCES
# ============================================================

env = st.session_state.env
agent = st.session_state.agent


# ============================================================
# HELPER
# ============================================================

def rolling_average(series, window=20):

    if len(series) == 0:
        return series

    return (
        pd.Series(series)
        .rolling(window)
        .mean()
        .bfill()
    )


def line_chart(
    x,
    y,
    title,
    y_title="",
    height=280
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(width=2),
            fill="tozeroy"
        )
    )

    fig.update_layout(
        title={
            "text": title,
            "font": {
                "size": 11,
                "color": "#5274A6"
            }
        },
        height=height,
        margin=dict(
            l=45,
            r=15,
            t=40,
            b=35
        ),
        paper_bgcolor="#080F1F",
        plot_bgcolor="#080F1F",
        font=dict(
            family="JetBrains Mono",
            size=9,
            color="#7893BA"
        ),
        xaxis=dict(
            gridcolor="#17253D"
        ),
        yaxis=dict(
            title=y_title,
            gridcolor="#17253D"
        ),
        showlegend=False
    )

    return fig


# ============================================================
# TRAINING STEP
# ============================================================

def training_step():

    state = env.current_state

    # DQN chooses action
    action_index = agent.select_action(state)

    # Environment step
    next_state, reward, done, info = env.step(
        action_index
    )

    # Save experience
    agent.memory.append(
        (
            state,
            action_index,
            reward,
            next_state,
            done
        )
    )

    # Train DQN
    loss = agent.train_step(
        batch_size=32
    )

    st.session_state.training_step += 1

    step = st.session_state.training_step

    bid = float(
        env.action_space[action_index]
    )

    clicked = int(
        info.get(
            "clicked",
            0
        )
    )

    won = int(
        info.get(
            "won",
            reward > 0
        )
    )

    epsilon = float(
        getattr(
            agent,
            "epsilon",
            1.0
        )
    )

    new_row = pd.DataFrame(
        [{
            "Step": step,
            "Reward": float(reward),
            "Epsilon": epsilon,
            "CTR": clicked,
            "WinRate": won,
            "Bid": bid,
            "Loss": float(loss)
        }]
    )

    st.session_state.training_history = pd.concat(
        [
            st.session_state.training_history,
            new_row
        ],
        ignore_index=True
    )


# ============================================================
# AUCTION STEP
# ============================================================

def auction_step():

    st.session_state.auction_step += 1

    auction = st.session_state.auction_step

    state = env.current_state

    action_index = agent.select_action(state)

    next_state, reward, done, info = env.step(
        action_index
    )

    bid = float(
        env.action_space[action_index]
    )

    clicked = int(
        info.get(
            "clicked",
            0
        )
    )

    won = int(
        info.get(
            "won",
            reward > 0
        )
    )

    revenue = bid if won else 0

    bidder = (
        f"Bidder-{(action_index % 4) + 1}"
    )

    new_row = pd.DataFrame(
        [{
            "Auction": auction,
            "Bid": bid,
            "Reward": float(reward),
            "Clicked": clicked,
            "Won": won,
            "Revenue": revenue,
            "Bidder": bidder
        }]
    )

    st.session_state.auction_history = pd.concat(
        [
            st.session_state.auction_history,
            new_row
        ],
        ignore_index=True
    )

    # Feed
    status = "WON" if won else "LOST"

    feed_text = (
        f"#{auction:04d}  |  "
        f"{bidder:<10} |  "
        f"BID ₹{bid:.0f}  |  "
        f"{status:<4}  |  "
        f"REWARD {reward:.2f}"
    )

    st.session_state.auction_feed.insert(
        0,
        feed_text
    )

    st.session_state.auction_feed = (
        st.session_state.auction_feed[:20]
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '⚡ ADBID INTELLIGENCE'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'PERSONALIZED REAL-TIME BIDDING INTELLIGENCE CONSOLE'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# GLOBAL METRICS
# ============================================================

train_df = st.session_state.training_history
auction_df = st.session_state.auction_history


if len(train_df) > 0:

    total_reward = train_df["Reward"].sum()

    ctr = (
        train_df["CTR"].mean()
        * 100
    )

    win_rate = (
        train_df["WinRate"].mean()
        * 100
    )

    total_spend = train_df["Bid"].sum()

else:

    total_reward = 0
    ctr = 0
    win_rate = 0
    total_spend = 0


if len(auction_df) > 0:

    revenue = auction_df["Revenue"].sum()

else:

    revenue = 0


roi = (
    ((revenue - total_spend) / total_spend) * 100
    if total_spend > 0
    else 0
)


epsilon = float(
    getattr(
        agent,
        "epsilon",
        1.0
    )
)


m1, m2, m3, m4, m5 = st.columns(5)


m1.metric(
    "REVENUE",
    f"₹{revenue:,.0f}"
)

m2.metric(
    "ROI",
    f"{roi:.1f}%"
)

m3.metric(
    "CTR",
    f"{ctr:.2f}%"
)

m4.metric(
    "TOTAL REWARD",
    f"{total_reward:.2f}"
)

m5.metric(
    "EPSILON",
    f"{epsilon:.3f}"
)


# ============================================================
# MAIN DASHBOARD
# ============================================================

left, center, right = st.columns(
    [1, 2, 1],
    gap="large"
)


# ============================================================
# LEFT
# ACTIVE CAMPAIGN
# ============================================================

with left:

    st.markdown(
        '<div class="panel-title">'
        'ACTIVE CAMPAIGN'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # CAMPAIGN CARD
    # --------------------------------------------------------

    st.markdown(
        '<div class="campaign">',
        unsafe_allow_html=True
    )

    # Actual campaign image
    st.image(
        "outputs/campaign.jpg",
        width="stretch"
    )

    st.markdown(
        """
        <div class="campaign-title">
            IPHONE 15 PRO
        </div>

        <div class="campaign-small">
            PERSONALIZED AD BIDDING
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # ========================================================
    # MODEL EXPLORATION
    # ========================================================

    st.markdown(
        '<div class="panel-title">'
        'MODEL EXPLORATION'
        '</div>',
        unsafe_allow_html=True
    )

    st.progress(
        epsilon
    )

    st.caption(
        f"EXPLORATION: {epsilon * 100:.1f}%"
    )

    st.caption(
        f"EXPLOITATION: {(1 - epsilon) * 100:.1f}%"
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # ========================================================
    # TRAINING CONTROL
    # ========================================================

    if not st.session_state.training_running:

        if st.button(
            "▶ START TRAINING",
            key="start_training"
        ):

            st.session_state.training_running = True

            st.rerun()

    else:

        if st.button(
            "⏸ STOP TRAINING",
            key="stop_training"
        ):

            st.session_state.training_running = False

            st.rerun()


    if st.session_state.training_running:

        st.markdown(
            '<p class="status-running">'
            '● DQN TRAINING ACTIVE'
            '</p>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<p class="status-stopped">'
            '● TRAINING PAUSED'
            '</p>',
            unsafe_allow_html=True
        )


    # ========================================================
    # MODEL SAVE / LOAD
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    s1, s2 = st.columns(2)

    with s1:

        if st.button(
            "SAVE MODEL",
            key="save_model"
        ):

            agent.save(
                "ad_agent.pth"
            )

            st.success(
                "Model saved"
            )


    with s2:

        if st.button(
            "LOAD MODEL",
            key="load_model"
        ):

            if agent.load(
                "ad_agent.pth"
            ):

                st.success(
                    "Model loaded"
                )

            else:

                st.warning(
                    "No saved model found"
                )


# ============================================================
# CENTER
# DQN CONSOLE
# ============================================================

with center:

    st.markdown(
        '<div class="section-title">'
        '🧠 DEEP Q-NETWORK OPTIMIZATION CONSOLE'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "TRAINING STEPS",
        len(train_df)
    )

    c2.metric(
        "EXPLORATION",
        f"{epsilon * 100:.1f}%"
    )

    c3.metric(
        "EXPLOITATION",
        f"{(1 - epsilon) * 100:.1f}%"
    )


    # ========================================================
    # REWARD GRAPH
    # ========================================================

    if len(train_df) > 0:

        fig = line_chart(
            train_df["Step"],
            rolling_average(
                train_df["Reward"],
                20
            ),
            "REAL-TIME REWARD HISTORY",
            "Rolling Reward",
            280
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="reward_chart"
        )

    else:

        st.markdown(
            """
            <div class="panel"
                 style="
                 height:280px;
                 display:flex;
                 align-items:center;
                 justify-content:center;
                 color:#315078;
                 ">
                PRESS START TRAINING TO BEGIN DQN LEARNING
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # EPSILON + CTR
    # ========================================================

    g1, g2 = st.columns(2)


    with g1:

        if len(train_df) > 0:

            fig = line_chart(
                train_df["Step"],
                train_df["Epsilon"],
                "EPSILON DECAY",
                "Epsilon",
                240
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key="epsilon_chart"
            )

        else:

            st.info(
                "ε starts at 1.000"
            )


    with g2:

        if len(train_df) > 0:

            ctr_series = (
                train_df["CTR"]
                .rolling(20)
                .mean()
                .bfill()
                * 100
            )

            fig = line_chart(
                train_df["Step"],
                ctr_series,
                "CLICK-THROUGH RATE TREND",
                "CTR %",
                240
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key="ctr_chart"
            )

        else:

            st.info(
                "CTR trend will appear during training."
            )


# ============================================================
# RIGHT
# LIVE AUCTION
# ============================================================

with right:

    st.markdown(
        '<div class="panel-title">'
        '🏷️ LIVE AUCTION'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="panel-title">'
        'REAL-TIME AUCTION FEED'
        '</div>',
        unsafe_allow_html=True
    )

    feed_html = '<div class="feed">'


    if len(
        st.session_state.auction_feed
    ) == 0:

        for _ in range(12):

            feed_html += (
                '<div class="feed-line">'
                'WAITING FOR AUCTION...'
                '</div>'
            )

    else:

        for item in st.session_state.auction_feed:

            feed_html += (
                f'<div class="feed-line">'
                f'{item}'
                f'</div>'
            )


    feed_html += "</div>"


    st.markdown(
        feed_html,
        unsafe_allow_html=True
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # ========================================================
    # AUCTION CONTROL
    # ========================================================

    if not st.session_state.auction_running:

        if st.button(
            "▶ START AUCTION",
            key="start_auction"
        ):

            st.session_state.auction_running = True

            st.rerun()

    else:

        if st.button(
            "⏸ STOP AUCTION",
            key="stop_auction"
        ):

            st.session_state.auction_running = False

            st.rerun()


    if st.session_state.auction_running:

        st.markdown(
            '<p class="status-running">'
            '● AUCTION LIVE'
            '</p>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<p class="status-stopped">'
            '● AUCTION PAUSED'
            '</p>',
            unsafe_allow_html=True
        )


# ============================================================
# PERFORMANCE TRENDS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📈 PERFORMANCE TRENDS'
    '</div>',
    unsafe_allow_html=True
)


p1, p2 = st.columns(2)


# ============================================================
# REVENUE
# ============================================================

with p1:

    if len(auction_df) > 0:

        cumulative_revenue = (
            auction_df["Revenue"]
            .cumsum()
        )

        fig = line_chart(
            auction_df["Auction"],
            cumulative_revenue,
            "CUMULATIVE REVENUE",
            "Revenue ₹",
            280
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="revenue_chart"
        )

    else:

        st.info(
            "Start auction to generate revenue."
        )


# ============================================================
# WIN RATE
# ============================================================

with p2:

    if len(train_df) > 0:

        win_series = (
            train_df["WinRate"]
            .rolling(20)
            .mean()
            .bfill()
            * 100
        )

        fig = line_chart(
            train_df["Step"],
            win_series,
            "WIN RATE TREND",
            "Win Rate %",
            280
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="winrate_chart"
        )

    else:

        st.info(
            "Start training to generate win-rate data."
        )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🧠 MODEL INTELLIGENCE'
    '</div>',
    unsafe_allow_html=True
)


if len(train_df) > 0:

    a1, a2, a3, a4 = st.columns(4)

    avg_reward = train_df["Reward"].mean()

    best_reward = train_df["Reward"].max()

    avg_bid = train_df["Bid"].mean()

    avg_loss = train_df["Loss"].mean()


    a1.metric(
        "AVERAGE REWARD",
        f"{avg_reward:.3f}"
    )

    a2.metric(
        "BEST REWARD",
        f"{best_reward:.3f}"
    )

    a3.metric(
        "AVERAGE BID",
        f"₹{avg_bid:.0f}"
    )

    a4.metric(
        "AVERAGE LOSS",
        f"{avg_loss:.5f}"
    )


    m1, m2 = st.columns(2)


    # ========================================================
    # LEARNING CURVE
    # ========================================================

    with m1:

        fig = line_chart(
            train_df["Step"],
            rolling_average(
                train_df["Reward"],
                30
            ),
            "DQN LEARNING CURVE",
            "Rolling Reward",
            300
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="learning_curve"
        )


    # ========================================================
    # LOSS
    # ========================================================

    with m2:

        fig = line_chart(
            train_df["Step"],
            rolling_average(
                train_df["Loss"],
                30
            ),
            "DQN TRAINING LOSS",
            "Loss",
            300
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="loss_chart"
        )


    m3, m4 = st.columns(2)


    # ========================================================
    # EXPLORATION VS EXPLOITATION
    # ========================================================

    with m3:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=train_df["Step"],
                y=train_df["Epsilon"] * 100,
                mode="lines",
                name="Exploration"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=train_df["Step"],
                y=(1 - train_df["Epsilon"]) * 100,
                mode="lines",
                name="Exploitation"
            )
        )

        fig.update_layout(
            title={
                "text": "EXPLORATION VS EXPLOITATION",
                "font": {
                    "size": 11,
                    "color": "#5274A6"
                }
            },
            height=300,
            paper_bgcolor="#080F1F",
            plot_bgcolor="#080F1F",
            font=dict(
                family="JetBrains Mono",
                size=9,
                color="#7893BA"
            ),
            xaxis=dict(
                gridcolor="#17253D"
            ),
            yaxis=dict(
                title="Percentage",
                gridcolor="#17253D"
            )
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="exploration_chart"
        )


    # ========================================================
    # BID DISTRIBUTION
    # ========================================================

    with m4:

        fig = px.histogram(
            train_df,
            x="Bid",
            nbins=15,
            title="DQN BID DISTRIBUTION"
        )

        fig.update_layout(
            height=300,
            paper_bgcolor="#080F1F",
            plot_bgcolor="#080F1F",
            font=dict(
                family="JetBrains Mono",
                size=9,
                color="#7893BA"
            ),
            xaxis=dict(
                gridcolor="#17253D"
            ),
            yaxis=dict(
                gridcolor="#17253D"
            )
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="bid_distribution"
        )


else:

    st.markdown(
        """
        <div class="info-box">

        MODEL INTELLIGENCE IS WAITING FOR TRAINING.

        The DQN begins with:

        ε = 1.000

        This represents maximum exploration — the
        "baby brain" stage.

        As training progresses:

        ε ↓
        Exploration ↓
        Exploitation ↑
        Reward should improve
        Bid strategy should stabilize
        CTR / Win Rate can improve

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# BIDDER ANALYTICS
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📊 BIDDER ANALYTICS'
    '</div>',
    unsafe_allow_html=True
)


if len(auction_df) > 0:

    b1, b2, b3, b4 = st.columns(4)

    b1.metric(
        "TOTAL AUCTIONS",
        len(auction_df)
    )

    b2.metric(
        "WIN RATE",
        f"{auction_df['Won'].mean() * 100:.1f}%"
    )

    b3.metric(
        "AVG BID",
        f"₹{auction_df['Bid'].mean():.0f}"
    )

    b4.metric(
        "TOTAL REVENUE",
        f"₹{auction_df['Revenue'].sum():,.0f}"
    )


    # ========================================================
    # BIDDER PERFORMANCE
    # ========================================================

    bidder_perf = (
        auction_df
        .groupby("Bidder")
        .agg(
            Auctions=("Auction", "count"),
            Win_Rate=("Won", "mean"),
            Avg_Bid=("Bid", "mean"),
            Revenue=("Revenue", "sum"),
            Avg_Reward=("Reward", "mean")
        )
        .reset_index()
    )

    bidder_perf["Win_Rate"] *= 100


    c1, c2 = st.columns(2)


    # ========================================================
    # WIN RATE BY BIDDER
    # ========================================================

    with c1:

        fig = px.bar(
            bidder_perf,
            x="Bidder",
            y="Win_Rate",
            title="BIDDER WIN-RATE COMPARISON"
        )

        fig.update_layout(
            height=300,
            paper_bgcolor="#080F1F",
            plot_bgcolor="#080F1F",
            font=dict(
                family="JetBrains Mono",
                size=9,
                color="#7893BA"
            )
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="bidder_winrate"
        )


    # ========================================================
    # REVENUE BY BIDDER
    # ========================================================

    with c2:

        fig = px.bar(
            bidder_perf,
            x="Bidder",
            y="Revenue",
            title="REVENUE CONTRIBUTION BY BIDDER"
        )

        fig.update_layout(
            height=300,
            paper_bgcolor="#080F1F",
            plot_bgcolor="#080F1F",
            font=dict(
                family="JetBrains Mono",
                size=9,
                color="#7893BA"
            )
        )

        st.plotly_chart(
            fig,
            width="stretch",
            key="bidder_revenue"
        )


    # ========================================================
    # LIVE BID TREND
    # ========================================================

    fig = line_chart(
        auction_df["Auction"],
        auction_df["Bid"],
        "LIVE BID TREND",
        "Bid ₹",
        300
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="live_bid_trend"
    )


    # ========================================================
    # BIDDER TABLE
    # ========================================================

    st.dataframe(
        bidder_perf.style.format(
            {
                "Win_Rate": "{:.2f}%",
                "Avg_Bid": "₹{:.0f}",
                "Revenue": "₹{:,.0f}",
                "Avg_Reward": "{:.3f}"
            }
        ),
        width="stretch",
        hide_index=True
    )


else:

    st.info(
        "Start Live Auction to generate bidder analytics."
    )


# ============================================================
# RUN TRAINING
# ============================================================

if st.session_state.training_running:

    training_step()

    time.sleep(0.08)

    st.rerun()


# ============================================================
# RUN AUCTION
# ============================================================

if st.session_state.auction_running:

    auction_step()

    time.sleep(0.10)

    st.rerun()