"""
dashboard.py
------------
Streamlit dashboard for the Personalized Ad Bidding System.
Launch with:   streamlit run dashboard.py

If main.py hasn't been run yet, the dashboard runs the pipeline
automatically in demo mode (10 000 rows for speed).
"""

import os, sys, pickle, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AdBid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Orbitron:wght@700;900&display=swap');

  html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }

  /* Dark background */
  .stApp { background: #0A0E1A; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #112240 100%);
    border-right: 1px solid #1E3A5F;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: linear-gradient(135deg, #112240 0%, #0D1B2A 100%);
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 16px !important;
  }
  [data-testid="metric-container"] label {
    color: #64FFDA !important; font-size: 11px !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #E6F1FF !important; font-size: 26px !important;
    font-family: 'Orbitron', sans-serif !important;
  }

  /* Section headers */
  h1 { font-family: 'Orbitron', sans-serif !important;
       color: #64FFDA !important; letter-spacing: 2px; }
  h2, h3 { color: #CCD6F6 !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab"] { color: #8892B0; }
  .stTabs [aria-selected="true"] {
    color: #64FFDA !important;
    border-bottom: 2px solid #64FFDA !important;
  }

  /* Dataframe */
  .dataframe { background: #112240 !important; color: #CCD6F6 !important; }

  /* Divider glow */
  hr { border-color: #1E3A5F !important; }

  /* Auction feed card */
  .auction-card {
    background: linear-gradient(90deg, #112240, #0D1B2A);
    border-left: 3px solid #64FFDA;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 6px;
    font-size: 12px;
    color: #A8B2D8;
  }
</style>
""", unsafe_allow_html=True)

DARK_BG  = "#0A0E1A"
PANEL_BG = "#112240"
ACCENT   = "#64FFDA"
TEXT     = "#CCD6F6"
GRID     = "#1E3A5F"
COLORS   = ["#64FFDA", "#FF6B9D", "#FFD166", "#06D6A0", "#118AB2",
            "#EF476F", "#FFB703", "#8ECAE6"]

# ─────────────────────────────────────────────────────────────────────────────
# Load or build state
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_state():
    state_path = Path("outputs/dashboard_state.pkl")
    if state_path.exists():
        with open(state_path, "rb") as f:
            return pickle.load(f)
    # Auto-run pipeline in demo mode
    st.info("⚙️  First run – training models on 20 000 simulated impressions…")
    sys.path.insert(0, str(Path(__file__).parent))
    from data_simulator      import generate_dataset
    from feature_engineering import prepare_dataset
    from model_trainer       import train_all_models
    from bidding_engine      import AuctionEngine, build_default_bidders
    from main                import build_bid_requests

    df = generate_dataset(n_rows=20_000)
    X_tr, X_v, X_te, y_tr, y_v, y_te, pipe, meta = prepare_dataset(df)
    out   = train_all_models(X_tr, y_tr, X_v, y_v, X_te, y_te)
    bids  = build_default_bidders()
    reqs  = build_bid_requests(df, pipe, n=1000)
    eng   = AuctionEngine(bids, out["trained"][out["best"]])
    adf   = eng.run_simulation(reqs, verbose=False)
    state = {
        "trained": out["trained"], "results": out["results"],
        "best_name": out["best"], "pipeline": pipe,
        "meta_test": meta, "X_test": X_te, "y_test": y_te,
        "auction_df": adf, "bidders": bids,
    }
    Path("outputs").mkdir(exist_ok=True)
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    return state


def _fig_defaults(fig):
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
        font_color=TEXT, font_family="JetBrains Mono",
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ AdBid Intelligence")
    st.markdown("*Personalized RTB System*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🧠 Models", "👤 User Personalization",
         "🏷️ Live Auction", "📊 Bidder Analytics"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Data: Criteo Click Logs (simulated)\n"
               "Model: CTR prediction → optimal bid")

state = load_state()
trained   = state["trained"]
results   = state["results"]
best_name = state["best_name"]
pipeline  = state["pipeline"]
meta_test = state["meta_test"]
X_test    = state["X_test"]
y_test    = state["y_test"]
auction_df = state["auction_df"]
bidders    = state["bidders"]
best_model = trained[best_name]


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("⚡ ADBID INTELLIGENCE")
    st.markdown("**Personalized Real-Time Bidding System** — Criteo Click Logs")
    st.divider()

    y_prob = best_model.predict_proba(X_test)
    best_test = results[best_name]["test"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best Model",    best_name.replace("_"," ").title())
    c2.metric("AUC-ROC",       f"{best_test['auc_roc']:.4f}")
    c3.metric("Log-Loss",      f"{best_test['log_loss']:.4f}")
    c4.metric("Lift @ Top 10%",f"{best_test['lift_top10pct']:.2f}×")
    c5.metric("Actual CTR",    f"{best_test['ctr_actual']*100:.2f}%")

    st.divider()
    col_l, col_r = st.columns(2)

    # pCTR distribution
    with col_l:
        st.subheader("Predicted CTR Distribution")
        fig = go.Figure(go.Histogram(
            x=y_prob, nbinsx=60,
            marker_color=ACCENT, opacity=0.8,
            name="pCTR",
        ))
        fig.add_vline(x=y_prob.mean(), line_dash="dash",
                      line_color="#FF6B9D",
                      annotation_text=f"μ={y_prob.mean():.3f}",
                      annotation_font_color="#FF6B9D")
        fig.update_layout(
            xaxis_title="Predicted Click Probability",
            yaxis_title="Count",
            showlegend=False,
        )
        _fig_defaults(fig)
        st.plotly_chart(fig, use_container_width=True)

    # bid price distribution from auction
    with col_r:
        st.subheader("Clearing Price Distribution")
        won = auction_df[auction_df["winner"].notna()]
        fig2 = go.Figure(go.Histogram(
            x=won["clearing_price"], nbinsx=50,
            marker_color="#FFD166", opacity=0.85,
        ))
        fig2.update_layout(
            xaxis_title="Clearing CPM ($)",
            yaxis_title="Count",
            showlegend=False,
        )
        _fig_defaults(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # System architecture diagram
    st.divider()
    st.subheader("System Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                   PERSONALIZED AD BIDDING SYSTEM                │
    │                                                                 │
    │  Criteo Click Logs                                              │
    │      │                                                          │
    │      ▼                                                          │
    │  [Feature Engineering]  ─── I1..I13 (integer) ─┐              │
    │  CriteoFeaturePipeline  ─── C1..C26 (hashed)  ─┼─► X matrix   │
    │                         ─── Interaction terms ─┘              │
    │      │                                                          │
    │      ▼                                                          │
    │  [CTR Model]  LogReg / DecTree / RF / LightGBM                 │
    │      │  pCTR = P(click | user, ad, context)                    │
    │      ▼                                                          │
    │  [Bidding Strategy]                                             │
    │      FixedBidder     bid = const                                │
    │      CTRBidder       bid = base_cpm × pCTR                     │
    │      ValueBidder     bid = pCTR × pConv × value  ← optimal     │
    │      PacingBidder    ValueBidder + budget pacing               │
    │      ThresholdBidder skip if pCTR < threshold                  │
    │      │                                                          │
    │      ▼                                                          │
    │  [Second-Price Auction]  winner pays 2nd-highest bid           │
    └─────────────────────────────────────────────────────────────────┘
    ```
    """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Models
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧠 Models":
    st.title("🧠 CTR PREDICTION MODELS")
    st.divider()

    # Metrics table
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name.replace("_", " ").title(),
            "Val AUC": r["val"]["auc_roc"],
            "Test AUC": r["test"]["auc_roc"],
            "Val AUC-PR": r["val"]["auc_pr"],
            "Test AUC-PR": r["test"]["auc_pr"],
            "Test LogLoss": r["test"]["log_loss"],
            "Lift@10%": r["test"]["lift_top10pct"],
        })
    mdf = pd.DataFrame(rows).sort_values("Test AUC", ascending=False)
    st.dataframe(
        mdf.style.background_gradient(
            cmap="Blues", subset=["Val AUC", "Test AUC", "Test AUC-PR"]
        ).format(precision=4),
        use_container_width=True,
    )

    st.divider()
    tab1, tab2 = st.tabs(["ROC Curves", "Calibration"])

    with tab1:
        fig = go.Figure()
        for i, (name, m) in enumerate(trained.items()):
            yp = m.predict_proba(X_test)
            from sklearn.metrics import roc_curve, auc as sk_auc
            fpr, tpr, _ = roc_curve(y_test, yp)
            a = sk_auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{name.replace('_',' ').title()} (AUC={a:.3f})",
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                mode="lines",
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], line=dict(dash="dot", color="#555"),
            showlegend=False, mode="lines",
        ))
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            title="ROC Curves — All Models",
        )
        _fig_defaults(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        from sklearn.calibration import calibration_curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], line=dict(dash="dot", color="#555"),
            showlegend=False, mode="lines",
        ))
        for i, (name, m) in enumerate(trained.items()):
            yp = m.predict_proba(X_test)
            fp, mp = calibration_curve(y_test, yp, n_bins=15)
            fig.add_trace(go.Scatter(
                x=mp, y=fp,
                name=name.replace("_", " ").title(),
                line=dict(color=COLORS[i], width=2),
                mode="lines+markers",
                marker=dict(size=6),
            ))
        fig.update_layout(
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            title="Calibration Curves",
        )
        _fig_defaults(fig)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: User Personalization
# ─────────────────────────────────────────────────────────────────────────────
elif page == "👤 User Personalization":
    st.title("👤 USER PERSONALIZATION")
    st.divider()

    from data_simulator import USER_PROFILES, AD_CATEGORIES, simulate_criteo_row

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.subheader("Build a User Profile")
        profile_sel = st.selectbox(
            "User Segment", list(USER_PROFILES.keys()),
            format_func=lambda x: x.replace("_", " ").title()
        )
        ad_sel = st.selectbox("Ad Category", AD_CATEGORIES)
        device = st.select_slider(
            "Device", options=["Desktop", "Mobile", "Tablet"]
        )
        hour_of_day = st.slider("Hour of Day", 0, 23, 19)
        ad_position = st.slider("Ad Position", 1, 5, 1)

    with col_right:
        # simulate & predict for this user config
        rng = np.random.default_rng(int(time.time()) % 1000)
        rows_sim = []
        for _ in range(500):
            r = simulate_criteo_row(rng, profile_sel, ad_sel)
            r["I5"] = hour_of_day
            r["I2"] = ad_position
            r["I7"] = ["Desktop", "Mobile", "Tablet"].index(device)
            rows_sim.append(r)
        df_sim = pd.DataFrame(rows_sim)
        X_sim  = pipeline.transform(df_sim)
        probs  = best_model.predict_proba(X_sim)

        mean_ctr = probs.mean()
        from bidding_engine import ValueBidder, BidRequest
        vb   = ValueBidder(name="demo", conversion_value=60, daily_budget=1e9)

        st.subheader("Predicted Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted CTR", f"{mean_ctr*100:.2f}%")
        bid_cpm = 30.0 * mean_ctr   # CTR-based bid
        m2.metric("Recommended Bid (CPM)", f"${bid_cpm:.3f}")
        ev = mean_ctr * 0.03 * 60 * 1000
        m3.metric("Expected Value (CPM)", f"${ev:.3f}")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=probs, nbinsx=30,
            marker_color=ACCENT, opacity=0.8, name="pCTR"
        ))
        fig.add_vline(x=mean_ctr, line_dash="dash", line_color="#FF6B9D",
                      annotation_text=f"mean={mean_ctr:.3f}",
                      annotation_font_color="#FF6B9D")
        fig.update_layout(
            xaxis_title="Predicted Click Probability",
            yaxis_title="Count",
            title=f"pCTR Distribution — {profile_sel} | {ad_sel}",
        )
        _fig_defaults(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Profile × Ad heatmap
    st.divider()
    st.subheader("CTR Heatmap: User Profile × Ad Category")
    if meta_test is not None and "_profile" in meta_test.columns:
        y_prob_all = best_model.predict_proba(X_test)
        meta_ext   = meta_test.copy()
        meta_ext["y_prob"] = y_prob_all
        hm = (meta_ext
              .groupby(["_profile", "_ad_category"])["y_prob"]
              .mean()
              .unstack(fill_value=0))
        fig_hm = px.imshow(
            hm.values * 100,
            x=[c.replace("_", " ").title() for c in hm.columns],
            y=[r.replace("_", " ").title() for r in hm.index],
            color_continuous_scale="Teal",
            aspect="auto",
            labels={"color": "Pred. CTR %"},
            title="Predicted CTR % by User Segment and Ad Category",
        )
        fig_hm.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
            font_color=TEXT, font_family="JetBrains Mono",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Meta-data not available — run with simulated data to see the heatmap.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Live Auction
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🏷️ Live Auction":
    st.title("🏷️ LIVE AUCTION SIMULATOR")
    st.divider()

    from data_simulator import USER_PROFILES, AD_CATEGORIES, simulate_criteo_row
    from bidding_engine import (AuctionEngine, BidRequest,
                                 build_default_bidders, FixedBidder,
                                 CTRBidder, ValueBidder, ThresholdBidder)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Auction Config")
        n_auctions = st.slider("Number of Auctions", 50, 2000, 500, step=50)
        floor_price = st.slider("Floor Price (CPM $)", 0.01, 1.0, 0.10, step=0.01)
        seed = st.number_input("Random Seed", value=42, step=1)
        run_btn = st.button("▶ Run Auction", use_container_width=True)

    if run_btn:
        rng = np.random.default_rng(int(seed))
        profiles = list(USER_PROFILES.keys())
        rows_live = []
        for _ in range(n_auctions):
            p  = rng.choice(profiles)
            ac = rng.choice(AD_CATEGORIES)
            rows_live.append(simulate_criteo_row(rng, p, ac))
        df_live = pd.DataFrame(rows_live)
        X_live  = pipeline.transform(df_live)

        reqs = [
            BidRequest(
                impression_id=str(i),
                user_profile=df_live.iloc[i].get("_profile", "?"),
                ad_category=df_live.iloc[i].get("_ad_category", "?"),
                features=X_live[i],
                floor_price=float(floor_price),
                timestamp=float(i),
            )
            for i in range(n_auctions)
        ]

        fresh_bidders = build_default_bidders()
        eng = AuctionEngine(fresh_bidders, best_model)
        live_df = eng.run_simulation(reqs, verbose=False)

        with col2:
            st.subheader("Results")
            won = live_df[live_df["winner"].notna()]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Win Rate",    f"{len(won)/len(live_df)*100:.1f}%")
            m2.metric("Avg Clear $", f"${won['clearing_price'].mean():.3f}")
            m3.metric("Clicks",      int(won["clicked"].sum()))
            m4.metric("Conversions", int(won["converted"].sum()))

        # winner breakdown
        winner_counts = live_df["winner"].value_counts().reset_index()
        winner_counts.columns = ["Bidder", "Wins"]
        fig_w = px.pie(
            winner_counts, names="Bidder", values="Wins",
            title="Impression Wins by Bidder",
            color_discrete_sequence=COLORS,
            hole=0.4,
        )
        fig_w.update_layout(
            paper_bgcolor=DARK_BG, font_color=TEXT,
            font_family="JetBrains Mono",
        )
        st.plotly_chart(fig_w, use_container_width=True)

        # auction feed
        st.subheader("Auction Feed (last 20)")
        for _, row in live_df.tail(20).iloc[::-1].iterrows():
            clicked_icon = "🟢" if row["clicked"] else "⚫"
            winner_str   = row["winner"] if pd.notna(row["winner"]) else "No winner"
            st.markdown(
                f'<div class="auction-card">'
                f'<b>#{row["impression_id"]}</b>  '
                f'winner=<span style="color:#64FFDA">{winner_str}</span>  '
                f'clear=${row["clearing_price"]:.3f}  '
                f'{clicked_icon} click={row["clicked"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # bidder stats table
        st.divider()
        st.subheader("Bidder Leaderboard")
        stats_df = pd.DataFrame([b.stats() for b in fresh_bidders])
        st.dataframe(
            stats_df.style.highlight_max(
                subset=["ctr", "won"], color="#1E3A5F"
            ).format(precision=4),
            use_container_width=True,
        )
    else:
        with col2:
            st.info("Configure auction parameters and click ▶ Run Auction")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Bidder Analytics
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Bidder Analytics":
    st.title("📊 BIDDER ANALYTICS")
    st.divider()

    stats_df = pd.DataFrame([b.stats() for b in bidders])
    metrics_list = [
        ("won",           "Impressions Won",    COLORS[0]),
        ("spent_$",       "Total Spend ($)",    COLORS[1]),
        ("ctr",           "CTR",                COLORS[2]),
        ("cvr",           "Conversion Rate",    COLORS[3]),
        ("cpc_$",         "Cost Per Click ($)", COLORS[4]),
        ("budget_used_%", "Budget Used (%)",    COLORS[5]),
    ]

    col_a, col_b = st.columns(2)
    col_c, col_d = st.columns(2)
    col_e, col_f = st.columns(2)
    columns_pairs = [
        (col_a, metrics_list[0]),
        (col_b, metrics_list[1]),
        (col_c, metrics_list[2]),
        (col_d, metrics_list[3]),
        (col_e, metrics_list[4]),
        (col_f, metrics_list[5]),
    ]

    for col, (metric, title, color) in columns_pairs:
        with col:
            fig = go.Figure(go.Bar(
                x=stats_df["name"],
                y=stats_df[metric],
                marker_color=color,
                marker_line_width=0,
                text=stats_df[metric].round(3),
                textposition="outside",
                textfont=dict(color=TEXT, size=10),
            ))
            fig.update_layout(
                title=title,
                showlegend=False,
                xaxis_tickangle=-30,
                yaxis=dict(showgrid=True),
            )
            _fig_defaults(fig)
            st.plotly_chart(fig, use_container_width=True)

    # Efficiency scatter: CTR vs Spend
    st.divider()
    st.subheader("Efficiency: CTR vs Spend")
    fig_sc = go.Figure()
    for i, row in stats_df.iterrows():
        fig_sc.add_trace(go.Scatter(
            x=[row["spent_$"]], y=[row["ctr"]],
            mode="markers+text",
            marker=dict(size=18, color=COLORS[i % len(COLORS)]),
            text=[row["name"]],
            textposition="top center",
            textfont=dict(size=9, color=TEXT),
            name=row["name"],
        ))
    fig_sc.update_layout(
        xaxis_title="Total Spend ($)",
        yaxis_title="CTR",
        title="Bidder Efficiency Scatter",
        showlegend=False,
    )
    _fig_defaults(fig_sc)
    st.plotly_chart(fig_sc, use_container_width=True)
