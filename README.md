# ⚡ Personalized Ad Bidding System

A full-stack machine learning pipeline for **real-time ad bidding (RTB)** built
on the [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

---

## 📁 Project Structure

```
ad_bidding_system/
├── data_simulator.py        # Simulate or load real Criteo TSV data
├── feature_engineering.py   # Preprocessing + feature pipeline
├── model_trainer.py         # LR / DecTree / RF / LightGBM training
├── bidding_engine.py        # 5 bidding strategies + auction engine
├── evaluation.py            # Metrics + Matplotlib report charts
├── main.py                  # End-to-end CLI pipeline runner
├── dashboard.py             # Streamlit interactive dashboard
├── requirements.txt
├── data/                    # Generated / Criteo TSV files
├── models/                  # Saved model .pkl files
└── outputs/                 # Evaluation plots + auction results
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
cd ad_bidding_system
pip install -r requirements.txt
```

### 2. Run the full pipeline (simulated data)
```bash
python main.py --rows 100000 --auctions 2000
```

### 3. Launch the dashboard
```bash
streamlit run dashboard.py
```

---

## 📦 Using Real Criteo Data

1. Register and download from:
   https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/

2. Unzip one day file (e.g. `day_0.gz → day_0.tsv`)

3. Run with real data:
```bash
python main.py --criteo path/to/day_0.tsv --rows 500000
```

The Criteo TSV format is:
```
label<TAB>I1 I2 ... I13<TAB>C1 C2 ... C26
```
No header. Integer features may be empty (NaN). Categorical features are hex hashes.

---

## 🧠 Models

| Model              | Description                                   |
|--------------------|-----------------------------------------------|
| Logistic Regression| Fast linear baseline, well-calibrated         |
| Decision Tree      | Interpretable, depth-8                        |
| Random Forest      | 100 trees, robust to noise                    |
| **LightGBM**       | **Best performer** — gradient boosted trees   |

Key metric: **AUC-ROC** (ranking quality for bid ordering)
Secondary: **Log-Loss** (calibration quality → accurate bid values)

---

## 💰 Bidding Strategies

| Strategy          | Formula                        | Best For               |
|-------------------|-------------------------------|------------------------|
| FixedBidder       | `bid = const`                 | Baseline comparison    |
| CTRBidder         | `bid = base_cpm × pCTR`       | Awareness campaigns    |
| **ValueBidder**   | `bid = pCTR × pConv × value`  | **Conversion goals**   |
| PacingBidder      | ValueBidder + budget pacing   | Budget-constrained     |
| ThresholdBidder   | Skip if pCTR < threshold      | Quality-focused        |

Auction type: **Second-price (Vickrey)** — winner pays the 2nd-highest bid.

---

## 📊 Evaluation Metrics

- **AUC-ROC**: Overall ranking quality
- **AUC-PR**: Precision-Recall (better for imbalanced CTR data)
- **Log-Loss**: Calibration (directly impacts bid accuracy)
- **Lift@10%**: How much better than random in top-scored impressions
- **CTR / CVR**: Click-through and conversion rates
- **CPC**: Cost per click (bidder efficiency)

---

## 🖥️ Dashboard Pages

| Page                  | Content                                                |
|-----------------------|--------------------------------------------------------|
| 🏠 Overview           | KPI cards, pCTR histogram, clearing price dist        |
| 🧠 Models             | ROC/PR curves, calibration, metrics table              |
| 👤 User Personalization| Interactive profile builder, CTR heatmap             |
| 🏷️ Live Auction       | Real-time auction simulation, auction feed             |
| 📊 Bidder Analytics   | Per-bidder CTR, spend, efficiency scatter             |

---

## 🔧 CLI Options

```
python main.py [OPTIONS]

  --rows INT       Simulated rows (default: 100000)
  --criteo PATH    Path to Criteo TSV file
  --skip-train     Load saved models, skip retraining
  --auctions INT   Number of simulated auctions (default: 2000)
  --out DIR        Output directory (default: outputs/)
```

---

## 📐 Feature Engineering Details

### Integer features (I1–I13)
- Log1p transform → StandardScaler
- Median imputation for NaN

### Categorical features (C1–C26)
- Top-50 vocab per column (Criteo-style hash clipping)
- LabelEncoder → normalized to [0, 1]

### Interaction features (engineered)
- `position × device`
- `time_on_site × past_conversions`
- `frequency × recency` (fatigue signal)
- `price_bucket × ads_seen`

---

## 📄 License
MIT
