"""
data_simulator.py
-----------------
Simulates the Criteo Click Logs dataset structure.
Criteo dataset format: Label | I1..I13 (integer) | C1..C26 (categorical hashed)
Source: https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/

If you have the actual Criteo TSV files, use load_criteo_file() instead.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import hashlib

# ── Criteo column schema ─────────────────────────────────────────────────────
INT_COLS   = [f"I{i}" for i in range(1, 14)]   # 13 integer features
CAT_COLS   = [f"C{i}" for i in range(1, 27)]   # 26 categorical features
ALL_COLS   = ["label"] + INT_COLS + CAT_COLS

# ── User-segment profiles (drive realistic click behaviour) ───────────────────
USER_PROFILES = {
    "tech_enthusiast": {
        "interests":    ["electronics", "software", "gadgets"],
        "ctr_base":     0.12,
        "conv_rate":    0.04,
        "age_range":    (22, 40),
        "income_level": 3,          # 1-5 scale
    },
    "fashion_lover": {
        "interests":    ["clothing", "accessories", "beauty"],
        "ctr_base":     0.09,
        "conv_rate":    0.03,
        "age_range":    (18, 35),
        "income_level": 2,
    },
    "sports_fan": {
        "interests":    ["sports", "fitness", "outdoors"],
        "ctr_base":     0.08,
        "conv_rate":    0.025,
        "age_range":    (20, 45),
        "income_level": 3,
    },
    "home_garden": {
        "interests":    ["home", "garden", "diy"],
        "ctr_base":     0.07,
        "conv_rate":    0.03,
        "age_range":    (30, 60),
        "income_level": 3,
    },
    "budget_shopper": {
        "interests":    ["deals", "discount", "coupons"],
        "ctr_base":     0.06,
        "conv_rate":    0.015,
        "age_range":    (25, 55),
        "income_level": 1,
    },
}

AD_CATEGORIES = ["electronics", "fashion", "sports", "home", "travel",
                 "food", "finance", "health", "automotive", "entertainment"]


def _hash_cat(value: str) -> str:
    """Simulate Criteo-style hex hashing of categorical features."""
    return hashlib.md5(value.encode()).hexdigest()[:8]


def simulate_criteo_row(rng: np.random.Generator,
                        profile_name: str,
                        ad_category: str) -> dict:
    """
    Generate one row in Criteo TSV format for a (user-profile, ad) pair.
    Integer features proxy real Criteo semantics:
      I1  = click count (last 24h)
      I2  = ad position (1=top, higher=worse)
      I3  = page depth
      I4  = time-on-site (seconds)
      I5  = hour of day (0-23)
      I6  = day of week (0=Mon)
      I7  = device type (0=desktop,1=mobile,2=tablet)
      I8  = browser_id
      I9  = past-conversion count
      I10 = ad frequency shown today
      I11 = price bucket of ad (0-9)
      I12 = recency of last visit (hours)
      I13 = number of ads seen this session
    """
    profile  = USER_PROFILES[profile_name]
    ctr_base = profile["ctr_base"]

    # ── relevance bonus when ad matches user interest ─────────────────────
    interest_match = any(
        ad_category in interest
        for interest in profile["interests"]
    )
    relevance_bonus = 0.06 if interest_match else 0.0

    # ── integer features ─────────────────────────────────────────────────
    age       = rng.integers(*profile["age_range"])
    position  = rng.integers(1, 6)
    depth     = rng.integers(1, 11)
    time_site = rng.integers(10, 900)
    hour      = rng.integers(0, 24)
    dow       = rng.integers(0, 7)
    device    = rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
    browser   = rng.integers(0, 6)
    past_conv = rng.integers(0, 5)
    freq_shown = rng.integers(0, 8)
    price_bkt = rng.integers(0, 10)
    recency   = rng.integers(0, 72)
    ads_seen  = rng.integers(0, 20)

    # ── click probability ─────────────────────────────────────────────────
    position_penalty = max(0, (position - 1) * 0.015)
    device_bonus     = 0.01 if device == 1 else 0.0   # mobile slight boost
    conv_history_b   = min(past_conv * 0.01, 0.05)
    hour_bonus       = 0.02 if 18 <= hour <= 22 else 0.0   # evening peak

    p_click = np.clip(
        ctr_base + relevance_bonus + device_bonus +
        conv_history_b + hour_bonus - position_penalty,
        0.01, 0.60
    )
    label = int(rng.random() < p_click)

    # ── categorical features (hashed, Criteo-style) ───────────────────────
    cats = {
        "C1":  _hash_cat(profile_name),
        "C2":  _hash_cat(ad_category),
        "C3":  _hash_cat(f"device_{device}"),
        "C4":  _hash_cat(f"browser_{browser}"),
        "C5":  _hash_cat(f"dow_{dow}"),
        "C6":  _hash_cat(f"hour_bucket_{hour // 6}"),
        "C7":  _hash_cat(f"age_bucket_{age // 10}"),
        "C8":  _hash_cat(f"income_{profile['income_level']}"),
        "C9":  _hash_cat(f"interest_match_{interest_match}"),
        "C10": _hash_cat(f"price_bkt_{price_bkt}"),
    }
    # fill remaining C11-C26 with realistic noise hashes
    for idx in range(11, 27):
        cats[f"C{idx}"] = _hash_cat(
            f"feat_{idx}_{rng.integers(0, 1000)}"
        )

    row = {
        "label":        label,
        "I1":  int(rng.integers(0, 10)),
        "I2":  int(position),
        "I3":  int(depth),
        "I4":  int(time_site),
        "I5":  int(hour),
        "I6":  int(dow),
        "I7":  int(device),
        "I8":  int(browser),
        "I9":  int(past_conv),
        "I10": int(freq_shown),
        "I11": int(price_bkt),
        "I12": int(recency),
        "I13": int(ads_seen),
        # metadata (not fed to model, used for dashboard)
        "_profile":      profile_name,
        "_ad_category":  ad_category,
        "_p_click":      round(p_click, 4),
        **cats
    }
    return row


def generate_dataset(n_rows: int = 200_000,
                     seed:   int = 42,
                     save_path: str | None = None) -> pd.DataFrame:
    """
    Generate n_rows of simulated Criteo-like ad impressions.

    Parameters
    ----------
    n_rows    : number of impression rows
    seed      : random seed for reproducibility
    save_path : if given, save as TSV (Criteo format) + Parquet

    Returns
    -------
    pd.DataFrame with columns matching Criteo schema + metadata cols
    """
    rng      = np.random.default_rng(seed)
    profiles = list(USER_PROFILES.keys())
    rows     = []

    print(f"[DataSimulator] Generating {n_rows:,} impressions …")
    for i in range(n_rows):
        profile    = rng.choice(profiles)
        ad_cat     = rng.choice(AD_CATEGORIES)
        rows.append(simulate_criteo_row(rng, profile, ad_cat))
        if (i + 1) % 50_000 == 0:
            print(f"  {i + 1:,} / {n_rows:,}")

    df = pd.DataFrame(rows)

    # ── persist ───────────────────────────────────────────────────────────
    if save_path:
        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)
        parquet_file = p / "criteo_simulated.parquet"
        df.to_parquet(parquet_file, index=False)
        # also write a small TSV sample (Criteo-compatible, no metadata cols)
        tsv_cols = ALL_COLS
        df[tsv_cols].head(10_000).to_csv(
            p / "criteo_sample.tsv", sep="\t", index=False, header=False
        )
        print(f"[DataSimulator] Saved → {parquet_file}")

    ctr = df["label"].mean() * 100
    print(f"[DataSimulator] Done. CTR = {ctr:.2f}%  |  shape = {df.shape}")
    return df


# ── Load REAL Criteo TSV (if you have downloaded it) ─────────────────────────
def load_criteo_file(tsv_path: str,
                     max_rows: int = 500_000) -> pd.DataFrame:
    """
    Load an actual Criteo 1-TB log TSV file.
    Format: label<TAB>I1..I13<TAB>C1..C26   (no header)
    Download: https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/
    """
    print(f"[DataSimulator] Loading Criteo file: {tsv_path}")
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=ALL_COLS,
        nrows=max_rows,
        dtype=str,
    )
    # cast integer columns
    for col in INT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["label"] = df["label"].astype(int)
    print(f"[DataSimulator] Loaded {len(df):,} rows. CTR = {df['label'].mean()*100:.2f}%")
    return df


if __name__ == "__main__":
    df = generate_dataset(n_rows=100_000, save_path="data")
    print(df.head(3).to_string())
