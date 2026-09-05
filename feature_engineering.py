"""
feature_engineering.py
-----------------------
Preprocessing & feature engineering pipeline for the Criteo-format data.
Handles:
  - Missing-value imputation
  - Integer feature scaling
  - Categorical hashing / ordinal encoding
  - Interaction feature creation
  - Train/val/test split
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

INT_COLS = [f"I{i}" for i in range(1, 14)]
CAT_COLS = [f"C{i}" for i in range(1, 27)]

# metadata columns added by the simulator (not fed to the model)
META_COLS = ["_profile", "_ad_category", "_p_click"]


class CriteoFeaturePipeline:
    """
    End-to-end feature pipeline compatible with both simulated
    and real Criteo TSV data.
    """

    def __init__(self, cat_vocab_size: int = 50):
        """
        Parameters
        ----------
        cat_vocab_size : top-k hash buckets to keep per categorical column
        """
        self.cat_vocab_size = cat_vocab_size
        self.int_imputer    = SimpleImputer(strategy="median")
        self.scaler         = StandardScaler()
        self.cat_encoders   : dict[str, LabelEncoder] = {}
        self.cat_top_vals   : dict[str, set] = {}
        self.feature_names  : list[str] = []
        self._fitted        = False

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit pipeline on df and return transformed X matrix."""
        X_int = self._fit_transform_int(df)
        X_cat = self._fit_transform_cat(df, fit=True)
        X_interact = self._interaction_features(df)
        self._fitted = True
        X = np.hstack([X_int, X_cat, X_interact])
        self._set_feature_names(X_cat.shape[1])
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform df using the fitted pipeline."""
        assert self._fitted, "Call fit_transform() first."
        X_int = self._transform_int(df)
        X_cat = self._fit_transform_cat(df, fit=False)
        X_interact = self._interaction_features(df)
        return np.hstack([X_int, X_cat, X_interact])

    def save(self, path: str = "models"):
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self, f"{path}/feature_pipeline.pkl")
        print(f"[Pipeline] Saved → {path}/feature_pipeline.pkl")

    @staticmethod
    def load(path: str = "models") -> "CriteoFeaturePipeline":
        obj = joblib.load(f"{path}/feature_pipeline.pkl")
        print(f"[Pipeline] Loaded from {path}/feature_pipeline.pkl")
        return obj

    # ─────────────────────────────────────────────────────────────────────
    # Integer features
    # ─────────────────────────────────────────────────────────────────────
    def _fit_transform_int(self, df: pd.DataFrame) -> np.ndarray:
        X = df[INT_COLS].values.astype(float)
        X = self.int_imputer.fit_transform(X)
        # log1p transform to compress heavy-tailed distributions
        X = np.log1p(np.clip(X, 0, None))
        X = self.scaler.fit_transform(X)
        return X

    def _transform_int(self, df: pd.DataFrame) -> np.ndarray:
        X = df[INT_COLS].values.astype(float)
        X = self.int_imputer.transform(X)
        X = np.log1p(np.clip(X, 0, None))
        X = self.scaler.transform(X)
        return X

    # ─────────────────────────────────────────────────────────────────────
    # Categorical features
    # ─────────────────────────────────────────────────────────────────────
    def _fit_transform_cat(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        parts = []
        for col in CAT_COLS:
            vals = df[col].fillna("__missing__").astype(str)
            if fit:
                top = (
                    vals.value_counts()
                        .head(self.cat_vocab_size)
                        .index.tolist()
                )
                self.cat_top_vals[col] = set(top)
                le = LabelEncoder()
                vals_clipped = vals.where(vals.isin(self.cat_top_vals[col]),
                                          "__other__")
                le.fit(vals_clipped.unique().tolist() + ["__other__"])
                self.cat_encoders[col] = le
            else:
                vals_clipped = vals.where(vals.isin(self.cat_top_vals[col]),
                                          "__other__")
                le = self.cat_encoders[col]

            encoded = le.transform(
                vals.where(vals.isin(le.classes_), "__other__")
            )
            # normalise to [0, 1]
            max_val = max(len(le.classes_) - 1, 1)
            parts.append((encoded / max_val).reshape(-1, 1))

        return np.hstack(parts)

    # ─────────────────────────────────────────────────────────────────────
    # Interaction features
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _interaction_features(df: pd.DataFrame) -> np.ndarray:
        """
        Hand-crafted interaction terms known to matter in CTR prediction:
          - position × device
          - time_on_site × past_conversions
          - frequency × recency
          - price_bucket × income_proxy (I11 × I13)
        """
        pos      = df["I2"].fillna(1).astype(float).values
        device   = df["I7"].fillna(0).astype(float).values
        time_s   = df["I4"].fillna(0).astype(float).values
        past_c   = df["I9"].fillna(0).astype(float).values
        freq     = df["I10"].fillna(0).astype(float).values
        recency  = df["I12"].fillna(0).astype(float).values
        price    = df["I11"].fillna(0).astype(float).values
        ads_seen = df["I13"].fillna(0).astype(float).values

        features = np.column_stack([
            np.log1p(pos) * np.log1p(device + 1),    # position × device
            np.log1p(time_s) * np.log1p(past_c + 1), # engagement × history
            np.log1p(freq) * np.log1p(recency + 1),  # fatigue signal
            np.log1p(price) * np.log1p(ads_seen + 1),# value × exposure
        ])
        # standardise interaction block
        mean = features.mean(axis=0)
        std  = features.std(axis=0) + 1e-8
        return (features - mean) / std

    def _set_feature_names(self, n_cat: int):
        int_names      = [f"int_{c}"    for c in INT_COLS]
        cat_names      = [f"cat_{c}"    for c in CAT_COLS]
        interact_names = ["ix_pos_dev", "ix_engage_hist",
                          "ix_fatigue",  "ix_value_exp"]
        self.feature_names = int_names + cat_names + interact_names


# ─────────────────────────────────────────────────────────────────────────────
# Dataset preparation helper
# ─────────────────────────────────────────────────────────────────────────────
def prepare_dataset(df: pd.DataFrame,
                    test_size: float  = 0.15,
                    val_size: float   = 0.15,
                    seed: int         = 42
                   ) -> tuple:
    """
    Clean, engineer features, and split into train/val/test.

    Returns
    -------
    (X_train, X_val, X_test, y_train, y_val, y_test, pipeline, meta_test)
    where meta_test holds profile / ad_category info for dashboard.
    """
    print("[FeatureEng] Preparing dataset …")

    # ── drop metadata cols for model input (keep for analysis) ───────────
    meta_cols_present = [c for c in META_COLS if c in df.columns]
    meta_df = df[meta_cols_present].copy() if meta_cols_present else None
    y       = df["label"].values

    # ── train / val+test split ────────────────────────────────────────────
    idx = np.arange(len(df))
    idx_train, idx_temp = train_test_split(
        idx, test_size=(test_size + val_size), random_state=seed, stratify=y
    )
    y_temp = y[idx_temp]
    val_frac_of_temp = val_size / (test_size + val_size)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=(1 - val_frac_of_temp),
        random_state=seed, stratify=y_temp
    )

    df_train = df.iloc[idx_train]
    df_val   = df.iloc[idx_val]
    df_test  = df.iloc[idx_test]

    print(f"  Train: {len(df_train):,}  Val: {len(df_val):,}  "
          f"Test: {len(df_test):,}")

    # ── feature pipeline ─────────────────────────────────────────────────
    pipeline = CriteoFeaturePipeline()
    X_train  = pipeline.fit_transform(df_train)
    X_val    = pipeline.transform(df_val)
    X_test   = pipeline.transform(df_test)

    y_train  = y[idx_train]
    y_val    = y[idx_val]
    y_test   = y[idx_test]

    meta_test = (meta_df.iloc[idx_test].reset_index(drop=True)
                 if meta_df is not None else None)

    print(f"[FeatureEng] Feature matrix shape: {X_train.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test, pipeline, meta_test


if __name__ == "__main__":
    from data_simulator import generate_dataset
    df = generate_dataset(n_rows=20_000)
    X_tr, X_v, X_te, y_tr, y_v, y_te, pipe, meta = prepare_dataset(df)
    print("X_train:", X_tr.shape, " | positives:", y_tr.mean().round(3))
