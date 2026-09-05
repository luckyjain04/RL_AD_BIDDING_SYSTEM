"""
model_trainer.py
----------------
Trains and evaluates multiple CTR-prediction models:
  1. Logistic Regression (fast baseline)
  2. Decision Tree
  3. Random Forest
  4. LightGBM (best performer)

All models expose a common interface:
  .fit(X, y)
  .predict_proba(X) → P(click)
  .evaluate(X, y) → dict of metrics
"""

import numpy as np
import pandas as pd
from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics       import (
    roc_auc_score, log_loss, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.calibration   import CalibratedClassifierCV
import joblib
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """
    Compute standard CTR-prediction metrics.

    Key metrics explained
    ---------------------
    AUC-ROC   : Area under ROC curve (ranking quality)
    AUC-PR    : Area under Precision-Recall curve (better for imbalanced data)
    Log-Loss  : Cross-entropy (calibration quality for bid pricing)
    CTR       : Average predicted click-through rate
    Lift@10%  : How much better than random in top-10% scored impressions
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Lift at top 10 %
    n_top       = max(1, int(len(y_true) * 0.10))
    top_idx     = np.argsort(y_prob)[::-1][:n_top]
    lift_10     = y_true[top_idx].mean() / (y_true.mean() + 1e-8)

    return {
        "auc_roc":       round(roc_auc_score(y_true, y_prob), 4),
        "auc_pr":        round(average_precision_score(y_true, y_prob), 4),
        "log_loss":      round(log_loss(y_true, y_prob), 4),
        "ctr_actual":    round(float(y_true.mean()), 4),
        "ctr_predicted": round(float(y_prob.mean()), 4),
        "precision":     round(tp / (tp + fp + 1e-8), 4),
        "recall":        round(tp / (tp + fn + 1e-8), 4),
        "lift_top10pct": round(lift_10, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model wrappers
# ─────────────────────────────────────────────────────────────────────────────
class BaseModel:
    """Thin wrapper that adds evaluate() and save/load helpers."""

    name: str = "base"

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y, split_name: str = "eval") -> dict:
        y_prob = self.predict_proba(X)
        metrics = compute_metrics(y, y_prob)
        print(f"  [{self.name}] {split_name}  "
              f"AUC={metrics['auc_roc']}  "
              f"LogLoss={metrics['log_loss']}  "
              f"Lift@10%={metrics['lift_top10pct']}")
        return metrics

    def save(self, path: str = "models"):
        Path(path).mkdir(parents=True, exist_ok=True)
        fp = f"{path}/{self.name}.pkl"
        joblib.dump(self, fp)
        print(f"  [{self.name}] Saved → {fp}")

    @classmethod
    def load(cls, path: str, name: str):
        return joblib.load(f"{path}/{name}.pkl")


class LogisticRegressionModel(BaseModel):
    name = "logistic_regression"

    def __init__(self):
        self.model = LogisticRegression(
            C=1.0, max_iter=500, solver="lbfgs",
            class_weight="balanced", random_state=42, n_jobs=-1
        )


class DecisionTreeModel(BaseModel):
    name = "decision_tree"

    def __init__(self, max_depth: int = 8):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=50,
            class_weight="balanced", random_state=42
        )


class RandomForestModel(BaseModel):
    name = "random_forest"

    def __init__(self, n_estimators: int = 150):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=10,
            min_samples_leaf=30, class_weight="balanced",
            random_state=42, n_jobs=-1
        )


class LightGBMModel(BaseModel):
    name = "lightgbm"

    def __init__(self):
        if not HAS_LGB:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")
        self.model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=63,
            min_child_samples=50,
            colsample_bytree=0.8,
            subsample=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X, y,
            eval_set=eval_set,
        )
        return self

    def feature_importances(self, feature_names: list) -> pd.DataFrame:
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Training orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def train_all_models(X_train, y_train,
                     X_val,   y_val,
                     X_test,  y_test,
                     save_dir: str = "models") -> dict:
    """
    Train every model, evaluate on val & test, return results dict.
    """
    print("\n" + "=" * 60)
    print("  MODEL TRAINING")
    print("=" * 60)

    models_to_train = [
        LogisticRegressionModel(),
        DecisionTreeModel(),
        RandomForestModel(n_estimators=100),
    ]
    if HAS_LGB:
        models_to_train.append(LightGBMModel())
    else:
        print("  [WARNING] LightGBM not found, skipping.")

    results = {}
    trained = {}

    for m in models_to_train:
        print(f"\n  Training {m.name} …")
        if isinstance(m, LightGBMModel):
            m.fit(X_train, y_train, X_val, y_val)
        else:
            m.fit(X_train, y_train)

        val_metrics  = m.evaluate(X_val,  y_val,  split_name="val ")
        test_metrics = m.evaluate(X_test, y_test, split_name="test")
        results[m.name] = {"val": val_metrics, "test": test_metrics}
        m.save(save_dir)
        trained[m.name] = m

    # ── pick best model by val AUC ────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["val"]["auc_roc"])
    print(f"\n  ✓ Best model: {best_name} "
          f"(val AUC = {results[best_name]['val']['auc_roc']})")

    return {"results": results, "trained": trained, "best": best_name}


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_simulator     import generate_dataset
    from feature_engineering import prepare_dataset

    df = generate_dataset(n_rows=30_000)
    X_tr, X_v, X_te, y_tr, y_v, y_te, pipe, meta = prepare_dataset(df)

    out = train_all_models(X_tr, y_tr, X_v, y_v, X_te, y_te)
    print("\nSummary:")
    for model_name, r in out["results"].items():
        print(f"  {model_name:25s}  test AUC={r['test']['auc_roc']}")
