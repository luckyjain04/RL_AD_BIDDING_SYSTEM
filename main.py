"""
main.py
-------
End-to-end pipeline runner for the Personalized Ad Bidding System.

Usage
-----
  python main.py                     # full run with simulated data
  python main.py --rows 50000        # custom row count
  python main.py --criteo path/to/file.tsv   # use real Criteo data
  python main.py --skip-train        # skip retraining (load saved models)

Then launch the dashboard:
  streamlit run dashboard.py
"""

import argparse
import time
import numpy as np
import joblib
from pathlib import Path

from data_simulator      import generate_dataset, load_criteo_file
from feature_engineering import prepare_dataset, CriteoFeaturePipeline
import model_trainer as _mt
from bidding_engine      import (AuctionEngine, BidRequest,
                                  build_default_bidders)
from evaluation          import run_full_evaluation


# ─────────────────────────────────────────────────────────────────────────────
def build_bid_requests(df, pipeline: CriteoFeaturePipeline,
                       n: int = 2000) -> list[BidRequest]:
    """Convert n rows of the test set into BidRequest objects."""
    sample = df.sample(min(n, len(df)), random_state=0)
    X      = pipeline.transform(sample)
    reqs   = []
    for i, (_, row) in enumerate(sample.iterrows()):
        reqs.append(BidRequest(
            impression_id=str(i),
            user_profile=row.get("_profile",  "unknown"),
            ad_category=row.get("_ad_category", "unknown"),
            features=X[i],
            floor_price=np.random.uniform(0.05, 0.30),
            timestamp=float(i),
        ))
    return reqs


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Personalized Ad Bidding System")
    ap.add_argument("--rows",       type=int,  default=100_000,
                    help="Number of simulated rows (default 100 000)")
    ap.add_argument("--criteo",     type=str,  default=None,
                    help="Path to real Criteo TSV file")
    ap.add_argument("--skip-train", action="store_true",
                    help="Load saved models instead of retraining")
    ap.add_argument("--auctions",   type=int,  default=2000,
                    help="Number of simulated auctions")
    ap.add_argument("--out",        type=str,  default="outputs",
                    help="Output directory for plots / artefacts")
    args = ap.parse_args()

    t0 = time.time()
    Path("models").mkdir(exist_ok=True)
    Path(args.out).mkdir(exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────────────────
    if args.criteo:
        df = load_criteo_file(args.criteo, max_rows=args.rows)
    else:
        df = generate_dataset(n_rows=args.rows, save_path="data")

    # ── 2. Feature engineering ───────────────────────────────────────────
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     pipeline, meta_test) = prepare_dataset(df)
    pipeline.save("models")

    # ── 3. Model training ────────────────────────────────────────────────
    if args.skip_train:
        print("[Main] Loading saved models …")
        from model_trainer import (LogisticRegressionModel, DecisionTreeModel,
                                   RandomForestModel, LightGBMModel,
                                   train_all_models)
        # rebuild results by evaluating loaded models
        model_classes = {
            "logistic_regression": _mt.LogisticRegressionModel,
            "decision_tree":       _mt.DecisionTreeModel,
            "random_forest":       _mt.RandomForestModel,
        }
        try:
            model_classes["lightgbm"] = _mt.LightGBMModel
        except Exception:
            pass
        trained = {}
        results = {}
        for name in model_classes:
            fp = f"models/{name}.pkl"
            if Path(fp).exists():
                m = joblib.load(fp)
                trained[name] = m
                results[name] = {
                    "val":  _mt.compute_metrics(y_val,  m.predict_proba(X_val)),
                    "test": _mt.compute_metrics(y_test, m.predict_proba(X_test)),
                }
        best_name = max(results, key=lambda k: results[k]["val"]["auc_roc"])
    else:
        out = _mt.train_all_models(X_train, y_train, X_val, y_val,
                                   X_test, y_test, save_dir="models")
        trained   = out["trained"]
        results   = out["results"]
        best_name = out["best"]

    # ── 4. Auction simulation ─────────────────────────────────────────────
    print(f"\n[Main] Running {args.auctions:,} simulated auctions …")
    best_model = trained[best_name]
    bidders    = build_default_bidders()
    requests   = build_bid_requests(df, pipeline, n=args.auctions)
    engine     = AuctionEngine(bidders, best_model)
    auction_df = engine.run_simulation(requests)
    auction_df.to_csv(f"{args.out}/auction_results.csv", index=False)
    print(f"[Main] Auction results → {args.out}/auction_results.csv")

    # ── 5. Evaluation plots ───────────────────────────────────────────────
    eval_out = run_full_evaluation(
        trained_models=trained,
        results=results,
        best_name=best_name,
        X_test=X_test,
        y_test=y_test,
        meta_test=meta_test,
        pipeline=pipeline,
        bidders=bidders,
        out_dir=args.out,
    )

    # ── 6. Save artefacts for dashboard ──────────────────────────────────
    import pickle
    state = {
        "trained":    trained,
        "results":    results,
        "best_name":  best_name,
        "pipeline":   pipeline,
        "meta_test":  meta_test,
        "X_test":     X_test,
        "y_test":     y_test,
        "auction_df": auction_df,
        "bidders":    bidders,
    }
    with open(f"{args.out}/dashboard_state.pkl", "wb") as f:
        pickle.dump(state, f)
    print(f"[Main] Dashboard state → {args.out}/dashboard_state.pkl")

    elapsed = time.time() - t0
    print(f"\n[Main] ✓ Pipeline complete in {elapsed:.1f}s")
    print(f"[Main] → Launch dashboard:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()
