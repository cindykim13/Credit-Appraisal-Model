"""
score_collector.py — Generate S₁–S₄ intermediate scores và save thành CSV.

Output:
  models/scores_train.csv  [S1, S2, S3, S4, default]
  models/scores_val.csv    [S1, S2, S3, S4, default]
  models/scores_test.csv   [S1, S2, S3, S4, default]
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from src.config import MODELS_DIR
from src.pipeline import CreditScoringPipeline
from src.utils.data_loader import load_splits_full

print("=== Score Collector ===")

# Load data
train_df, val_df, test_df = load_splits_full(verbose=True)

# Fit pipeline (fits preprocessor + loads 4 modules)
pipeline = CreditScoringPipeline(verbose=True)
pipeline.fit(train_df)

# Generate scores
print("\n--- Generating scores ---")
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    scores_df = pipeline.predict_scores_with_label(df)
    out_path = MODELS_DIR / f"scores_{name}.csv"
    scores_df.to_csv(out_path, index=False)

    stats = scores_df.drop(columns=["default"]).describe().loc[["mean", "std"]]
    default_rate = scores_df["default"].mean()
    print(f"\n  [{name}] saved → {out_path.name}")
    print(f"    rows={len(scores_df)} | default_rate={default_rate:.1%}")
    print(f"    Score means: S1={scores_df['S1'].mean():.3f} | S2={scores_df['S2'].mean():.3f} | "
          f"S3={scores_df['S3'].mean():.3f} | S4={scores_df['S4'].mean():.3f}")
    # NaN check
    n_nan = scores_df.isna().sum().sum()
    print(f"    NaN count: {n_nan}")
    assert n_nan == 0, f"NaN detected in {name} scores!"

print("\nScore collection complete. ✅")
