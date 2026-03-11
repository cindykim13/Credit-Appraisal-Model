"""Retrain Module 4 — semi-supervised blend (70% domain + 30% default signal)"""
import sys
sys.path.insert(0, '.')

import numpy as np
from src.utils.data_loader import load_splits_full, split_xy
from src.utils.preprocessor import DataPreprocessor
from src.modules.module4_stability import Module4Stability
from src.config import MODELS_DIR

train_df, val_df, _ = load_splits_full(verbose=False)
prep = DataPreprocessor(verbose=False)
X_train, y_train = prep.fit_transform(train_df)
X_val = prep.transform(val_df)
_, y_val = split_xy(val_df)

mod = Module4Stability(verbose=True)
mod.fit(X_train, y_train, X_val, y_val)

metrics = mod.validate(X_val, y_val)
print(f"\n=== Validation Metrics ===")
print(f"R2  = {metrics['r2']}")
print(f"MAE = {metrics['mae']}")

scores_all = mod.predict_score(X_val)
s_default0 = scores_all[y_val.values == 0].mean()
s_default1 = scores_all[y_val.values == 1].mean()
gap = s_default0 - s_default1
corr = np.corrcoef(scores_all, y_val.values)[0, 1]

print(f"\n=== Sanity Check ===")
print(f"score(default=0) = {s_default0:.3f}")
print(f"score(default=1) = {s_default1:.3f}")
print(f"gap              = {gap:.3f}")
print(f"default_corr     = {corr:.3f}")
print("Sanity: PASS ✅" if gap > 0.01 else f"Sanity: {'OK' if gap > 0 else 'WEAK'}")

fi = mod.get_feature_importance()
print(f"\n=== Top 5 Feature Importance ===")
print(fi.head(5).to_string())

mod.save(MODELS_DIR / "module4.pkl")
print("\nSaved: models/module4.pkl")
