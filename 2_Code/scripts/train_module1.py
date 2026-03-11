"""Retrain Module 1 — semi-supervised blend (70% domain + 30% default signal)"""
import sys
sys.path.insert(0, '.')

import numpy as np
from src.utils.data_loader import load_splits_full, split_xy
from src.utils.preprocessor import DataPreprocessor
from src.modules.module1_financial import Module1Financial
from src.config import MODELS_DIR

train_df, val_df, _ = load_splits_full(verbose=False)
prep = DataPreprocessor(verbose=False)
X_train, y_train = prep.fit_transform(train_df)
X_val = prep.transform(val_df)
_, y_val = split_xy(val_df)

mod = Module1Financial(verbose=True)
mod.fit(X_train, y_train, X_val, y_val)

metrics = mod.validate(X_val, y_val)
print(f"\n=== Validation Metrics ===")
print(f"R2  = {metrics['r2']}")
print(f"MAE = {metrics['mae']}")

scores_all = mod.predict_score(X_val)
s0 = scores_all[y_val.values == 0].mean()
s1 = scores_all[y_val.values == 1].mean()
corr = np.corrcoef(scores_all, y_val.values)[0, 1]
print(f"\n=== Sanity Check ===")
print(f"score(default=0) = {s0:.3f}")
print(f"score(default=1) = {s1:.3f}")
print(f"gap              = {s0-s1:.3f}")
print(f"default_corr     = {corr:.3f}")

mod.save(MODELS_DIR / "module1.pkl")
print("\nSaved: models/module1.pkl")
