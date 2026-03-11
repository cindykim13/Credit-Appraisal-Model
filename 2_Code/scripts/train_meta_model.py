"""Train and validate MetaModel (Phase 4)"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from src.config import MODELS_DIR
from src.modules.meta_model import MetaModel

# Load pre-generated scores
S_train = pd.read_csv(MODELS_DIR / "scores_train.csv")
S_val   = pd.read_csv(MODELS_DIR / "scores_val.csv")
S_test  = pd.read_csv(MODELS_DIR / "scores_test.csv")

y_train = S_train.pop("default")
y_val   = S_val.pop("default")
y_test  = S_test.pop("default")

print(f"Train: {S_train.shape} | default={y_train.mean():.1%}")
print(f"Val:   {S_val.shape}   | default={y_val.mean():.1%}")
print(f"Test:  {S_test.shape}  | default={y_test.mean():.1%}")

# Train
mm = MetaModel(verbose=True)
mm.fit(S_train, y_train, S_val, y_val)

# Validate on val
print("\n=== Val Metrics ===")
val_metrics = mm.validate(S_val, y_val)

# Validate on test
print("\n=== Test Metrics ===")
test_metrics = mm.validate(S_test, y_test)

# LR Coefficients
coef = mm.get_lr_coefficients()
if coef is not None:
    print("\n=== LR Coefficients (module weights) ===")
    print(coef.to_string())

# SHAP
shap_summary = mm.get_shap_summary()
if shap_summary is not None:
    print("\n=== SHAP Summary (mean |shap| per module) ===")
    print(shap_summary.to_string())

# Sample explanation
print("\n=== Sample Explanation (val[0]) ===")
exp = mm.explain(S_val.iloc[0])
for k, v in exp.items():
    print(f"  {k}: {v}")

# Save
mm.save(MODELS_DIR / "meta_model.pkl")
print("\n✅ MetaModel saved → models/meta_model.pkl")
