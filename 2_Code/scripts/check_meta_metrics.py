"""Get clean metrics from trained MetaModel"""
import sys
sys.path.insert(0, '.')
import pandas as pd
from src.config import MODELS_DIR
from src.modules.meta_model import MetaModel

mm = MetaModel.load(MODELS_DIR / "meta_model.pkl")
mm.verbose = False

S_val  = pd.read_csv(MODELS_DIR / "scores_val.csv");  y_val  = S_val.pop("default")
S_test = pd.read_csv(MODELS_DIR / "scores_test.csv"); y_test = S_test.pop("default")

print("=== Val Metrics ===")
m = mm.validate(S_val, y_val); mm.verbose = True
for k,v in m.items(): print(f"  {k}: {v}")

mm.verbose = False
print("\n=== Test Metrics ===")
m2 = mm.validate(S_test, y_test); mm.verbose = True
for k,v in m2.items(): print(f"  {k}: {v}")

coef = mm.get_lr_coefficients()
if coef is not None:
    print("\n=== LR Coefficients ===")
    for k,v in coef.items(): print(f"  {k}: {v:.4f}")

shap = mm.get_shap_summary()
if shap is not None:
    print("\n=== SHAP Summary ===")
    for k,v in shap.items(): print(f"  {k}: {v:.4f}")

strat = "ENSEMBLE" if mm._ensemble_w else f"SINGLE {'LR' if isinstance(mm._final, type(mm._model_a)) else 'XGB'}"
print(f"\nStrategy: {strat}")

print("\n=== Sample Explanation ===")
exp = mm.explain(S_val.iloc[0])
for k,v in exp.items(): print(f"  {k}: {v}")
