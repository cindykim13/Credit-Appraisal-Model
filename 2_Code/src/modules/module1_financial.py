"""
module1_financial.py — Financial Health Module (22 features → S₁)

Target proxy:
    health_raw = 0.25*norm(roa) + 0.20*norm(dscr) + 0.20*norm(current_ratio)
               + 0.15*norm(profit_margin) + 0.10*norm(revenue_growth)
               - 0.10*norm(debt_to_asset)
    S₁ = MinMaxScale(health_raw) → [0,1]

Models: XGBoost (A) vs LightGBM (B) → compare → best/ensemble
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler

from src.modules.base_module import BaseModule
from src.config import MODULE1_FEATURES


def _norm(s: pd.Series) -> pd.Series:
    """Min-max normalize một Series."""
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


class Module1Financial(BaseModule):
    """Financial Health Scoring Module."""

    FEATURES = MODULE1_FEATURES

    # Weights theo plan (domain + Basel II)
    _WEIGHTS = {
        "roa":            +0.25,
        "dscr":           +0.20,
        "current_ratio":  +0.20,
        "profit_margin":  +0.15,
        "revenue_growth": +0.10,
        "debt_to_asset":  -0.10,   # leverage cao → risk cao
    }

    def __init__(self, verbose: bool = True):
        super().__init__(name="Module1_Financial", verbose=verbose)

    # Semi-supervised blend (như Module 4):
    #   70% domain-designed proxy + 30% actual default signal
    # Giải quyết limitation: financial ratios trong synthetic data
    # không có causal correlation với default label.
    STABILITY_WEIGHT = 0.70
    DEFAULT_WEIGHT   = 0.30

    def compute_target(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Blended target = 70% financial health proxy + 30% (1 - default).
        """
        score = pd.Series(np.zeros(len(X)), index=X.index)
        for col, w in self._WEIGHTS.items():
            if col in X.columns:
                score += w * _norm(X[col])

        # Component 1: domain proxy → [0, 1]
        stab = _norm(score)

        # Component 2: actual default signal (1=good, 0=default)
        if y is not None and len(y) == len(X):
            default_signal = 1.0 - y.values.astype(float)
        else:
            default_signal = np.full(len(X), 0.5)

        return self.STABILITY_WEIGHT * stab.values + self.DEFAULT_WEIGHT * default_signal

    # ── Model A: XGBoost ──────────────────────────────────────────────────────

    def _build_model_a(self):
        estimator = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
        param_grid = {
            "max_depth":    [3, 5, 7],
            "n_estimators": [100, 300],
            "learning_rate":[0.05, 0.1],
            "reg_lambda":   [1, 5],
        }
        return estimator, param_grid

    # ── Model B: LightGBM ────────────────────────────────────────────────────

    def _build_model_b(self):
        estimator = LGBMRegressor(
            objective="regression",
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        param_grid = {
            "num_leaves":       [31, 63],
            "n_estimators":     [100, 300],
            "learning_rate":    [0.05, 0.1],
            "min_child_samples":[20, 50],
        }
        return estimator, param_grid


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.utils.data_loader import load_splits
    from src.utils.preprocessor import DataPreprocessor

    print("=== Module 1: Financial Health — Smoke Test ===")
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(verbose=False)

    prep = DataPreprocessor(verbose=False)
    X_train = prep.fit_transform(X_train)
    X_val   = prep.transform(X_val)
    X_test  = prep.transform(X_test)

    mod = Module1Financial(verbose=True)
    mod.fit(X_train, y_train, X_val, y_val)

    metrics = mod.validate(X_val, y_val)
    print(f"\n  Val  R²={metrics['r2']}  MAE={metrics['mae']}")

    scores = mod.predict_score(X_val)
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Score mean(default=0): {scores[y_val==0].mean():.4f}")
    print(f"  Score mean(default=1): {scores[y_val==1].mean():.4f}")

    fi = mod.get_feature_importance()
    print(f"\n  Top-5 feature importance:\n{fi.head(5).to_string()}")

    mod.save("models/module1.pkl")

    # Reload test
    mod2 = Module1Financial.load("models/module1.pkl")
    scores2 = mod2.predict_score(X_val)
    assert np.allclose(scores, scores2, atol=1e-6), "Save/load mismatch!"
    print("\n✅ Module 1 OK")
