"""
module2_credit.py — Credit Behavior Module (13 features → S₂)

Target proxy:
    credit_raw = -0.30×norm(max_past_due_days) - 0.25×norm(num_past_due_90d)
               + 0.20×norm(cic_score)          - 0.15×binary(previous_default_history)
               + 0.05×norm(credit_history_length) - 0.05×norm(debt_burden_ratio)*
    S₂ = MinMaxScale(1 - norm(|credit_raw|)) → [0,1]  (cao = tốt)
    (* debt_burden_ratio có thể đã bị drop bởi preprocessor)

Models: XGBoost (A) vs Random Forest (B) → compare → best/ensemble
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

from src.modules.base_module import BaseModule
from src.config import MODULE2_FEATURES


def _norm(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


class Module2Credit(BaseModule):
    """Credit Behavior Scoring Module."""

    FEATURES = MODULE2_FEATURES   # 13 features (debt_burden_ratio có thể vắng)

    # Weights v2 — Option A: tăng past_due weights + Option B: interaction feature
    _WEIGHTS = {
        "max_past_due_days":        -0.35,   # A: đầy mạnh (was -0.30)
        "num_past_due_90d":         -0.25,   # giữ nguyên
        "past_due_severity":        -0.15,   # B: NEW interaction feature
        "cic_score":                +0.15,   # A: giảm nhẹ (was +0.20)
        "previous_default_history": -0.05,   # A: giảm nhẹ (was -0.15) — đã có severity capture
        "credit_history_length":    +0.05,   # giữ nguyên
    }  # |w| sum = 0.35+0.25+0.15+0.15+0.05+0.05 = 1.00

    def __init__(self, verbose: bool = True):
        super().__init__(name="Module2_Credit", verbose=verbose)

    # ── Target proxy ──────────────────────────────────────────────────────────

    def compute_target(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Credit behavior score thô (v2 w/ interaction feature).
        Option B: tạo `past_due_severity` = max_past_due_days × (num_past_due_90d + 1)
        Được tính trước khi áp dụng weights — capture severity của late payment.
        """
        X = X.copy()
        # Option B: engineer interaction feature
        if "max_past_due_days" in X.columns and "num_past_due_90d" in X.columns:
            X["past_due_severity"] = (
                X["max_past_due_days"].fillna(0)
                * (X["num_past_due_90d"].fillna(0) + 1)
            )

        score = pd.Series(np.zeros(len(X)), index=X.index)
        for col, w in self._WEIGHTS.items():
            if col not in X.columns:
                continue
            if col == "previous_default_history":
                score += w * X[col].fillna(0).astype(float)
            else:
                score += w * _norm(X[col].fillna(X[col].median()))

        return score.values

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

    # ── Model B: Random Forest ────────────────────────────────────────────────

    def _build_model_b(self):
        estimator = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
        )
        param_grid = {
            "n_estimators":    [100, 300],
            "max_depth":       [None, 10, 20],
            "min_samples_leaf":[5, 20],
        }
        return estimator, param_grid

    # ── Override _select_cols để handle possible drop của debt_burden_ratio ──

    def _select_cols(self) -> list[str]:
        """Chỉ dùng features hiện có trong dataset (debt_burden_ratio có thể đã drop)."""
        return self.FEATURES   # BaseModule._select() đã fillna 0 nếu thiếu


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.utils.data_loader import load_splits_full, split_xy
    from src.utils.preprocessor import DataPreprocessor

    print("=== Module 2: Credit Behavior — Smoke Test ===")
    train_df, val_df, test_df = load_splits_full(verbose=False)

    prep = DataPreprocessor(verbose=False)
    X_train, y_train = prep.fit_transform(train_df)
    X_val  = prep.transform(val_df)
    _, y_val = split_xy(val_df)

    mod = Module2Credit(verbose=True)
    mod.fit(X_train, y_train, X_val, y_val)

    metrics = mod.validate(X_val, y_val)
    print(f"\n  Val  R²={metrics['r2']}  MAE={metrics['mae']}")

    scores = mod.predict_score(X_val)
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Score mean(default=0): {scores[y_val==0].mean():.4f}")
    print(f"  Score mean(default=1): {scores[y_val==1].mean():.4f}")

    fi = mod.get_feature_importance()
    print(f"\n  Top-5 feature importance:\n{fi.head(5).to_string()}")
    mod.save("models/module2.pkl")
    print("\n✅ Module 2 OK")
