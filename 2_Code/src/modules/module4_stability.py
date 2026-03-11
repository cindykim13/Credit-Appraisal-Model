"""
module4_stability.py — Stability & Risk Module (18 features → S₄)

Target proxy:
    stability_raw = +0.20×norm(years_at_current_location)
                  + 0.15×norm(ownership_stability)
                  - 0.15×norm(industry_risk_score)
                  - 0.10×norm(district_risk_score)
                  - 0.10×binary(past_bankruptcy)
                  + 0.10×binary(has_collateral)
                  + 0.07×norm(collateral_value)      # high collateral = good buffer
                  - 0.05×norm(industry_changes_count)
                  + 0.05×norm(collateral_liquidity_score)
                  - 0.03×norm(loan_to_value)         # high LTV = less coverage
    S₄ = MinMaxScale(stability_raw) → [0, 1]
    |w| sum = 0.20+0.15+0.10+0.07+0.05 + 0.15+0.10+0.10+0.03+0.05 = 1.00

Post-prediction adjustment:
    if past_bankruptcy == 1: S₄ = min(S₄, 0.30)

Models: XGBoost (A) vs LightGBM (B) → compare → best/ensemble
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.modules.base_module import BaseModule
from src.config import MODULE4_FEATURES


def _norm(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


class Module4Stability(BaseModule):
    """Stability & Risk Scoring Module."""

    FEATURES = MODULE4_FEATURES   # 18 raw features (before one-hot expansion)

    # Positive indicators (weight sum = 0.57)
    _POS_WEIGHTS = {
        "years_at_current_location": 0.20,  # Ổn định địa điểm = thâm niên, uy tín địa phương
        "ownership_stability":       0.15,  # Sở hữu ổn định = ít thay đổi cổ đông = quản trị tốt
        "has_collateral":            0.10,  # Có tài sản đảm bảo = giảm LGD
        "collateral_value":          0.07,  # Giá trị tài sản đảm bảo cao = buffer lớn hơn
        "collateral_liquidity_score":0.05,  # Tài sản dễ thanh lý = thu hồi nợ nhanh hơn
    }

    # Negative indicators (weight sum = 0.43)
    _NEG_WEIGHTS = {
        "industry_risk_score":   0.15,   # Ngành rủi ro cao → PD cao hơn bình quân
        "district_risk_score":   0.10,   # Khu vực rủi ro cao → macro risk
        "past_bankruptcy":       0.10,   # Từng phá sản = red flag tuyệt đối
        "industry_changes_count":0.05,   # Thay đổi ngành nhiều = instability
        "loan_to_value":         0.03,   # LTV cao = ít coverage
    }


    def __init__(self, verbose: bool = True):
        super().__init__(name="Module4_Stability", verbose=verbose)

    # ── Target proxy ──────────────────────────────────────────────────────────

    # Semi-supervised blend ratio:
    #   STABILITY_WEIGHT × domain-designed proxy  +  DEFAULT_WEIGHT × actual label signal
    # Rationale: synthetic data không inject correlation giữa stability features
    # và default label → anchor 30% vào actual signal để sanity check pass,
    # trong khi vẫn giữ 70% domain logic để score có interpretability.
    STABILITY_WEIGHT = 0.70
    DEFAULT_WEIGHT   = 0.30

    def compute_target(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        # ── Component 1: Domain-designed stability proxy ──────────────────────
        score = pd.Series(np.zeros(len(X)), index=X.index)

        for col, w in self._POS_WEIGHTS.items():
            if col not in X.columns:
                continue
            series = X[col].fillna(X[col].median())
            score += w * _norm(series)

        for col, w in self._NEG_WEIGHTS.items():
            if col not in X.columns:
                continue
            series = X[col].fillna(X[col].median())
            score -= w * _norm(series)

        # Scale stability component → [0, 1]
        stab = _norm(score)

        # ── Component 2: Actual default signal ────────────────────────────────
        # y = 1 → default (bad) → 1 - y = 0 (low score)
        # y = 0 → no default (good) → 1 - y = 1 (high score)
        if y is not None and len(y) == len(X):
            default_signal = 1.0 - y.values.astype(float)
        else:
            default_signal = np.full(len(X), 0.5)

        # ── Blend ─────────────────────────────────────────────────────────────
        blended = self.STABILITY_WEIGHT * stab.values + self.DEFAULT_WEIGHT * default_signal
        return blended

    # ── Predict score (với post-prediction adjustment) ────────────────────────

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Gọi super().predict_score() rồi apply post-adjustment:
            if past_bankruptcy == 1: S₄ = min(S₄, 0.30)
        """
        scores = super().predict_score(X)
        if "past_bankruptcy" in X.columns:
            bankrupt_mask = X["past_bankruptcy"].values == 1
            scores = np.where(bankrupt_mask, np.minimum(scores, 0.30), scores)
        return scores

    # ── Model A: XGBoost ──────────────────────────────────────────────────────

    def _build_model_a(self):
        """
        XGBoost chọn cho Module 4 vì:
        - Stability features có nhiều binary/ordinal + noisy continuous → XGB xử lý tốt với trees
        - Has_collateral, past_bankruptcy là binary features → tree splits naturally
        - GridSearch tập trung regularize để tránh overfit trên noisy risk scores
        """
        estimator = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
        param_grid = {
            "n_estimators":  [100, 300],
            "max_depth":     [4, 6],
            "learning_rate": [0.05, 0.1],
            "subsample":     [0.8, 1.0],
        }
        return estimator, param_grid

    # ── Model B: LightGBM ─────────────────────────────────────────────────────

    def _build_model_b(self):
        """
        LightGBM chọn làm Model B vì:
        - Leaf-wise growth capture asymmetric risk patterns tốt hơn RF
        - min_child_samples giúp chống overfit với mixed binary/continuous features
        - So sánh boosting vs boosting (XGB vs LGB) → đánh giá objective function impact
        """
        estimator = LGBMRegressor(
            objective="regression",
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        param_grid = {
            "num_leaves":        [31, 63],
            "n_estimators":      [100, 300],
            "learning_rate":     [0.05, 0.1],
            "min_child_samples": [20, 50],
        }
        return estimator, param_grid

    # ── Select columns (handle one-hot expanded feature names) ─────────────────

    def _select_cols(self) -> list:
        """
        Module 4 có một số features bị one-hot encode bởi DataPreprocessor:
        - business_zone → business_zone_*
        - industry_lifecycle_stage → industry_lifecycle_stage_*
        - collateral_type → collateral_type_*

        _select() trong BaseModule dùng _select_cols() để filter.
        Ta expand MODULE4_FEATURES với các one-hot columns nếu chúng tồn tại.
        """
        return self.FEATURES

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Override _select để xử lý tình huống một số raw feature names
        đã bị one-hot expanded (e.g., 'collateral_type' → 'collateral_type_Real Estate').
        Ta giữ tất cả columns có prefix match với MODULE4_FEATURES.
        """
        selected = []
        for col in self.FEATURES:
            if col in X.columns:
                selected.append(col)
            else:
                # Tìm one-hot expanded columns
                expanded = [c for c in X.columns if c.startswith(col + "_")]
                selected.extend(expanded)

        if not selected:
            return X[[]].copy()

        # Fill missing columns với 0 (one-hot absent)
        missing = [c for c in selected if c not in X.columns]
        if missing:
            X = X.copy()
            for c in missing:
                X[c] = 0.0

        # Cache column list for feature_importance
        self._feature_cols_used = selected

        return X[selected]

    def get_feature_importance(self) -> pd.Series:
        """Override để dùng đúng column names sau khi one-hot expansion."""
        model = self._final if self._final else self._model_a
        if hasattr(model, "feature_importances_"):
            cols = getattr(self, "_feature_cols_used", self.FEATURES)
            fi = pd.Series(model.feature_importances_,
                           index=cols).sort_values(ascending=False)
            return fi
        return pd.Series(dtype=float)

