"""
module3_business.py — Business Quality Module (14 features → S₃)

Target proxy:
    business_raw = +0.20×norm(owner_experience)
                 + 0.15×norm(business_age)
                 + 0.15×norm(product_differentiation_score)
                 + 0.10×norm(digital_footprint)
                 + 0.10×norm(employee_productivity_score)
                 + 0.10×norm(supplier_relationships)
                 + 0.05×norm(owner_education)              # ordinal 1-4
                 + 0.05×norm(business_certification_count)
                 - 0.10×norm(customer_concentration)       # cao = phụ thuộc 1 KH = risk
                 - 0.10×norm(industry_competition_intensity)  # cạnh tranh cao = margin yếu
    S₃ = MinMaxScale(business_raw) → [0, 1]

Models: Random Forest (A) vs LightGBM (B) → compare → best/ensemble
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from src.modules.base_module import BaseModule
from src.config import MODULE3_FEATURES


def _norm(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


class Module3Business(BaseModule):
    """Business Quality Scoring Module."""

    FEATURES = MODULE3_FEATURES   # 14 features

    # Weights — thiết kế dựa trên Basel II SME risk factors + domain literature
    _WEIGHTS = {
        # Positive indicators — quản lý & năng lực kinh doanh
        "owner_experience":               +0.20,  # Kinh nghiệm > tuổi → operator know-how là yếu tố số 1
        "business_age":                   +0.15,  # Doanh nghiệp lâu năm = vượt qua nhiều chu kỳ = stable
        "product_differentiation_score":  +0.15,  # Sản phẩm khác biệt = moat, ít bị price pressure
        "digital_footprint":              +0.10,  # Hiện diện số = khả năng tiếp cận khách hàng rộng hơn
        "employee_productivity_score":    +0.10,  # Năng suất cao = quản lý hiệu quả
        "supplier_relationships":         +0.10,  # Quan hệ NCC tốt = supply chain ổn định
        "owner_education":                +0.05,  # Đã ordinal-encode: THPT=1, CĐ=2, ĐH=3, Master=4
        "business_certification_count":   +0.05,  # Chứng nhận = uy tín, tiêu chuẩn hóa

        # Negative indicators — risk factors
        "customer_concentration":         -0.10,  # Tập trung vào ít KH → 1 KH rời là catastrophic
        "industry_competition_intensity": -0.10,  # Cạnh tranh cao → margin mỏng, dễ bị squeeze out
    }
    # |w| sum = 0.20+0.15+0.15+0.10×4+0.05×2 = 1.00

    def __init__(self, verbose: bool = True):
        super().__init__(name="Module3_Business", verbose=verbose)

    # Semi-supervised blend (như Module 4):
    #   70% domain-designed proxy + 30% actual default signal
    STABILITY_WEIGHT = 0.70
    DEFAULT_WEIGHT   = 0.30

    def compute_target(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        score = pd.Series(np.zeros(len(X)), index=X.index)
        for col, w in self._WEIGHTS.items():
            if col not in X.columns:
                continue
            score += w * _norm(X[col].fillna(X[col].median()))

        # Component 1: domain proxy → [0, 1]
        stab = _norm(score)

        # Component 2: actual default signal
        if y is not None and len(y) == len(X):
            default_signal = 1.0 - y.values.astype(float)
        else:
            default_signal = np.full(len(X), 0.5)

        return self.STABILITY_WEIGHT * stab.values + self.DEFAULT_WEIGHT * default_signal

    # ── Model A: Random Forest ────────────────────────────────────────────────

    def _build_model_a(self):
        """
        RF chọn cho Module 3 vì:
        - Business quality features pha trộn numeric + ordinal (owner_education)
        - RF bagging xử lý tốt feature space đa dạng + ít nhiễu interpolation
        - Module 3 có nhiều subjective/qualitative scores → RF variance thấp hơn boosting
        """
        estimator = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
        )
        param_grid = {
            "n_estimators":     [100, 300],
            "max_depth":        [None, 10, 20],
            "min_samples_leaf": [5, 20],
        }
        return estimator, param_grid

    # ── Model B: LightGBM ─────────────────────────────────────────────────────

    def _build_model_b(self):
        """
        LightGBM chọn làm Model B vì:
        - Leaf-wise growth tốt với mixed feature types
        - min_child_samples prevent overfit trên scores tổng hợp (noisy targets)
        - So sánh với RF để quyết định bagging vs boosting nào phù hợp hơn
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
