"""
base_module.py — Abstract base class cho tất cả 4 scoring modules.

Mỗi module con phải implement:
  - _build_model_a() → estimator A (e.g. XGBoost)
  - _build_model_b() → estimator B (e.g. LightGBM)
  - compute_target(X, y) → np.ndarray  (target proxy [0,1])
  - FEATURES: List[str]
"""
from __future__ import annotations
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


class BaseModule(ABC):
    """
    Luồng chuẩn:
        mod.fit(X_train, y_train, X_val, y_val)  → train + compare A vs B → ensemble
        score = mod.predict_score(X)              → np.ndarray [0, 1]
    """

    FEATURES: list[str] = []   # Override ở subclass

    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self._model_a = None          # estimator A
        self._model_b = None          # estimator B
        self._final   = None          # best single / ensemble
        self._ensemble_w: Optional[tuple] = None  # (w_a, w_b) nếu ensemble
        self._scaler_target = MinMaxScaler()
        self._fitted = False

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def _build_model_a(self):
        """Trả về untrained estimator A với param_grid."""
        ...

    @abstractmethod
    def _build_model_b(self):
        """Trả về untrained estimator B với param_grid."""
        ...

    @abstractmethod
    def compute_target(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Tính target proxy từ features (composite score chưa scale).
        Sau đó BaseModule sẽ MinMaxScale về [0,1].
        """
        ...

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
    ) -> "BaseModule":
        self._log(f"\n{'='*60}")
        self._log(f"Training {self.name}")
        self._log(f"  Train: {X_train.shape} | Val: {X_val.shape}")

        # Lấy features của module này
        X_tr = self._select(X_train)
        X_vl = self._select(X_val)

        # Tính target proxy
        t_tr_raw = self.compute_target(X_tr, y_train)
        t_vl_raw = self.compute_target(X_vl, y_val)

        # Scale target → [0,1]
        t_tr = self._scaler_target.fit_transform(t_tr_raw.reshape(-1, 1)).ravel()
        t_vl = self._scaler_target.transform(t_vl_raw.reshape(-1, 1)).ravel()

        self._log(f"  Target: mean={t_tr.mean():.3f} std={t_tr.std():.3f} "
                  f"| default corr={np.corrcoef(t_tr, y_train)[0,1]:.3f}")

        # Train A
        est_a, pg_a = self._build_model_a()
        r2_a, mae_a, fitted_a = self._train_one(est_a, pg_a, "Model A", X_tr, t_tr, X_vl, t_vl)

        # Train B
        est_b, pg_b = self._build_model_b()
        r2_b, mae_b, fitted_b = self._train_one(est_b, pg_b, "Model B", X_tr, t_tr, X_vl, t_vl)

        # So sánh → quyết định
        self._model_a = fitted_a
        self._model_b = fitted_b
        self._decide_strategy(r2_a, mae_a, r2_b, mae_b, X_vl, t_vl)
        self._fitted = True
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Trả về score ∈ [0, 1] cho từng row."""
        assert self._fitted, f"{self.name}: gọi fit() trước!"
        X_sel = self._select(X)
        if self._ensemble_w:
            wa, wb = self._ensemble_w
            raw = wa * self._model_a.predict(X_sel) + wb * self._model_b.predict(X_sel)
        else:
            raw = self._final.predict(X_sel)
        return np.clip(raw, 0.0, 1.0)

    def get_feature_importance(self) -> pd.Series:
        model = self._final if self._final else self._model_a
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_,
                           index=self._select_cols()).sort_values(ascending=False)
            return fi
        return pd.Series(dtype=float)

    def validate(self, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """Trả về R² và MAE trên val set."""
        t_vl_raw = self.compute_target(self._select(X_val), y_val)
        t_vl = self._scaler_target.transform(t_vl_raw.reshape(-1, 1)).ravel()
        preds = self.predict_score(X_val)
        return {
            "r2":  round(r2_score(t_vl, preds), 4),
            "mae": round(mean_absolute_error(t_vl, preds), 4),
        }

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self._log(f"  Saved → {path}")

    @classmethod
    def load(cls, path: Path | str) -> "BaseModule":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _train_one(self, estimator, param_grid, label,
                   X_tr, t_tr, X_vl, t_vl):
        gs = GridSearchCV(estimator, param_grid, cv=5,
                          scoring="r2", n_jobs=-1, refit=True)
        gs.fit(X_tr, t_tr)
        best = gs.best_estimator_
        preds = best.predict(X_vl)
        r2  = r2_score(t_vl, preds)
        mae = mean_absolute_error(t_vl, preds)
        self._log(f"  {label}: R²={r2:.4f}  MAE={mae:.4f}  "
                  f"best_params={gs.best_params_}")
        return r2, mae, best

    def _decide_strategy(self, r2_a, mae_a, r2_b, mae_b, X_vl, t_vl):
        delta = abs(r2_a - r2_b)
        if delta > 0.03:
            # Dùng model tốt hơn đơn lẻ
            if r2_a >= r2_b:
                self._final = self._model_a
                self._log(f"  Strategy: SINGLE Model A (ΔR²={delta:.4f})")
            else:
                self._final = self._model_b
                self._log(f"  Strategy: SINGLE Model B (ΔR²={delta:.4f})")
        else:
            # Ensemble: weight ∝ val R²
            total = r2_a + r2_b
            if total <= 0:
                wa, wb = 0.5, 0.5
            else:
                wa = r2_a / total
                wb = r2_b / total
            self._ensemble_w = (wa, wb)
            self._final = None
            # Tính metric ensemble
            ens = wa * self._model_a.predict(X_vl) + wb * self._model_b.predict(X_vl)
            r2_ens  = r2_score(t_vl, ens)
            mae_ens = mean_absolute_error(t_vl, ens)
            self._log(f"  Strategy: ENSEMBLE (w_A={wa:.2f}, w_B={wb:.2f}) "
                      f"→ R²={r2_ens:.4f}  MAE={mae_ens:.4f}")

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = self._select_cols()
        missing = [c for c in cols if c not in X.columns]
        if missing:
            # one-hot expansion or genuinely missing → fill 0
            for c in missing:
                X = X.copy()
                X[c] = 0.0
        return X[cols]

    def _select_cols(self) -> list[str]:
        return self.FEATURES

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
