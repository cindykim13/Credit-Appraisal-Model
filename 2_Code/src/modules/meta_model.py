"""
meta_model.py — Phase 4: Meta-Model (Tầng 2)

Input:  DataFrame (S1, S2, S3, S4) — intermediate scores từ 4 modules
Output: PD (Probability of Default) [0, 1]

Models:
  A: Logistic Regression  (primary — interpretable coefficients)
  B: XGBoost Classifier   (alternative — capture non-linear S interactions)

Strategy: nếu |ΔAUC| ≤ 0.02 → soft-vote ensemble; nếu > 0.02 → single winner.

Targets: AUC > 0.87, F1 > 0.65, KS > 0.70, Brier < 0.09

Step 2 — Feature Engineering (4 gốc + 4 interaction → 8 features total):
  min_score    = min(S1, S2, S3, S4)           — bottleneck signal
  S1_x_S2      = S1 × S2                        — financial × credit interaction
  score_std    = std(S1, S2, S3, S4)            — inconsistency = uncertainty
  weighted_avg = 0.15·S1 + 0.45·S2 + 0.15·S3 + 0.25·S4  — domain composite

Step 3 — Threshold Optimization:
  best threshold tìm trên val PR curve → maximize F1
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, brier_score_loss,
    confusion_matrix, classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Base score columns (output of 4 modules)
SCORE_COLS = ["S1", "S2", "S3", "S4"]

# All feature columns after engineering (8 total)
FEATURE_COLS = ["S1", "S2", "S3", "S4", "min_score", "S1_x_S2", "score_std", "weighted_avg"]

MODULE_LABELS = {
    "S1": "Financial Health",
    "S2": "Credit Behavior",
    "S3": "Business Quality",
    "S4": "Stability & Risk",
}
RISK_THRESHOLDS = {"Low": 0.05, "Medium": 0.15}   # PD < 0.05 → Low; < 0.15 → Medium; else High


def _ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """KS = max(TPR − FPR) trên ROC curve."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def _risk_grade(pd_score: float) -> str:
    if pd_score < RISK_THRESHOLDS["Low"]:
        return "Low"
    elif pd_score < RISK_THRESHOLDS["Medium"]:
        return "Medium"
    return "High"


def _engineer_features(S: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm 4 interaction features vào DataFrame score.

    Input:  DataFrame chứa cột S1, S2, S3, S4
    Output: DataFrame 8 cột [S1, S2, S3, S4, min_score, S1_x_S2, score_std, weighted_avg]
    """
    df = S[SCORE_COLS].copy()
    df["min_score"]    = df[SCORE_COLS].min(axis=1)
    df["S1_x_S2"]      = df["S1"] * df["S2"]
    df["score_std"]    = df[SCORE_COLS].std(axis=1)
    df["weighted_avg"] = (0.15 * df["S1"] + 0.45 * df["S2"]
                        + 0.15 * df["S3"] + 0.25 * df["S4"])
    return df[FEATURE_COLS]


def _optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Tìm threshold tối ưu từ PR curve trên val set để maximize F1.
    """
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (prec + rec) > 0,
        2 * prec * rec / (prec + rec),
        0.0,
    )
    best_idx = np.argmax(f1_scores[:-1])   # thresholds has len-1 vs prec/rec
    return float(thresholds[best_idx])


class MetaModel:
    """
    Phase 4 Meta-Model — tầng 2 của kiến trúc 2-tầng.

    Usage:
        mm = MetaModel()
        mm.fit(S_train, y_train, S_val, y_val)
        pd_scores = mm.predict_proba(S_val)
        classes   = mm.predict_class(S_val)
        explanation = mm.explain(S_val.iloc[0])
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._scaler = StandardScaler()
        self._model_a: Optional[LogisticRegression] = None   # Logistic Regression
        self._model_b: Optional[XGBClassifier] = None        # XGBoost
        self._final = None          # selected single model (if not ensemble)
        self._ensemble_w: Optional[tuple] = None  # (w_a, w_b)
        self._auc_a: float = 0.0
        self._auc_b: float = 0.0
        self._threshold: float = 0.5   # optimized classification threshold
        self._shap_values: Optional[np.ndarray] = None
        self._shap_explainer = None
        self._feature_names = FEATURE_COLS
        self._fitted = False

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        S_train: pd.DataFrame,
        y_train: pd.Series,
        S_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "MetaModel":
        self._log("\n" + "=" * 60)
        self._log("Training MetaModel (Phase 4 — Step 2+3)")
        self._log(f"  Train: {S_train.shape} | Val: {S_val.shape}")
        self._log(f"  Default rate — train: {y_train.mean():.1%} | val: {y_val.mean():.1%}")
        self._log(f"  Features: {FEATURE_COLS}")

        # ── Step 2: Feature Engineering
        X_tr_eng = _engineer_features(S_train)
        X_vl_eng = _engineer_features(S_val)

        X_tr = self._scaler.fit_transform(X_tr_eng)
        X_vl = self._scaler.transform(X_vl_eng)
        y_tr = y_train.values
        y_vl = y_val.values

        # ── Model A: Logistic Regression
        self._model_a, self._auc_a = self._train_lr(X_tr, y_tr, X_vl, y_vl)

        # ── Model B: XGBoost Classifier
        self._model_b, self._auc_b = self._train_xgb(X_tr, y_tr, X_vl, y_vl)

        # ── Strategy decision
        self._decide_strategy(X_vl, y_vl)
        self._fitted = True

        # ── Step 3: Threshold Optimization
        y_prob_val = self._predict_proba_from_X(X_vl)
        self._threshold = _optimize_threshold(y_vl, y_prob_val)
        f1_opt = f1_score(y_vl, (y_prob_val >= self._threshold).astype(int), zero_division=0)
        self._log(f"\n  === Threshold Optimization ===")
        self._log(f"  Best threshold = {self._threshold:.4f} (F1_val = {f1_opt:.4f})")

        # ── SHAP (optional — best effort)
        self._compute_shap(X_vl)

        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_proba(self, S: pd.DataFrame) -> np.ndarray:
        """Trả về PD score ∈ [0, 1] cho mỗi row."""
        assert self._fitted
        X = self._scaler.transform(_engineer_features(S))
        return np.clip(self._predict_proba_from_X(X), 0.0, 1.0)

    def predict_class(self, S: pd.DataFrame) -> np.ndarray:
        """Phân loại 0/1 dùng optimized threshold."""
        return (self.predict_proba(S) >= self._threshold).astype(int)

    def predict_grade(self, S: pd.DataFrame) -> list[str]:
        """Trả về risk grade ["Low", "Medium", "High"]."""
        return [_risk_grade(p) for p in self.predict_proba(S)]

    # ── Validate ─────────────────────────────────────────────────────────────

    def validate(self, S_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """Tính đầy đủ metrics trên val set (dùng optimized threshold cho F1)."""
        y_prob = self.predict_proba(S_val)
        y_pred = (y_prob >= self._threshold).astype(int)
        y_true = y_val.values

        auc   = roc_auc_score(y_true, y_prob)
        f1    = f1_score(y_true, y_pred, zero_division=0)
        ks    = _ks_statistic(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)

        metrics = {
            "auc":       round(auc,   4),
            "f1":        round(f1,    4),
            "ks":        round(ks,    4),
            "brier":     round(brier, 4),
            "threshold": round(self._threshold, 4),
        }
        self._log(f"\n  === Validation ===")
        for k, v in metrics.items():
            self._log(f"  {k.upper():10s} = {v}")
        return metrics

    # ── Explain ──────────────────────────────────────────────────────────────

    def explain(self, S_row: pd.Series | dict) -> dict:
        """
        Sinh natural language explanation cho 1 record.

        Returns dict:
          pd_score, risk_grade, scores, weakest_module, explanation
        """
        assert self._fitted
        if isinstance(S_row, dict):
            S_row = pd.Series(S_row)
        S_df = pd.DataFrame([S_row[SCORE_COLS]])
        pd_score = float(self.predict_proba(S_df)[0])
        grade = _risk_grade(pd_score)

        scores = {col: round(float(S_row[col]), 3) for col in SCORE_COLS}
        weakest_key = min(scores, key=scores.get)
        weakest_name = MODULE_LABELS[weakest_key]
        weakest_val = scores[weakest_key]

        # Module-specific reason templates
        reasons = {
            "S1": f"chỉ số tài chính yếu ({'ROA/DSCR/Current Ratio thấp'})",
            "S2": f"lịch sử tín dụng rủi ro ({'nợ quá hạn hoặc tiền sử default'})",
            "S3": f"chất lượng kinh doanh thấp ({'kinh nghiệm/sản phẩm/năng suất yếu'})",
            "S4": f"ổn định & rủi ro cao ({'ngành/vùng rủi ro hoặc thiếu tài sản đảm bảo'})",
        }

        explanation = (
            f"Điểm PD = {pd_score:.1%} → Hạng {grade}. "
            f"Module yếu nhất: {weakest_name} ({weakest_val:.3f}) — "
            f"{reasons[weakest_key]}."
        )

        return {
            "pd_score":       round(pd_score, 4),
            "risk_grade":     grade,
            "scores":         scores,
            "weakest_module": f"{weakest_key} — {weakest_name}",
            "explanation":    explanation,
        }

    # ── SHAP ────────────────────────────────────────────────────────────────

    def get_shap_summary(self) -> Optional[pd.Series]:
        """Trả về mean |SHAP| per feature, nếu SHAP đã được compute."""
        if self._shap_values is None:
            return None
        mean_abs = np.abs(self._shap_values).mean(axis=0)
        return pd.Series(mean_abs, index=FEATURE_COLS).sort_values(ascending=False)

    # ── LR Coefficients ─────────────────────────────────────────────────────

    def get_lr_coefficients(self) -> Optional[pd.Series]:
        """Trả về coefficients của Logistic Regression (module weights)."""
        if self._model_a is None:
            return None
        coef = self._model_a.coef_[0]
        return pd.Series(coef, index=FEATURE_COLS).sort_values(ascending=False)

    # ── Save / Load ──────────────────────────────────────────────────────────

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self._log(f"  Saved MetaModel → {path}")

    @classmethod
    def load(cls, path: Path | str) -> "MetaModel":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Private: predict from already-transformed X ───────────────────────────

    def _predict_proba_from_X(self, X: np.ndarray) -> np.ndarray:
        if self._ensemble_w:
            wa, wb = self._ensemble_w
            pa = self._model_a.predict_proba(X)[:, 1]
            pb = self._model_b.predict_proba(X)[:, 1]
            return wa * pa + wb * pb
        return self._final.predict_proba(X)[:, 1]

    # ── Private: train helpers ────────────────────────────────────────────────

    def _train_lr(self, X_tr, y_tr, X_vl, y_vl):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        gs = GridSearchCV(
            LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            param_grid={"C": [0.01, 0.1, 1.0, 10.0]},
            cv=cv, scoring="roc_auc", n_jobs=-1, refit=True,
        )
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        auc = roc_auc_score(y_vl, best.predict_proba(X_vl)[:, 1])
        self._log(f"  Model A (LR):  AUC={auc:.4f}  C={gs.best_params_['C']}")
        coef = pd.Series(best.coef_[0], index=FEATURE_COLS)
        self._log(f"    Coefficients: {coef.to_dict()}")
        return best, auc

    def _train_xgb(self, X_tr, y_tr, X_vl, y_vl):
        scale_pos = int((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        gs = GridSearchCV(
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                scale_pos_weight=scale_pos,
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            ),
            param_grid={
                "n_estimators":  [100, 300],
                "max_depth":     [2, 3, 5],
                "learning_rate": [0.05, 0.1],
            },
            cv=cv, scoring="roc_auc", n_jobs=-1, refit=True,
        )
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        auc = roc_auc_score(y_vl, best.predict_proba(X_vl)[:, 1])
        self._log(f"  Model B (XGB): AUC={auc:.4f}  params={gs.best_params_}")
        fi = pd.Series(best.feature_importances_, index=FEATURE_COLS)
        self._log(f"    Feature importance: {fi.to_dict()}")
        return best, auc

    def _decide_strategy(self, X_vl, y_vl):
        delta = abs(self._auc_a - self._auc_b)
        if delta > 0.02:
            if self._auc_a >= self._auc_b:
                self._final = self._model_a
                self._log(f"  Strategy: SINGLE LR (ΔAUC={delta:.4f}, LR wins)")
            else:
                self._final = self._model_b
                self._log(f"  Strategy: SINGLE XGB (ΔAUC={delta:.4f}, XGB wins)")
        else:
            wa = self._auc_a / (self._auc_a + self._auc_b)
            wb = self._auc_b / (self._auc_a + self._auc_b)
            self._ensemble_w = (wa, wb)
            self._final = None
            pa = self._model_a.predict_proba(X_vl)[:, 1]
            pb = self._model_b.predict_proba(X_vl)[:, 1]
            ens_auc = roc_auc_score(y_vl, wa * pa + wb * pb)
            self._log(f"  Strategy: ENSEMBLE (w_LR={wa:.2f}, w_XGB={wb:.2f}) → AUC={ens_auc:.4f}")

    def _compute_shap(self, X_vl):
        try:
            import shap
            if self._ensemble_w:
                # Dùng model có weight cao hơn để explain
                model = self._model_a if self._ensemble_w[0] >= self._ensemble_w[1] else self._model_b
            else:
                model = self._final

            if isinstance(model, LogisticRegression):
                explainer = shap.LinearExplainer(model, X_vl, feature_perturbation="interventional")
            else:
                explainer = shap.TreeExplainer(model)

            self._shap_values = explainer.shap_values(X_vl)
            self._shap_explainer = explainer
            # Đối với binary classification, TreeExplainer có thể trả về list
            if isinstance(self._shap_values, list):
                self._shap_values = self._shap_values[1]

            shap_summary = self.get_shap_summary()
            self._log(f"\n  SHAP (mean |shap| per feature):")
            for k, v in shap_summary.items():
                label = MODULE_LABELS.get(k, k)
                self._log(f"    {k} ({label}): {v:.4f}")

        except Exception as e:
            self._log(f"  [SHAP skipped: {e}]")
            self._shap_values = None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
