"""
pipeline.py — CreditScoringPipeline: orchestrate 4 specialized modules.

Usage:
    pipeline = CreditScoringPipeline()
    pipeline.fit(train_df, val_df)
    scores = pipeline.predict_scores(val_df)  # → DataFrame [S1, S2, S3, S4]
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.modules.module1_financial import Module1Financial
from src.modules.module2_credit import Module2Credit
from src.modules.module3_business import Module3Business
from src.modules.module4_stability import Module4Stability
from src.utils.data_loader import split_xy
from src.utils.preprocessor import DataPreprocessor


MODULE_PKL = {
    "S1": MODELS_DIR / "module1.pkl",
    "S2": MODELS_DIR / "module2.pkl",
    "S3": MODELS_DIR / "module3.pkl",
    "S4": MODELS_DIR / "module4.pkl",
}

MODULE_NAMES = {
    "S1": "Financial Health",
    "S2": "Credit Behavior",
    "S3": "Business Quality",
    "S4": "Stability & Risk",
}


class CreditScoringPipeline:
    """
    Tầng 1: Load 4 trained modules, transform data, predict S₁–S₄.

    Không retrain modules — chỉ load .pkl và orchestrate.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._preprocessor: DataPreprocessor = None
        self._modules: dict = {}       # {"S1": Module1Financial, ...}
        self._fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame, verbose: bool = None) -> "CreditScoringPipeline":
        """
        Fit preprocessor trên train_df và load 4 module pkl.
        """
        if verbose is not None:
            self.verbose = verbose

        self._log("=== CreditScoringPipeline.fit() ===")

        # 1. Fit preprocessor
        self._preprocessor = DataPreprocessor(verbose=False)
        X_train, y_train = self._preprocessor.fit_transform(train_df)
        self._log(f"  Preprocessor fitted: X{X_train.shape}")

        # 2. Load 4 modules
        for key, path in MODULE_PKL.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Module {key} not found at {path}. "
                    f"Hãy train module trước khi dùng pipeline."
                )
            self._modules[key] = _load_pkl(path)
            self._log(f"  Loaded {key} ({MODULE_NAMES[key]}) ← {path.name}")

        self._fitted = True
        return self

    def predict_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nhận raw DataFrame (có hoặc không có cột 'default'), trả về
        DataFrame với cột [S1, S2, S3, S4], index aligned với df.
        """
        assert self._fitted, "Gọi fit() trước!"
        X = self._preprocessor.transform(df)
        scores = {}
        for key, mod in self._modules.items():
            scores[key] = mod.predict_score(X)
        result = pd.DataFrame(scores, index=df.index.copy())
        # Clip về [0,1] safety
        return result.clip(0.0, 1.0)

    def predict_scores_with_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tiện ích cho score_collector: trả về [S1, S2, S3, S4, default].
        """
        scores = self.predict_scores(df)
        if "default" in df.columns:
            scores["default"] = df["default"].values
        return scores

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self._log(f"  Pipeline saved → {path}")

    @classmethod
    def load(cls, path: Path | str) -> "CreditScoringPipeline":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Private ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


# ── Helper ────────────────────────────────────────────────────────────────────

def _load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)
