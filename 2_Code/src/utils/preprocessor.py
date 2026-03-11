"""
preprocessor.py — Xử lý data quality issues + encoding + scaling.

Issues được fix:
  A. collateral_type: 1316/2164 NaN (60.8%) → impute 'None' khi has_collateral=0
  B. Duplicates: 44 rows → drop
  C. debt_burden_ratio: 92.2% = 3.0 (hard-capped) → tính lại hoặc drop
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ── Categorical maps ──────────────────────────────────────────────────────────
EDUCATION_MAP = {"THPT": 1, "CĐ": 2, "ĐH": 3, "Master": 4}

# Columns cần log-transform (đơn vị VND, phân phối lệch phải)
LOG_TRANSFORM_COLS = [
    "free_cash_flow", "net_cash_flow",           # có thể âm → dùng log1p(x + shift)
    "total_outstanding_debt", "avg_daily_balance",
    "min_balance_3m", "collateral_value",
    "avg_monthly_deposits", "avg_monthly_withdrawals",
]


class DataPreprocessor:
    """
    Fit trên train, apply lên val/test (no leakage).
    
    Usage:
        prep = DataPreprocessor()
        X_train_clean = prep.fit_transform(X_train)
        X_val_clean   = prep.transform(X_val)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._log_shifts: dict = {}      # shift values cho log-transform
        self._debt_burden_strategy = None  # 'recalc' hoặc 'drop'
        self.feature_names_out_: list = []
        self._fitted = False
        self._train_medians: dict = {}   # for imputation on val/test

    # ─────────────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────────────

    def fit_transform(self, full_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Nhận full DataFrame (gồm cả cột 'default'), trả về (X_clean, y_clean).
        Drop duplicates ở đây để y luôn sync với X.
        """
        df = full_df.copy()
        # Issue B: drop duplicates trên toàn bộ df (X + y)
        df = self._fix_duplicates(df)

        y = df["default"].astype(int).reset_index(drop=True) if "default" in df.columns else None
        drop_cols = [c for c in ["sample_id", "default"] if c in df.columns]
        X = df.drop(columns=drop_cols).reset_index(drop=True)

        X = self._fix_collateral_type(X)
        X = self._fix_debt_burden_ratio(X, is_fit=True)
        X = self._encode_categoricals(X)
        X = self._log_transform(X, is_fit=True)
        X = self._impute_remaining(X, is_fit=True)
        self._fitted = True
        self.feature_names_out_ = list(X.columns)
        if self.verbose:
            print(f"  [Preprocessor] fit_transform done → X{X.shape}, y{y.shape}")
        return X, y

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply transform đã fit lên val/test (không fit lại)."""
        assert self._fitted, "Gọi fit_transform trước!"
        # strip non-feature cols
        drop_cols = [c for c in ["sample_id", "default"] if c in X.columns]
        X = X.drop(columns=drop_cols, errors="ignore").copy().reset_index(drop=True)
        # Không drop duplicates trên val/test
        X = self._fix_collateral_type(X)
        X = self._fix_debt_burden_ratio(X, is_fit=False)
        X = self._encode_categoricals(X)
        X = self._log_transform(X, is_fit=False)
        X = self._impute_remaining(X, is_fit=False)
        # Đảm bảo cùng columns như train
        X = self._align_columns(X)
        return X

    # ─────────────────────────────────────────────────────────────────────────
    # Issue A: collateral_type missing values
    # ─────────────────────────────────────────────────────────────────────────

    def _fix_collateral_type(self, X: pd.DataFrame) -> pd.DataFrame:
        if "collateral_type" not in X.columns:
            return X
        before = X["collateral_type"].isna().sum()
        if "has_collateral" in X.columns:
            # has_collateral=0 → không có tài sản → type = 'None'
            mask_no_collateral = X["has_collateral"] == 0
            X.loc[mask_no_collateral & X["collateral_type"].isna(), "collateral_type"] = "None"
        # Phần còn lại (has_collateral=1 nhưng type NaN) → mode imputation
        if X["collateral_type"].isna().any():
            mode_val = X["collateral_type"].mode()[0]
            X["collateral_type"] = X["collateral_type"].fillna(mode_val)
        after = X["collateral_type"].isna().sum()
        if self.verbose:
            print(f"  [Issue A] collateral_type NaN: {before} → {after}")
        return X

    # ─────────────────────────────────────────────────────────────────────────
    # Issue B: Duplicates (chỉ áp dụng trên train)
    # ─────────────────────────────────────────────────────────────────────────

    def _fix_duplicates(self, X: pd.DataFrame) -> pd.DataFrame:
        n_before = len(X)
        X = X.drop_duplicates()
        n_after = len(X)
        if self.verbose and n_before != n_after:
            print(f"  [Issue B] Duplicates dropped: {n_before - n_after} rows ({n_before} → {n_after})")
        return X

    # ─────────────────────────────────────────────────────────────────────────
    # Issue C: debt_burden_ratio hard-capped at 3.0
    # ─────────────────────────────────────────────────────────────────────────

    def _fix_debt_burden_ratio(self, X: pd.DataFrame, is_fit: bool) -> pd.DataFrame:
        col = "debt_burden_ratio"
        if col not in X.columns:
            return X

        if is_fit:
            pct_at_cap = (X[col] == 3.0).mean()
            if self.verbose:
                print(f"  [Issue C] debt_burden_ratio — {pct_at_cap*100:.1f}% = 3.0 (hard-cap)")

            # Thử recalculate từ raw columns
            if "total_outstanding_debt" in X.columns and "avg_monthly_deposits" in X.columns:
                annual_revenue_proxy = X["avg_monthly_deposits"] * 12
                recalc = (X["total_outstanding_debt"] / annual_revenue_proxy.replace(0, np.nan)).clip(0, 10)
                corr = X[col].corr(recalc)
                if self.verbose:
                    print(f"  [Issue C] Correlation (original vs recalculated): {corr:.3f}")
                if not np.isnan(corr) and abs(corr) > 0.3:
                    self._debt_burden_strategy = "recalc"
                    if self.verbose:
                        print(f"  [Issue C] Strategy: RECALCULATE (corr={corr:.3f})")
                else:
                    self._debt_burden_strategy = "drop"
                    if self.verbose:
                        print(f"  [Issue C] Strategy: DROP (corr too low = {corr:.3f})")
            else:
                self._debt_burden_strategy = "drop"

        # Apply strategy
        if self._debt_burden_strategy == "recalc" and "avg_monthly_deposits" in X.columns:
            annual_revenue_proxy = X["avg_monthly_deposits"] * 12
            X[col] = (X["total_outstanding_debt"] / annual_revenue_proxy.replace(0, np.nan)).clip(0, 10)
            X[col] = X[col].fillna(X[col].median())
        elif self._debt_burden_strategy == "drop":
            X = X.drop(columns=[col], errors="ignore")
            if self.verbose and is_fit:
                print(f"  [Issue C] Dropped '{col}' from feature set")

        return X

    # ─────────────────────────────────────────────────────────────────────────
    # Encoding
    # ─────────────────────────────────────────────────────────────────────────

    def _encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        # Ordinal: owner_education
        if "owner_education" in X.columns:
            X["owner_education"] = X["owner_education"].map(EDUCATION_MAP).fillna(2)

        # One-hot: business_zone, industry_lifecycle_stage, collateral_type
        for col in ["business_zone", "industry_lifecycle_stage", "collateral_type"]:
            if col in X.columns:
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=False)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)

        return X

    # ─────────────────────────────────────────────────────────────────────────
    # Log-transform
    # ─────────────────────────────────────────────────────────────────────────

    def _log_transform(self, X: pd.DataFrame, is_fit: bool) -> pd.DataFrame:
        """Log1p transform sau khi shift để xử lý giá trị âm."""
        for col in LOG_TRANSFORM_COLS:
            if col not in X.columns:
                continue
            if is_fit:
                min_val = X[col].min()
                shift = abs(min_val) + 1 if min_val < 0 else 0
                self._log_shifts[col] = shift
            shift = self._log_shifts.get(col, 0)
            X[col] = np.log1p(X[col] + shift)
        return X

    # ─────────────────────────────────────────────────────────────────────────
    # Remaining missing values (safety net)
    # ─────────────────────────────────────────────────────────────────────────

    def _impute_remaining(self, X: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if X[numeric_cols].isna().any().any():
            if is_fit:
                self._train_medians = X[numeric_cols].median().to_dict()
            medians = pd.Series(self._train_medians)
            for col in numeric_cols:
                if col in medians:
                    X[col] = X[col].fillna(medians[col])
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")
        return X

    # ─────────────────────────────────────────────────────────────────────────
    # Align columns (val/test phải có cùng columns như train)
    # ─────────────────────────────────────────────────────────────────────────

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.feature_names_out_:
            if col not in X.columns:
                X[col] = 0   # missing one-hot column → fill 0
        return X[self.feature_names_out_]


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.utils.data_loader import load_splits_full, split_xy

    print("=== Preprocessor Smoke Test ===")
    train_df, val_df, test_df = load_splits_full()

    prep = DataPreprocessor(verbose=True)
    X_train, y_train = prep.fit_transform(train_df)
    X_val            = prep.transform(val_df)
    X_test           = prep.transform(test_df)

    _, y_val  = split_xy(val_df)
    _, y_test = split_xy(test_df)

    print(f"\n  Train: {X_train.shape}, y={y_train.shape}")
    print(f"  Val:   {X_val.shape},   y={y_val.shape}")
    print(f"  Test:  {X_test.shape},  y={y_test.shape}")

    remaining_nan = X_train.isna().sum().sum()
    print(f"\n  Remaining NaN in train: {remaining_nan}")
    assert remaining_nan == 0, "FAIL: Con NaN!"
    assert len(X_train) == len(y_train), f"FAIL: X/y mismatch {len(X_train)} vs {len(y_train)}"

    print("\nPreprocessor OK")
