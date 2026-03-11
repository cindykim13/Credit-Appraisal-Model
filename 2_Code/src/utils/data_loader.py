"""
data_loader.py — Load và validate train/val/test splits.

NOTE: load_splits_full() trả về full DataFrames (bao gồm cả 'default')
để preprocessor có thể drop duplicates mà vẫn giữ y đồng bộ.
"""
import pandas as pd
from src.config import TRAIN_PATH, VAL_PATH, TEST_PATH


def load_splits_full(verbose: bool = True):
    """
    Trả về full DataFrames (X+y chưa tách) để preprocessor có thể
    drop duplicates mà vẫn giữ y sync với X.
    Returns: train_df, val_df, test_df
    """
    result = {}
    for name, path in [("train", TRAIN_PATH), ("val", VAL_PATH), ("test", TEST_PATH)]:
        df = pd.read_csv(path)
        _validate(df, name)
        result[name] = df
        if verbose:
            default_rate = df["default"].mean() * 100
            print(f"  [{name}] {len(df):,} rows × {df.shape[1]} cols | default={default_rate:.1f}%")
    return result["train"], result["val"], result["test"]


def split_xy(df: pd.DataFrame):
    """Tách X và y từ một split DataFrame."""
    drop_cols = ["sample_id", "default"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).reset_index(drop=True)
    y = df["default"].astype(int).reset_index(drop=True)
    return X, y


def load_splits(verbose: bool = True):
    """Backward-compatible: trả về X_train, y_train, ... (KHÔNG dùng cho fit_transform)."""
    train_df, val_df, test_df = load_splits_full(verbose=verbose)
    X_train, y_train = split_xy(train_df)
    X_val,   y_val   = split_xy(val_df)
    X_test,  y_test  = split_xy(test_df)
    return X_train, y_train, X_val, y_val, X_test, y_test


def _validate(df: pd.DataFrame, name: str):
    assert "default" in df.columns, f"[{name}] Missing 'default' column"
    assert len(df) > 0, f"[{name}] Empty dataframe"
    assert df["default"].isin([0, 1]).all(), f"[{name}] 'default' contains non-binary values"


if __name__ == "__main__":
    print("Loading splits...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_val   shape: {X_val.shape}")
    print(f"X_test  shape: {X_test.shape}")
