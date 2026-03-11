"""
Central config — tất cả paths và constants dùng chung cho src/
"""
from pathlib import Path

# ── Root ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # d:/SystemFolders/Desktop/NCKH

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_FINAL_DIR  = ROOT / "data_organized" / "03_final"
TRAIN_PATH      = DATA_FINAL_DIR / "train.csv"
VAL_PATH        = DATA_FINAL_DIR / "val.csv"
TEST_PATH       = DATA_FINAL_DIR / "test.csv"

LOOKUP_DIR      = ROOT / "data_organized" / "01_raw" / "reference"
INDUSTRY_RISK   = LOOKUP_DIR / "industry_risk.csv"
DISTRICT_LOOKUP = LOOKUP_DIR / "district_lookup.csv"

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR      = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Features per module ───────────────────────────────────────────────────────
MODULE1_FEATURES = [
    "revenue_growth", "profit_margin", "roa", "roe",
    "current_ratio", "quick_ratio", "debt_to_equity", "debt_to_asset", "dscr",
    "inventory_turnover", "receivable_turnover", "asset_turnover",
    "free_cash_flow", "operating_cash_flow_ratio", "cash_conversion_cycle",
    "days_sales_outstanding", "days_payables_outstanding",
    "avg_daily_balance", "min_balance_3m", "cash_flow_volatility",
    "net_cash_flow", "overdraft_count",
]

MODULE2_FEATURES = [
    "cic_score", "num_active_loans", "total_outstanding_debt",
    "max_past_due_days", "num_past_due_30d", "num_past_due_90d",
    "debt_burden_ratio", "credit_history_length", "previous_default_history",
    "num_transactions_3m", "avg_monthly_deposits", "avg_monthly_withdrawals",
    "transaction_regularity_score",
]

MODULE3_FEATURES = [
    "owner_age", "owner_education", "owner_experience",
    "product_differentiation_score", "industry_competition_intensity",
    "supplier_relationships", "customer_concentration", "digital_footprint",
    "business_age", "num_employees", "employee_productivity_score",
    "revenue_capacity_ratio", "financial_data_reliability_score",
    "business_certification_count",
]

MODULE4_FEATURES = [
    "founding_year", "years_at_current_location", "industry_changes_count",
    "past_bankruptcy", "ownership_stability",
    "industry_code", "industry_risk_score", "industry_lifecycle_stage",
    "district_code", "district_risk_score", "business_zone",
    "district_business_density", "avg_income_district",
    "has_collateral", "collateral_value", "loan_to_value",
    "collateral_liquidity_score", "collateral_type",
]

# ── Metrics thresholds ────────────────────────────────────────────────────────
MODULE_R2_TARGET  = 0.65
MODULE_MAE_TARGET = 0.15
META_AUC_TARGET   = 0.80
