"""
Script 1: Tổ Chức và Clean Dữ Liệu Hiện Có
- Tổng hợp dữ liệu từ 5 nguồn
- Xóa cột Unnamed
- Validate và document features
"""
import pandas as pd
import numpy as np
import shutil
import os
from pathlib import Path

# ============================================================================
# PHASE 1: COPY RAW DATA TO ORGANIZED STRUCTURE
# ============================================================================

def organize_raw_data():
    """Copy dữ liệu gốc vào cấu trúc thư mục chuẩn"""
    print("=" * 80)
    print("PHASE 1: ORGANIZING RAW DATA")
    print("=" * 80)
    
    base_dir = Path("d:/SystemFolders/Desktop/NCKH")
    org_dir = base_dir / "data_organized"
    
    # Create directory structure
    (org_dir / "01_raw" / "financial").mkdir(parents=True, exist_ok=True)
    (org_dir / "01_raw" / "cic").mkdir(parents=True, exist_ok=True)
    (org_dir / "01_raw" / "transaction").mkdir(parents=True, exist_ok=True)
    (org_dir / "01_raw" / "reference").mkdir(parents=True, exist_ok=True)
    (org_dir / "01_raw" / "demographic").mkdir(parents=True, exist_ok=True)
    (org_dir / "02_cleaned").mkdir(parents=True, exist_ok=True)
    (org_dir / "03_final").mkdir(parents=True, exist_ok=True)
    
    # Copy files
    copies = [
        # Financial (Người A)
        ("data/NguoiA/synthetic_financial_3k.csv", "01_raw/financial/"),
        ("data/NguoiA/comprehensive_report.md", "01_raw/financial/"),
        
        # CIC (Người C)
        ("data/NguoiC/output/synthetic_cic_3k.csv", "01_raw/cic/"),
        ("data/NguoiC/output/qa_report.txt", "01_raw/cic/"),
        
        # Transaction (Người D)
        ("data/NguoiD/synthetic_transaction_3k.csv", "01_raw/transaction/"),
        ("data/NguoiD/REPORT_TRANSACTION_MODULE.md", "01_raw/transaction/"),
        
        # Reference (Người B)
        ("data/NguoiB/industry_risk.csv", "01_raw/reference/"),
        ("data/NguoiB/district_lookup.csv", "01_raw/reference/"),
        
        # Demographic (Người E)
        ("data/NguoiE/data/intermediate/synthetic_demographic_3k.csv", "01_raw/demographic/"),
        
        # Integrated (Người E)
        ("data/NguoiE/data/final/full_3k.csv", "01_raw/"),
        ("data/NguoiE/data/final/QA_Report.txt", "01_raw/"),
    ]
    
    for src, dst in copies:
        src_path = base_dir / src
        dst_path = org_dir / dst / src_path.name
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"✅ Copied: {src_path.name}")
        else:
            print(f"⚠️  Missing: {src}")
    
    print("\n✅ Phase 1 Complete: Raw data organized\n")

# ============================================================================
# PHASE 2: CLEAN AND ANALYZE
# ============================================================================

def clean_and_analyze():
    """Clean dữ liệu và phân tích chi tiết"""
    print("=" * 80)
    print("PHASE 2: CLEANING AND ANALYZING")
    print("=" * 80)
    
    org_dir = Path("d:/SystemFolders/Desktop/NCKH/data_organized")
    
    # Load full dataset
    df = pd.read_csv(org_dir / "01_raw" / "full_3k.csv")
    
    print(f"\n📊 Original Data Shape: {df.shape}")
    print(f"   Rows: {df.shape[0]:,}")
    print(f"   Columns: {df.shape[1]}")
    
    # Identify and drop Unnamed columns
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    print(f"\n🗑️  Dropping {len(unnamed_cols)} Unnamed columns:")
    for col in unnamed_cols:
        null_pct = df[col].isnull().sum() / len(df) * 100
        print(f"   - {col}: {null_pct:.1f}% nulls")
    
    df_cleaned = df.drop(columns=unnamed_cols)
    
    print(f"\n✅ Cleaned Data Shape: {df_cleaned.shape}")
    print(f"   Columns remaining: {df_cleaned.shape[1]}")
    
    # Save cleaned version
    output_path = org_dir / "02_cleaned" / "full_3k_cleaned.csv"
    df_cleaned.to_csv(output_path, index=False)
    print(f"\n💾 Saved: {output_path}")
    
    # Analyze feature types
    print("\n" + "=" * 80)
    print("FEATURE ANALYSIS")
    print("=" * 80)
    
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    
    print(f"\n📈 Numeric Features: {len(numeric_cols)}")
    print(f"📋 Categorical Features: {len(categorical_cols)}")
    
    # Check missing values
    print("\n" + "-" * 80)
    print("Missing Values Analysis:")
    print("-" * 80)
    missing = df_cleaned.isnull().sum()
    missing_pct = (missing / len(df_cleaned) * 100).round(2)
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing': missing.values,
        'Percent': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("✅ No missing values!")
    
    # Categorical value counts
    print("\n" + "-" * 80)
    print("Categorical Features Distribution:")
    print("-" * 80)
    for col in categorical_cols[:10]:  # First 10
        unique_count = df_cleaned[col].nunique()
        print(f"\n{col}:")
        print(f"  Unique values: {unique_count}")
        if unique_count <= 10:
            print(df_cleaned[col].value_counts().to_string())
    
    # Industry distribution
    if 'industry_code' in df_cleaned.columns:
        print("\n" + "-" * 80)
        print("INDUSTRY DISTRIBUTION (Critical Check):")
        print("-" * 80)
        industry_dist = df_cleaned['industry_code'].value_counts()
        print(industry_dist)
        print(f"\nPercentages:")
        print((industry_dist / len(df_cleaned) * 100).round(1))
    
    # Default rate
    if 'default' in df_cleaned.columns:
        print("\n" + "-" * 80)
        print("DEFAULT RATE:")
        print("-" * 80)
        default_dist = df_cleaned['default'].value_counts()
        default_rate = df_cleaned['default'].mean() * 100
        print(f"Default rate: {default_rate:.2f}%")
        print(default_dist)
    
    return df_cleaned

# ============================================================================
# PHASE 3: MAP FEATURES TO PLAN
# ============================================================================

def map_features_to_plan(df):
    """Map các features hiện có với 67 features trong plan"""
    print("\n" + "=" * 80)
    print("PHASE 3: MAPPING FEATURES TO PLAN")
    print("=" * 80)
    
    # Expected features theo plan
    expected_features = {
        'Module 1: Financial Health (22)': [
            # Báo cáo tài chính (12)
            'revenue_growth', 'profit_margin', 'roa', 'roe',
            'current_ratio', 'quick_ratio',
            'debt_to_equity', 'debt_to_asset', 'dscr',
            'inventory_turnover', 'receivable_turnover', 'asset_turnover',
            # Dòng tiền (10)
            'free_cash_flow', 'operating_cash_flow_ratio',
            'cash_conversion_cycle', 'days_sales_outstanding', 'days_payables_outstanding',
            'avg_daily_balance', 'min_balance_3m', 'cash_flow_volatility', 
            'net_cash_flow', 'overdraft_count'
        ],
        'Module 2: Credit Behavior (13)': [
            # CIC (9)
            'cic_score', 'num_active_loans', 'total_outstanding_debt',
            'max_past_due_days', 'num_past_due_30d', 'num_past_due_90d',
            'debt_burden_ratio', 'credit_history_length', 'previous_default_history',
            # Transaction (4)
            'num_transactions_3m', 'avg_monthly_deposits', 'avg_monthly_withdrawals',
            'transaction_regularity_score'
        ],
        'Module 3: Business Quality (14)': [
            # Owner (3)
            'owner_age', 'owner_education', 'owner_experience',
            # Product (2)
            'product_differentiation_score', 'industry_competition_intensity',
            # Relationships (3)
            'supplier_relationships', 'customer_concentration', 'digital_footprint',
            # Operations (6)
            'business_age', 'num_employees', 'employee_productivity_score',
            'revenue_capacity_ratio', 'financial_data_reliability_score',
            'business_certification_count'  # placeholder
        ],
        'Module 4: Stability & Risk (18)': [
            # History (5)
            'founding_year', 'years_at_current_location', 'industry_changes_count',
            'past_bankruptcy', 'ownership_stability',
            # Industry/District (8)
            'industry_code', 'industry_risk_score', 'industry_lifecycle_stage',
            'district_code', 'district_risk_score', 'business_zone',
            'district_business_density', 'avg_income_district',
            # Collateral (5)
            'has_collateral', 'collateral_value', 'loan_to_value',
            'collateral_liquidity_score', 'collateral_type'
        ]
    }
    
    # Current features
    current_features = set(df.columns.tolist())
    
    # Mapping results
    mapping_results = {}
    
    print("\nFEATURE MAPPING RESULTS:\n")
    
    for module, features in expected_features.items():
        found = []
        missing = []
        
        for feat in features:
            if feat in current_features:
                found.append(feat)
            else:
                missing.append(feat)
        
        coverage = len(found) / len(features) * 100
        mapping_results[module] = {
            'expected': len(features),
            'found': len(found),
            'missing': len(missing),
            'coverage': coverage,
            'missing_list': missing
        }
        
        status = "✅" if coverage >= 80 else "⚠️" if coverage >= 50 else "❌"
        print(f"{status} {module}")
        print(f"   Found: {len(found)}/{len(features)} ({coverage:.1f}%)")
        if missing:
            print(f"   Missing: {', '.join(missing[:5])}" + ("..." if len(missing) > 5 else ""))
        print()
    
    # Overall summary
    total_expected = sum(r['expected'] for r in mapping_results.values())
    total_found = sum(r['found'] for r in mapping_results.values())
    overall_coverage = total_found / total_expected * 100
    
    print("=" * 80)
    print(f"OVERALL: {total_found}/{total_expected} features ({overall_coverage:.1f}%)")
    print("=" * 80)
    
    # Save mapping report
    org_dir = Path("d:/SystemFolders/Desktop/NCKH/data_organized")
    report_path = org_dir / "02_cleaned" / "feature_mapping_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("FEATURE MAPPING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for module, result in mapping_results.items():
            f.write(f"{module}\n")
            f.write(f"Coverage: {result['coverage']:.1f}%\n")
            f.write(f"Found: {result['found']}/{result['expected']}\n")
            if result['missing_list']:
                f.write(f"Missing features:\n")
                for feat in result['missing_list']:
                    f.write(f"  - {feat}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"TOTAL: {total_found}/{total_expected} ({overall_coverage:.1f}%)\n")
    
    print(f"\n💾 Mapping report saved: {report_path}")
    
    return mapping_results

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n")
    print("=" * 80)
    print("DATA ORGANIZATION AND CLEANING SCRIPT")
    print("=" * 80)
    print("\n")
    
    # Phase 1: Organize
    organize_raw_data()
    
    # Phase 2: Clean and analyze
    df_cleaned = clean_and_analyze()
    
    # Phase 3: Map features
    mapping_results = map_features_to_plan(df_cleaned)
    
    print("\n" + "=" * 80)
    print("✅ ALL PHASES COMPLETED!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review feature_mapping_report.txt")
    print("2. Run script 02 to generate missing features")
    print("\n")

if __name__ == "__main__":
    main()
