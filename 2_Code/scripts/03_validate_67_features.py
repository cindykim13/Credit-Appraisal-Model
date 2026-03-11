"""
Script 3: Validate and Map to 67 Required Features
"""
import pandas as pd
from pathlib import Path

def validate_67_features():
    """Validate that we have all 67 required features"""
    
    # Expected features theo plan
    required_features = {
        'Module 1: Financial Health': [
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
        'Module 2: Credit Behavior': [
            'cic_score', 'num_active_loans', 'total_outstanding_debt',
            'max_past_due_days', 'num_past_due_30d', 'num_past_due_90d',
            'debt_burden_ratio', 'credit_history_length', 'previous_default_history',
            'num_transactions_3m', 'avg_monthly_deposits', 'avg_monthly_withdrawals',
            'transaction_regularity_score'
        ],
        'Module 3: Business Quality': [
            'owner_age', 'owner_education', 'owner_experience',
            'product_differentiation_score', 'industry_competition_intensity',
            'supplier_relationships', 'customer_concentration', 'digital_footprint',
            'business_age', 'num_employees', 'employee_productivity_score',
            'revenue_capacity_ratio', 'financial_data_reliability_score',
            'business_certification_count'
        ],
        'Module 4: Stability & Risk': [
            'founding_year', 'years_at_current_location', 'industry_changes_count',
            'past_bankruptcy', 'ownership_stability',
            'industry_code', 'industry_risk_score', 'industry_lifecycle_stage',
            'district_code', 'district_risk_score', 'business_zone',
            'district_business_density', 'avg_income_district',
            'has_collateral', 'collateral_value', 'loan_to_value',
            'collateral_liquidity_score', 'collateral_type'
        ]
    }
    
    # Load generated data
    base_dir = Path("d:/SystemFolders/Desktop/NCKH/data_organized")
    df = pd.read_csv(base_dir / "02_cleaned" / "full_with_all_features.csv")
    
    current_features = set(df.columns.tolist())
    
    print("=" * 80)
    print("VALIDATION: 67 REQUIRED FEATURES")
    print("=" * 80)
    
    all_found = []
    all_missing = []
    
    for module, features in required_features.items():
        found = [f for f in features if f in current_features]
        missing = [f for f in features if f not in current_features]
        
        all_found.extend(found)
        all_missing.extend(missing)
        
        coverage = len(found) / len(features) * 100
        status = "✅" if coverage == 100 else "⚠️" if coverage >= 80 else "❌"
        
        print(f"\n{status} {module}")
        print(f"   Found: {len(found)}/{len(features)} ({coverage:.1f}%)")
        if missing:
            print(f"   Missing: {', '.join(missing)}")
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {len(all_found)}/67 features ({len(all_found)/67*100:.1f}%)")
    print("=" * 80)
    
    if len(all_missing) == 0:
        print("\n🎉 SUCCESS! All 67 required features are present!")
        
        # Create final dataset with only required features + metadata
        metadata_cols = ['sample_id', 'default']
        final_cols = metadata_cols + all_found
        
        # Ensure columns exist
        final_cols = [c for c in final_cols if c in df.columns]
        
        df_final = df[final_cols]
        
        output_path = base_dir / "03_final" / "dataset_67_features_3k.csv"
        df_final.to_csv(output_path, index=False)
        
        print(f"\n💾 Final dataset saved: {output_path}")
        print(f"   Shape: {df_final.shape}")
        
        return True, df_final
    else:
        print(f"\n⚠️  Still missing {len(all_missing)} features:")
        for feat in all_missing:
            print(f"   - {feat}")
        return False, df
    
if __name__ == "__main__":
    success, df = validate_67_features()
    
    if success:
        print("\n" + "=" * 80)
        print("✅ PHASE 2 COMPLETE: All required features generated")
        print("=" * 80)
        print("\nReady for Phase 3: Increase to 5,000 samples")
    else:
        print("\n⚠️  Need to generate remaining features")
