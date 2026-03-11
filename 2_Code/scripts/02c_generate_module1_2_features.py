"""
Script 2c: Generate Missing Features - Modules 1 & 2

Module 1 (5 features):
- free_cash_flow, operating_cash_flow_ratio, cash_conversion_cycle,
  days_sales_outstanding, days_payables_outstanding

Module 2 (2 features):
- previous_default_history, transaction_regularity_score
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed( 42)

def generate_module1_2_features(df):
    """Generate 7 missing features for Modules 1 & 2"""
    
    print("=" * 80)
    print("GENERATING MODULES 1 & 2: REMAINING FEATURES")
    print("=" * 80)
    
    n = len(df)
    
    # =================================================================
    # MODULE 1: FINANCIAL HEALTH (5 features)
    # =================================================================
    print("\n📊 Module 1: Financial Health...")
    
    # free_cash_flow
    # OCF - CapEx ≈ net_cash_flow *  (0.7-0.9)
    if 'net_cash_flow' in df.columns:
        fcf_ratio = np.random.uniform(0.7, 0.9, n)
        df['free_cash_flow'] = df['net_cash_flow'] * fcf_ratio
    else:
        df['free_cash_flow'] = np.random.uniform(-50000000, 500000000, n)
    
    # operating_cash_flow_ratio
    # OCF / Current Liabilities (typical 0.3-1.5)
    if 'total_outstanding_debt' in df.columns:
        base_ratio = np.random.uniform(0.4, 1.2, n)
        # Correlation with DSCR
        if 'dscr' in df.columns:
            dscr_factor = (df['dscr'] - 1) / 3  # Normalize
            df['operating_cash_flow_ratio'] = (base_ratio + dscr_factor * 0.3).clip(0.1, 2.0)
        else:
            df['operating_cash_flow_ratio'] = base_ratio
    else:
        df['operating_cash_flow_ratio'] = np.random.uniform(0.3, 1.5, n)
    
    # cash_conversion_cycle (days)
    # CCC = DSO + DIO - DPO
    # Typically 30-90 days
    
    # days_sales_outstanding (DSO)
    # 365 / Receivable Turnover
    if 'receivable_turnover' in df.columns:
        df['days_sales_outstanding'] = (365 / df['receivable_turnover']).clip(10, 120)
    else:
        df['days_sales_outstanding'] = np.random.uniform(30, 90, n)
    
    # days_inventory_outstanding (DIO) - từ inventory_turnover
    if 'inventory_turnover' in df.columns:
        days_inventory_outstanding = (365 / df['inventory_turnover']).clip(10, 180)
    else:
        days_inventory_outstanding = np.random.uniform(30, 120, n)
    
    # days_payables_outstanding (DPO)
    # Typically 30-60 days
    # Bán buôn thường cao hơn (negotiate better terms)
    dpo_by_industry = {
        10: np.random.uniform(30, 50, n),
        46: np.random.uniform(40, 70, n),  # Wholesale - higher
        56: np.random.uniform(25, 45, n),
    }
    
    df['days_payables_outstanding'] = df['industry_code'].map(
        lambda x: np.random.choice(dpo_by_industry.get(x, np.random.uniform(30, 60, 1)))
    )
    
    # cash_conversion_cycle
    df['cash_conversion_cycle'] = (
        df['days_sales_outstanding'] + 
        days_inventory_outstanding - 
        df['days_payables_outstanding']
    ).clip(-30, 180)
    
    print("  ✓ Module 1: 5 features added")
    
    # =================================================================
    # MODULE 2: CREDIT BEHAVIOR (2 features)
    # =================================================================
    print("\n💳 Module 2: Credit Behavior...")
    
    # previous_default_history [0/1]
    # Correlation with max_past_due_days and current default
    if 'max_past_due_days' in df.columns:
        # 20% probability if max_past_due > 90
        # 5% probability if max_past_due < 30
        prob_default = np.where(df['max_past_due_days'] > 90, 0.20, 0.05)
        df['previous_default_history'] = np.random.binomial(1, prob_default)
    else:
        df['previous_default_history'] = np.random.choice([0, 1], size=n, p=[0.90, 0.10])
    
    # transaction_regularity_score [1-10]
    # Based on num_transactions and volatility
    if 'num_transactions_3m' in df.columns and 'cash_flow_volatility' in df.columns:
        # More transactions + low volatility = high regularity
        tx_normalized = (df['num_transactions_3m'] / df['num_transactions_3m'].max()).fillna(0.5)
        volatility_penalty = df['cash_flow_volatility'] / df['cash_flow_volatility'].max()
        
        df['transaction_regularity_score'] = (
            5 +  # Base
            tx_normalized * 3 -  # More transactions = higher score
            volatility_penalty * 2  # High volatility = lower score
        ).clip(1, 10)
    else:
        df['transaction_regularity_score'] = np.random.uniform(4, 8, n)
    
    print("  ✓ Module 2: 2 features added")
    
    print("\n✅ Modules 1 & 2 features generated successfully!")
    
    new_features = [
        # Module 1
        'free_cash_flow', 'operating_cash_flow_ratio', 'cash_conversion_cycle',
        'days_sales_outstanding', 'days_payables_outstanding',
        # Module 2
        'previous_default_history', 'transaction_regularity_score'
    ]
    
    print(f"\nFeatures added ({len(new_features)}):")
    for feat in new_features:
        print(f"  ✓ {feat}")
    
    return df

if __name__ == "__main__":
    # Load data with Module 3 & 4
    base_dir = Path("d:/SystemFolders/Desktop/NCKH/data_organized")
    df = pd.read_csv(base_dir / "02_cleaned" / "with_module3_4.csv")
    
    print(f"\nOriginal shape: {df.shape}")
    
    # Generate Module 1 & 2 features
    df = generate_module1_2_features(df)
    
    print(f"\nNew shape: {df.shape}")
    print(f"Total features: {df.shape[1]}")
    
    # Save
    output_path = base_dir / "02_cleaned" / "full_67_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved: {output_path}")
    print(f"\n🎉 ALL 67 FEATURES COMPLETE!")
