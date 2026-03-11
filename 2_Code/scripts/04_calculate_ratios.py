"""
Script 04: Calculate Financial Ratios
Goal: Transform raw financial data into 12 key ratios
"""

import pandas as pd
import numpy as np

def calculate_ratios(df):
    """
    Calculate 12 financial ratios from raw data
    """
    print("📊 Calculating financial ratios...")
    
    # Make a copy
    ratios_df = df.copy()
    
    # 1. ROA = Net Income / Total Assets
    ratios_df['roa'] = ratios_df['net_income'] / ratios_df['total_assets']
    
    # 2. ROE = Net Income / Equity
    ratios_df['roe'] = ratios_df['net_income'] / ratios_df['equity']
    
    # 3. Profit Margin = Net Income / Revenue
    ratios_df['profit_margin'] = ratios_df['net_income'] / ratios_df['revenue']
    
    # 4. Revenue Growth = (Revenue_t - Revenue_t-1) / Revenue_t-1
    ratios_df = ratios_df.sort_values(['ticker', 'year'])
    ratios_df['revenue_growth'] = ratios_df.groupby('ticker')['revenue'].pct_change()
    ratios_df['revenue_growth'] = ratios_df['revenue_growth'].fillna(0.10)  # Fill first year with 10%
    
    # 5. Current Ratio = Current Assets / Current Liabilities
    ratios_df['current_ratio'] = ratios_df['current_assets'] / ratios_df['current_liabilities']
    
    # 6. Quick Ratio = (Current Assets - Inventory) / Current Liabilities
    ratios_df['quick_ratio'] = (ratios_df['current_assets'] - ratios_df['inventory']) / ratios_df['current_liabilities']
    
    # 7. Debt-to-Equity = Total Debt / Equity
    ratios_df['debt_to_equity'] = ratios_df['total_debt'] / ratios_df['equity']
    
    # 8. Debt-to-Asset = Total Debt / Total Assets
    ratios_df['debt_to_asset'] = ratios_df['total_debt'] / ratios_df['total_assets']
    
    # 9. Asset Turnover = Revenue / Total Assets
    ratios_df['asset_turnover'] = ratios_df['revenue'] / ratios_df['total_assets']
    
    # 10. Inventory Turnover = COGS / Inventory
    ratios_df['inventory_turnover'] = ratios_df['cogs'] / ratios_df['inventory']
    
    # 11. Receivable Turnover = Revenue / Receivables
    ratios_df['receivable_turnover'] = ratios_df['revenue'] / ratios_df['receivables']
    
    # 12. DSCR (Debt Service Coverage Ratio) - Estimated
    # DSCR = Operating Income / Debt Service
    # Approximate: Net Income * 1.5 / (Total Debt * 0.15)
    ratios_df['dscr'] = (ratios_df['net_income'] * 1.5) / (ratios_df['total_debt'] * 0.15 + 1)
    
    print(f"✓ Calculated 12 ratios for {len(ratios_df)} records")
    
    return ratios_df

def remove_outliers(df, ratio_cols):
    """
    Remove extreme outliers (beyond 5 standard deviations)
    """
    print("\n🧹 Removing outliers...")
    original_count = len(df)
    
    df_clean = df.copy()
    
    for ratio in ratio_cols:
        mean = df_clean[ratio].mean()
        std = df_clean[ratio].std()
        
        # Keep only values within 5 std
        lower_bound = mean - 5 * std
        upper_bound = mean + 5 * std
        
        df_clean = df_clean[(df_clean[ratio] >= lower_bound) & (df_clean[ratio] <= upper_bound)]
    
    removed = original_count - len(df_clean)
    print(f"✓ Removed {removed} outlier records ({removed/original_count*100:.1f}%)")
    print(f"✓ Remaining: {len(df_clean)} records")
    
    return df_clean

def main():
    print("="*60)
    print("STEP 2.1: CALCULATE FINANCIAL RATIOS")
    print("="*60)
    
    # Load raw data
    df = pd.read_csv('data/processed/financial_raw.csv')
    print(f"✓ Loaded {len(df)} raw records\n")
    
    # Calculate ratios
    df_ratios = calculate_ratios(df)
    
    # Define 12 ratio columns
    ratio_cols = [
        'roa', 'roe', 'profit_margin', 'revenue_growth',
        'current_ratio', 'quick_ratio', 'debt_to_equity', 'debt_to_asset',
        'asset_turnover', 'inventory_turnover', 'receivable_turnover', 'dscr'
    ]
    
    # Remove outliers
    df_clean = remove_outliers(df_ratios, ratio_cols)
    
    # Select final columns
    final_cols = ['ticker', 'company_name', 'year', 'industry_code'] + ratio_cols
    df_final = df_clean[final_cols]
    
    # Save
    df_final.to_csv('data/processed/financial_features.csv', index=False)
    print(f"\n✓ Saved: data/processed/financial_features.csv")
    
    # Summary statistics
    print("\n📊 Summary Statistics by Industry:")
    for industry in sorted(df_final['industry_code'].unique()):
        subset = df_final[df_final['industry_code'] == industry]
        print(f"\n  Industry {industry}:")
        print(f"    Records: {len(subset)}")
        print(f"    ROA (avg): {subset['roa'].mean()*100:.2f}%")
        print(f"    ROE (avg): {subset['roe'].mean()*100:.2f}%")
        print(f"    Profit Margin (avg): {subset['profit_margin'].mean()*100:.2f}%")
        print(f"    Current Ratio (avg): {subset['current_ratio'].mean():.2f}")
        print(f"    Debt-to-Equity (avg): {subset['debt_to_equity'].mean():.2f}")
    
    print("\n" + "="*60)
    print("✅ Phase 1 Data Collection COMPLETE!")
    print(f"   Final dataset: {len(df_final)} records")
    print(f"   Companies: {df_final['ticker'].nunique()}")
    print(f"   Features: 12 financial ratios")
    print("   Ready for Phase 3: Distribution Analysis")
    print("="*60)

if __name__ == "__main__":
    main()
