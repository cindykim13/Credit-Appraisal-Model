"""
Script 2: Generate ALL 31 Missing Features
Simplified version - all in one file
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

def generate_all_missing_features(df):
    """Generate all 31 missing features at once"""
    
    print("=" * 80)
    print("GENERATING 31 MISSING FEATURES")
    print("=" * 80)
    
    n = len(df)
    
    # ========================================================================
    # MODULE 1: FINANCIAL HEALTH (5 features)
    # ========================================================================
    print("\n📊 MODULE 1: Financial Health (5 features)...")
    
    # 1. free_cash_flow
    if 'net_cash_flow' in df.columns:
        df['free_cash_flow'] = df['net_cash_flow'] * np.random.uniform(0.7, 0.9, n)
    else:
        df['free_cash_flow'] = np.random.uniform(-50000000, 500000000, n)
    
    # 2. operating_cash_flow_ratio
    df['operating_cash_flow_ratio'] = np.random.uniform(0.3, 1.5, n)
    
    # 3. days_sales_outstanding
    if 'receivable_turnover' in df.columns:
        df['days_sales_outstanding'] = (365 / df['receivable_turnover']).clip(10, 120)
    else:
        df['days_sales_outstanding'] = np.random.uniform(30, 90, n)
    
    # 4. days_payables_outstanding
    df['days_payables_outstanding'] = np.random.uniform(30, 60, n)
    
    # 5. cash_conversion_cycle
    if 'inventory_turnover' in df.columns:
        dio = (365 / df['inventory_turnover']).clip(10, 180)
    else:
        dio = np.random.uniform(30, 120, n)
    
    df['cash_conversion_cycle'] = (df['days_sales_outstanding'] + dio - df['days_payables_outstanding']).clip(-30, 180)
    
    print("  ✓ 5 features added")
    
    # ========================================================================
    # MODULE 2: CREDIT BEHAVIOR (2 features)
    # ========================================================================
    print("\n💳 MODULE 2: Credit Behavior (2 features)...")
    
    # 1. previous_default_history
    df['previous_default_history'] = np.random.choice([0, 1], size=n, p=[0.90, 0.10])
    
    # 2. transaction_regularity_score
    df['transaction_regularity_score'] = np.random.uniform(4, 8, n)
    
    print("  ✓ 2 features added")
    
    # ========================================================================
    # MODULE 3: BUSINESS QUALITY (12 features)
    # ========================================================================
    print("\n🏢 MODULE 3: Business Quality (12 features)...")
    
    # 1. owner_education
    education_choices = ['High School', 'Vocational', 'Bachelor', 'Master']
    df['owner_education'] = np.random.choice(education_choices, size=n, p=[0.15, 0.30, 0.45, 0.10])
    
    # 2. product_differentiation_score
    df['product_differentiation_score'] = np.random.uniform(3, 8, n)
    
    # 3. industry_competition_intensity
    # F&B cao nhất, Manufacturing thấp nhất
    competition = []
    for ind in df['industry_code']:
        if ind == 56:  # F&B
            competition.append(np.random.uniform(8, 10))
        elif ind == 46:  # Wholesale
            competition.append(np.random.uniform(6, 8))
        else:  # Manufacturing
            competition.append(np.random.uniform(4, 6))
    df['industry_competition_intensity'] = competition
    
    # 4. supplier_relationships
    df['supplier_relationships'] = np.random.uniform(4, 9, n)
    
    # 5. customer_concentration
    df['customer_concentration'] = np.random.uniform(0.2, 0.6, n)
    
    # 6. digital_footprint
    digital = []
    for ind in df['industry_code']:
        if ind == 56:  # F&B - cao
            digital.append(np.random.uniform(7, 10))
        elif ind == 46:  # Wholesale
            digital.append(np.random.uniform(5, 8))
        else:  # Manufacturing - thấp
            digital.append(np.random.uniform(3, 6))
    df['digital_footprint'] = digital
    
    # 7. business_age
    df['business_age'] = np.random.lognormal(mean=2.0, sigma=0.8, size=n).astype(int).clip(1, 50)
    
    # 8. num_employees
    df['num_employees'] = np.random.lognormal(mean=3.0, sigma=0.8, size=n).astype(int).clip(5, 200)
    
    # 9. employee_productivity_score
    df['employee_productivity_score'] = np.random.uniform(3, 8, n)
    
    # 10. revenue_capacity_ratio
    df['revenue_capacity_ratio'] = np.random.beta(a=5, b=2, size=n)
    
    # 11. financial_data_reliability_score
    df['financial_data_reliability_score'] = np.random.uniform(4, 9, n)
    
    # 12. business_certification_count
    cert = []
    for ind in df['industry_code']:
        if ind == 10:  # Manufacturing - nhiều cert
            cert.append(np.random.poisson(lam=2.0))
        elif ind == 46:  # Wholesale
            cert.append(np.random.poisson(lam=1.0))
        else:  # F&B - ít
            cert.append(np.random.poisson(lam=0.5))
    df['business_certification_count'] = np.array(cert).clip(0, 5)
    
    print("  ✓ 12 features added")
    
    # ========================================================================
    # MODULE 4: STABILITY & RISK (12 features)
    # ========================================================================
    print("\n🏗️  MODULE 4: Stability & Risk (12 features)...")
    
    current_year = 2024
    
    # 1. founding_year
    df['founding_year'] = (current_year - df['business_age']).clip(1980, current_year)
    
    # 2. years_at_current_location
    location_ratio = np.random.beta(a=3, b=1, size=n)
    df['years_at_current_location'] = (df['business_age'] * location_ratio).astype(int)
    
    # 3. industry_changes_count
    df['industry_changes_count'] = np.random.poisson(lam=0.3, size=n).clip(0, 5)
    
    # 4. past_bankruptcy
    df['past_bankruptcy'] = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    
    # 5. ownership_stability
    df['ownership_stability'] = np.random.uniform(5, 9, n)
    
    # 6. industry_risk_score
    # Load from reference table
    base_dir = Path("d:/SystemFolders/Desktop/NCKH/data_organized")
    try:
        industry_risk_df = pd.read_csv(base_dir / "01_raw" / "reference" / "industry_risk.csv")
        risk_map = dict(zip(industry_risk_df['vsic_code'], industry_risk_df['risk_score']))
        df['industry_risk_score'] = df['industry_code'].map(risk_map)
    except:
        # Fallback
        risk_scores = []
        for ind in df['industry_code']:
            if ind == 56:
                risk_scores.append(8)
            elif ind == 46:
                risk_scores.append(6)
            else:
                risk_scores.append(4)
        df['industry_risk_score'] = risk_scores
    
    # 7. industry_lifecycle_stage
    lifecycle = []
    for ind in df['industry_code']:
        if ind == 56:
            lifecycle.append('Growth')
        else:
            lifecycle.append('Mature')
    df['industry_lifecycle_stage'] = lifecycle
    
    # 8. district_risk_score
    df['district_risk_score'] = np.random.uniform(3, 7, n)
    
    # 9. business_zone
    zones = ['CBD', 'Urban', 'Suburban', 'Industrial']
    df['business_zone'] = np.random.choice(zones, size=n, p=[0.15, 0.40, 0.30, 0.15])
    
    # 10. district_business_density
    density = []
    for zone in df['business_zone']:
        if zone == 'CBD':
            density.append(np.random.uniform(800, 1200))
        elif zone == 'Urban':
            density.append(np.random.uniform(400, 800))
        elif zone == 'Industrial':
            density.append(np.random.uniform(200, 400))
        else:  # Suburban
            density.append(np.random.uniform(100, 300))
    df['district_business_density'] = density
    
    # 11. avg_income_district
    income = []
    for zone in df['business_zone']:
        if zone == 'CBD':
            income.append(np.random.uniform(15, 20))
        elif zone == 'Urban':
            income.append(np.random.uniform(12, 16))
        elif zone == 'Industrial':
            income.append(np.random.uniform(10, 14))
        else:  # Suburban
            income.append(np.random.uniform(8, 12))
    df['avg_income_district'] = income
    
    # 12. collateral_liquidity_score
    df['collateral_liquidity_score'] = np.random.uniform(3, 8, n)
    
    print("  ✓ 12 features added")
    
    return df

def main():
    print("\n" + "=" * 80)
    print("FEATURE GENERATION SCRIPT")
    print("=" * 80 + "\n")
    
    # Load cleaned data
    base_dir = Path("d:/SystemFolders/Desktop/NCKH/data_organized")
    input_path = base_dir / "02_cleaned" / "full_3k_cleaned.csv"
    
    print(f"📂 Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Original shape: {df.shape}")
    
    # Generate features
    df = generate_all_missing_features(df)
    
    print(f"\n📊 New shape: {df.shape}")
    print(f"   Features added: {df.shape[1] - 55}")
    print(f"   Expected total: 86 columns (55 + 31)")
    
    # Save
    output_path = base_dir / "02_cleaned" / "full_with_all_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved: {output_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ SUCCESS!")
    print("=" * 80)
    print(f"\nTotal features: {df.shape[1]}")
    print(f"Total rows: {df.shape[0]:,}")
    print("\nNext step: Run validation script to verify 67 required features")

if __name__ == "__main__":
    main()
