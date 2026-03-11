"""
Script 2b: Generate Missing Features - Module 4 (Stability & Risk)

Sinh 12 features còn thiếu cho Module 4:
- Business history: founding_year, years_at_current_location, industry_changes_count, 
                    past_bankruptcy, ownership_stability
- Risk scores: industry_risk_score, industry_lifecycle_stage, district_risk_score,
               business_zone, district_business_density, avg_income_district
- Collateral: collateral_liquidity_score
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

def generate_module4_features(df):
    """Generate 12 missing features for Module 4"""
    
    print("=" * 80)
    print("GENERATING MODULE 4: STABILITY & RISK FEATURES")
    print("=" * 80)
    
    n = len(df)
    current_year = 2024
    
    # =================================================================
    # 1. BUSINESS HISTORY (5 features)
    # =================================================================
    print("\n1. Business History...")
    
    # founding_year [1980-2024]
    # Correlation với business_age
    if 'business_age' in df.columns:
        df['founding_year'] = (current_year - df['business_age']).clip(1980, current_year)
    else:
        # Lognormal distribution (nhiều DN mới)
        ages = np.random.lognormal(mean=2.0, sigma=0.8, size=n).astype(int).clip(0, 50)
        df['founding_year'] = (current_year - ages).clip(1980, current_year)
    
    # years_at_current_location [0-business_age]
    # Usually ≤ business_age
    if 'business_age' in df.columns:
        location_ratio = np.random.beta(a=3, b=1, size=n)  # Skewed toward high values
        df['years_at_current_location'] = (df['business_age'] * location_ratio).astype(int)
    else:
        df['years_at_current_location'] = np.random.lognormal(mean=1.5, sigma=0.7, size=n).astype(int).clip(0, 40)
    
    # industry_changes_count [0-5]
    # Poisson (most SMEs don't change industry)
    df['industry_changes_count'] = np.random.poisson(lam=0.3, size=n).clip(0, 5)
    
    # past_bankruptcy [0/1]
    # 5% có lịch sử phá sản
    df['past_bankruptcy'] = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    
    # ownership_stability [1-10]
    # Higher if no industry changes, no bankruptcy, long at location
    base_stability = np.random.uniform(5, 9, n)
    penalty = (df['industry_changes_count'] * 0.5 + df['past_bankruptcy'] * 3)
    bonus = (df['years_at_current_location'] / df['years_at_current_location'].max() * 2)
    df['ownership_stability'] = (base_stability - penalty + bonus).clip(1, 10)
    
    # =================================================================
    # 2. RISK SCORES FROM LOOKUP TABLES (7 features)
    # =================================================================
    print("\n2. Risk Scores (from reference tables)...")
    
    # Load reference tables
    base_dir = Path("d:/SystemFolders/Desktop/NCKH/data_organized")
    industry_risk_df = pd.read_csv(base_dir / "01_raw" / "reference" / "industry_risk.csv")
    district_lookup_df = pd.read_csv(base_dir / "01_raw" / "reference" / "district_lookup.csv")
    
    print(f"   Industry risk table: {industry_risk_df.shape}")
    print(f"   District lookup table: {district_lookup_df.shape}")
    
    # industry_risk_score
    # Map từ industry_code
    industry_risk_map = dict(zip(industry_risk_df['vsic_code'], industry_risk_df['risk_score']))
    df['industry_risk_score'] = df[' industry_code'].map(industry_risk_map)
    
    # industry_lifecycle_stage
    # ['Emerging', 'Growth', 'Mature', 'Decline']
    lifecycle_map = {
        10: 'Mature',   # Sản xuất TP - mature industry
        46: 'Mature',   # Bán buôn - mature
        56: 'Growth',   # F&B - still growing
    }
    df['industry_lifecycle_stage'] = df['industry_code'].map(lifecycle_map)
    
    # district_risk_score
    # Map từ district_code
    if 'district_code' in df.columns:
        # Tạo random risk scores cho 24 quận (1-10)
        # Quận 1, 3, 5 (trung tâm) - risk thấp (2-4)
        # Quận ngoại thành - risk cao hơn (5-8)
        district_risk_scores = {}
        for district in df['district_code'].unique():
            if district in [1, 3, 5, 10]:  # Central districts
                district_risk_scores[district] = np.random.uniform(2, 4)
            elif district in [2, 4, 6, 7, 8, 11]:  # Semi-central
                district_risk_scores[district] = np.random.uniform(4, 6)
            else:  # Suburban
                district_risk_scores[district] = np.random.uniform(6, 8)
        
        df['district_risk_score'] = df['district_code'].map(district_risk_scores)
    else:
        df['district_risk_score'] = np.random.uniform(3, 7, n)
    
    # business_zone ['CBD', 'Urban', 'Suburban', 'Industrial']
    # Based on district_code
    if 'district_code' in df.columns:
        zone_map = {}
        for district in df['district_code'].unique():
            if district in [1, 3, 5]:
                zone_map[district] = 'CBD'
            elif district in [2, 4, 6, 7, 8, 10, 11]:
                zone_map[district] = 'Urban'
            elif district in [9, 12]:
                zone_map[district] = 'Industrial'
            else:
                zone_map[district] = 'Suburban'
        
        df['business_zone'] = df['district_code'].map(zone_map)
    else:
        df['business_zone'] = np.random.choice(['CBD', 'Urban', 'Suburban', 'Industrial'], size=n)
    
    # district_business_density [businesses per km²]
    # CBD highest, Suburban lowest
    zone_density_map = {
        'CBD': np.random.uniform(800, 1200, n),
        'Urban': np.random.uniform(400, 800, n),
        'Industrial': np.random.uniform(200, 400, n),
        'Suburban': np.random.uniform(100, 300, n),
    }
    
    df['district_business_density'] = df['business_zone'].map(
        lambda zone: np.random.choice(zone_density_map.get(zone, np.random.uniform(300, 700, 1)))
    )
    
    # avg_income_district [million VND/month]
    # CBD highest (~15-20M), Suburban lowest (~8-12M)
    zone_income_map = {
        'CBD': np.random.uniform(15, 20, n),
        'Urban': np.random.uniform(12, 16, n),
        'Industrial': np.random.uniform(10, 14, n),
        'Suburban': np.random.uniform(8, 12, n),
    }
    
    df['avg_income_district'] = df['business_zone'].map(
        lambda zone: np.random.choice(zone_income_map.get(zone, np.random.uniform(10, 15, 1)))
    )
    
    # =================================================================
    # 3. COLLATERAL (1 feature)
    # =================================================================
    print("\n3. Collateral...")
    
    # collateral_liquidity_score [1-10]
    # Dựa trên collateral_type
    if 'collateral_type' in df.columns:
        liquidity_map = {
            'Real Estate': np.random.uniform(6, 9, n),      # High liquidity
            'Machinery': np.random.uniform(4, 7, n),        # Medium
            'Inventory': np.random.uniform(3, 6, n),        # Lower
            'None': np.ones(n),                             # Lowest
        }
        
        df['collateral_liquidity_score'] = df['collateral_type'].map(
            lambda ct: np.random.choice(liquidity_map.get(ct, np.random.uniform(3, 7, 1)))
        )
    else:
        df['collateral_liquidity_score'] = np.random.uniform(3, 8, n)
    
    print("\n✅ Module 4 features generated successfully!")
    
    new_features = [
        'founding_year', 'years_at_current_location', 'industry_changes_count',
        'past_bankruptcy', 'ownership_stability',
        'industry_risk_score', 'industry_lifecycle_stage',
        'district_risk_score', 'business_zone', 'district_business_density', 'avg_income_district',
        'collateral_liquidity_score'
    ]
    
    print(f"\nFeatures added ({len(new_features)}):")
    for feat in new_features:
        print(f"  ✓ {feat}")
    
    return df

if __name__ == "__main__":
    # Load data with Module 3
    base_dir = Path("d:/SystemFolders/Desktop/NCKH/data_organized")
    df = pd.read_csv(base_dir / "02_cleaned" / "with_module3.csv")
    
    print(f"\nOriginal shape: {df.shape}")
    
    # Generate Module 4 features
    df = generate_module4_features(df)
    
    print(f"\nNew shape: {df.shape}")
    print(f"Features added: {df.shape[1] - 67}")
    
    # Save
    output_path = base_dir / "02_cleaned" / "with_module3_4.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved: {output_path}")
