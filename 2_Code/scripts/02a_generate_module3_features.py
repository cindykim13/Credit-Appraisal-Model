"""
Script 2a: Generate Missing Features - Module 3 (Business Quality)

Sinh 12 features còn thiếu cho Module 3:
- Owner demographics: owner_education
- Product quality: product_differentiation_score, industry_competition_intensity
- Business relationships: supplier_relationships, customer_concentration, digital_footprint
- Operations: business_age, num_employees, employee_productivity_score,
               revenue_capacity_ratio, financial_data_reliability_score, business_certification_count
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

def generate_module3_features(df):
    """Generate 12 missing features for Module 3"""
    
    print("=" * 80)
    print("GENERATING MODULE 3: BUSINESS QUALITY FEATURES")
    print("=" * 80)
    
    n = len(df)
    
    # =================================================================
    # 1. OWNER DEMOGRAPHICS
    # =================================================================
    print("\n1. Owner Demographics...")
    
    # owner_education: ['High School', 'Vocational', 'Bachelor', 'Master']
    # Ngành sản xuất thường có education cao hơn
    education_mapping = {
        10: [0.10, 0.20, 0.50, 0.20],  # Sản xuất - Bachelor nhiều
        46: [0.15, 0.30, 0.45, 0.10],  # Bán buôn - Vocational/Bachelor
        56: [0.20, 0.35, 0.40, 0.05],  # F&B - Thấp hơn
    }
    
    education_levels = ['High School', 'Vocational', 'Bachelor', 'Master']
    owner_education = []
    
    for industry in df['industry_code']:
        probs = education_mapping.get(industry, [0.25, 0.25, 0.25, 0.25])
        edu = np.random.choice(education_levels, p=probs)
        owner_education.append(edu)
    
    df['owner_education'] = owner_education
    
    # =================================================================
    # 2. PRODUCT QUALITY
    # =================================================================
    print("2. Product Quality...")
    
    # product_differentiation_score [1-10]
    # Correlation với profit_margin (sản phẩm độc đáo thường margin cao)
    base_differentiation = np.random.uniform(3, 8, n)
    
    if 'profit_margin' in df.columns:
        # Normalize profit margin to [0,1]
        pm_normalized = (df['profit_margin'] - df['profit_margin'].min()) / \
                       (df['profit_margin'].max() - df['profit_margin'].min())
        # Add correlation
        df['product_differentiation_score'] = (base_differentiation + pm_normalized * 3).clip(1, 10)
    else:
        df['product_differentiation_score'] = base_differentiation
    
    # industry_competition_intensity [1-10]
    # F&B highest competition (10), Wholesale medium (6-7), Manufacturing lower (5)
    competition_by_industry = {
        10: np.random.uniform(4, 6, n),    # Sản xuất - ít cạnh tranh hơn
        46: np.random.uniform(6, 8, n),    # Bán buôn - cạnh tranh khá cao
        56: np.random.uniform(8, 10, n),   # F&B - cạnh tranh rất cao
    }
    
    df['industry_competition_intensity'] = df['industry_code'].map(
        lambda x: np.random.choice(competition_by_industry.get(x, np.random.uniform(5, 8, 1)))
    )
    
    # =================================================================
    # 3. BUSINESS RELATIONSHIPS
    # =================================================================
    print("3. Business Relationships...")
    
    # supplier_relationships [1-10]
    # Correlation với business_type và industry
    # Sản xuất thường có quan hệ nhà cung cấp tốt hơn
    supplier_base = np.random.uniform(4, 9, n)
    industry_bonus = df['industry_code'].map({10: 1.5, 46: 1.0, 56: 0.5})
    df['supplier_relationships'] = (supplier_base + industry_bonus).clip(1, 10)
    
    # customer_concentration [0-1]
    # % doanh thu từ top 3 khách hàng
    # Bán buôn thường cao (B2B), F&B thấp (B2C)
    concentration_by_industry = {
        10: np.random.uniform(0.20, 0.50, n),  # Sản xuất - moderate
        46: np.random.uniform(0.40, 0.70, n),  # Bán buôn - high (B2B)
        56: np.random.uniform(0.10, 0.30, n),  # F&B - low (B2C)
    }
    
    df['customer_concentration'] = df['industry_code'].map(
        lambda x: np.random.choice(concentration_by_industry.get(x, np.random.uniform(0.2, 0.5, 1)))
    )
    
    # digital_footprint [1-10]
    # F&B cao nhất (social media), Sản xuất thấp nhất
    digital_by_industry = {
        10: np.random.uniform(3, 6, n),    # Sản xuất - thấp
        46: np.random.uniform(5, 8, n),    # Bán buôn - moderate
        56: np.random.uniform(7, 10, n),   # F&B - cao
    }
    
    df['digital_footprint'] = df['industry_code'].map(
        lambda x: np.random.choice(digital_by_industry.get(x, np.random.uniform(5, 8, 1)))
    )
    
    # =================================================================
    # 4. OPERATIONAL CAPACITY
    # =================================================================
    print("4. Operational Capacity...")
    
    # business_age [0-50 years]
    # Lognormal distribution (nhiều DN trẻ)
    df['business_age'] = np.random.lognormal(mean=2.0, sigma=0.8, size=n).astype(int).clip(0, 50)
    
    # num_employees [5-200]
    # Correlation với revenue
    if 'revenue' in df.columns:
        # revenue → employees (log scale)
        log_revenue = np.log1p(df['revenue'])
        base_employees = 5 + (log_revenue - log_revenue.min()) / (log_revenue.max() - log_revenue.min()) * 150
        noise = np.random.normal(0, 20, n)
        df['num_employees'] = (base_employees + noise).clip(5, 200).astype(int)
    else:
        df['num_employees'] = np.random.lognormal(mean=3.0, sigma=0.8, size=n).astype(int).clip(5, 200)
    
    # employee_productivity_score [1-10]
    # Revenue per employee (normalized)
    if 'revenue' in df.columns and 'num_employees' in df.columns:
        revenue_per_employee = df['revenue'] / df['num_employees']
        # Normalize to [1, 10]
        min_rpe, max_rpe = revenue_per_employee.quantile([0.05, 0.95])
        df['employee_productivity_score'] = ((revenue_per_employee - min_rpe) / (max_rpe - min_rpe) * 9 + 1).clip(1, 10)
    else:
        df['employee_productivity_score'] = np.random.uniform(3, 8, n)
    
    # revenue_capacity_ratio [0-1]
    # Actual revenue / Potential revenue
    # Phân phối beta (skewed towards 0.5-0.8)
    df['revenue_capacity_ratio'] = np.random.beta(a=5, b=2, size=n)
    
    # financial_data_reliability_score [1-10]
    # Correlation với business_age (DN lớn tuổi thường reliable hơn)
    age_normalized = (df['business_age'] / df['business_age'].max()).fillna(0.5)
    base_reliability = np.random.uniform(4, 9, n)
    df['financial_data_reliability_score'] = (base_reliability + age_normalized * 2).clip(1, 10)
    
    # business_certification_count [0-5]
    # ISO, HACCP, etc. - Sản xuất cao nhất
    cert_by_industry = {
        10: np.random.poisson(lam=2.0, size=n),  # Sản xuất - nhiều cert
        46: np.random.poisson(lam=1.0, size=n),  # Bán buôn - ít hơn
        56: np.random.poisson(lam=0.5, size=n),  # F&B - ít nhất
    }
    
    df['business_certification_count'] = df['industry_code'].map(
        lambda x: np.random.choice(cert_by_industry.get(x, np.random.poisson(lam=1.0, 1)))
    ).clip(0, 5).astype(int)
    
    print("\n✅ Module 3 features generated successfully!")
    
    new_features = [
        'owner_education', 'product_differentiation_score', 'industry_competition_intensity',
        'supplier_relationships', 'customer_concentration', 'digital_footprint',
        'business_age', 'num_employees', 'employee_productivity_score',
        'revenue_capacity_ratio', 'financial_data_reliability_score', 'business_certification_count'
    ]
    
    print(f"\nFeatures added ({len(new_features)}):")
    for feat in new_features:
        print(f"  ✓ {feat}")
    
    return df

if __name__ == "__main__":
    # Load cleaned data
    base_dir = Path("d:/SystemFolders/Desktop/NCKH/data_organized")
    df = pd.read_csv(base_dir / "02_cleaned" / "full_3k_cleaned.csv")
    
    print(f"\nOriginal shape: {df.shape}")
    
    # Generate Module 3 features
    df = generate_module3_features(df)
    
    print(f"\nNew shape: {df.shape}")
    print(f"Features added: {df.shape[1] - 55}")
    
    # Save
    output_path = base_dir / "02_cleaned" / "with_module3.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved: {output_path}")
