"""
Sample company list for demonstration purposes
Since real UPCOM company collection requires manual work,
this creates a realistic sample dataset for prototyping
"""

import pandas as pd

# Sample companies for each industry
# These are REAL ticker examples from Vietnamese market for realism

SAMPLE_COMPANIES = {
    'Wholesale': [
        # VSIC 46 - Bán buôn
        {'ticker': 'PAN', 'company_name': 'PAN Group', 'industry_code': 46},
        {'ticker': 'DXP', 'company_name': 'Cty CP Cảng Đoạn Xá', 'industry_code': 46},
        {'ticker': 'MLS', 'company_name': 'Cty CP Chăn nuôi Mitraco', 'industry_code': 46},
        # Add 27 more for total 30
    ] + [
        {'ticker': f'W{i:02d}', 'company_name': f'Wholesale Company {i}', 'industry_code': 46}
        for i in range(1, 28)
    ],
    
    'Food_Manufacturing': [
        # VSIC 10 - Sản xuất thực phẩm  
        {'ticker': 'VNM', 'company_name': 'Vinamilk', 'industry_code': 10},
        {'ticker': 'SAB', 'company_name': 'Sabeco', 'industry_code': 10},
        {'ticker': 'MSN', 'company_name': 'Masan Group', 'industry_code': 10},
    ] + [
        {'ticker': f'FO{i:02d}', 'company_name': f'Food Mfg Company {i}', 'industry_code': 10}
        for i in range(1, 28)
    ],
    
    'FnB': [
        # VSIC 56 - F&B
        {'ticker': 'QNS', 'company_name': 'Cty CP Quốc tế Sơn Hà', 'industry_code': 56},
        {'ticker': 'COM', 'company_name': 'Cty CP Vật tư Xăng dầu', 'industry_code': 56},
    ] + [
        {'ticker': f'FB{i:02d}', 'company_name': f'F&B Company {i}', 'industry_code': 56}
        for i in range(1, 29)
    ]
}

def create_sample_company_list():
    """
    Create a sample company list with realistic structure
    """
    print("📝 Creating sample company list...")
    
    with pd.ExcelWriter('data/processed/company_list_by_industry.xlsx', engine='openpyxl') as writer:
        for sheet_name, companies in SAMPLE_COMPANIES.items():
            df = pd.DataFrame(companies)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"✓ Created {sheet_name}: {len(companies)} companies")
    
    print(f"✓ Total companies: {sum(len(c) for c in SAMPLE_COMPANIES.values())}")
    print("✓ Saved: data/processed/company_list_by_industry.xlsx")

if __name__ == "__main__":
    create_sample_company_list()
    print("\n✅ Sample company list ready for BCTC download simulation!")
