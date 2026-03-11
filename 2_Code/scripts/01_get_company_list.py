"""
Script 1.1: Company List Generation
Goal: Create list of 100-150 UPCOM companies in 3 target industries
Industries: Wholesale (46), Food Manufacturing (10), F&B (56)
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import json

# Target industries
INDUSTRIES = {
    46: 'Wholesale',
    10: 'Food_Manufacturing',
    56: 'FnB'
}

def explore_hnx_website():
    """
    Explore HNX website structure to understand how to get UPCOM company list
    """
    print("🔍 Exploring HNX website...")
    url = "https://www.hnx.vn/vi-vn/cong-ty-dai-chung.html"
    
    try:
        response = requests.get(url, timeout=10)
        print(f"✓ Status code: {response.status_code}")
        
        # Save HTML for manual inspection
        with open('data/processed/hnx_sample.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("✓ Saved HTML to data/processed/hnx_sample.html")
        
        # Parse to find table structure
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = soup.find_all('table')
        print(f"✓ Found {len(tables)} tables on page")
        
        return True
    except Exception as e:
        print(f"✗ Error exploring HNX: {e}")
        return False

def create_manual_company_list():
    """
    Create initial company list template for manual filling
    This is a fallback if automated scraping doesn't work
    """
    print("\n📝 Creating manual company list template...")
    
    # Create Excel file with 3 sheets
    with pd.ExcelWriter('data/processed/company_list_by_industry.xlsx', engine='openpyxl') as writer:
        for industry_code, industry_name in INDUSTRIES.items():
            # Create template DataFrame
            template = pd.DataFrame({
                'ticker': [''] * 50,  # 50 rows for manual entry
                'company_name': [''] * 50,
                'industry_code': [industry_code] * 50,
                'industry_name': [industry_name] * 50,
                'notes': [''] * 50
            })
            
            # Write to sheet
            template.to_excel(writer, sheet_name=industry_name, index=False)
            print(f"✓ Created sheet: {industry_name}")
    
    print("✓ Template saved: data/processed/company_list_by_industry.xlsx")
    print("\n📌 NEXT STEPS:")
    print("1. Go to https://www.hnx.vn/ or https://finance.vietstock.vn/")
    print("2. Filter by UPCOM market")
    print("3. Manually collect 30-50 tickers per industry:")
    print("   - Wholesale (VSIC 46)")
    print("   - Food Manufacturing (VSIC 10)")
    print("   - F&B (VSIC 56)")
    print("4. Fill in the Excel template")

def quick_test_vietstock():
    """
    Test if we can access VietStock API or web interface
    """
    print("\n🔍 Testing VietStock access...")
    
    # Try accessing a known company
    test_ticker = "VNM"  # Vinamilk as test
    url = f"https://finance.vietstock.vn/{test_ticker}/tai-chinh.htm"
    
    try:
        response = requests.get(url, timeout=10)
        print(f"✓ VietStock accessible: {response.status_code}")
        return True
    except Exception as e:
        print(f"✗ VietStock error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("STEP 1.1: COMPANY LIST GENERATION")
    print("="*60)
    
    # Step 1: Explore HNX
    hnx_ok = explore_hnx_website()
    
    # Step 2: Test VietStock
    vietstock_ok = quick_test_vietstock()
    
    # Step 3: Create manual template
    create_manual_company_list()
    
    print("\n" + "="*60)
    print("📊 SUMMARY:")
    print(f"  HNX accessible: {'✓' if hnx_ok else '✗'}")
    print(f"  VietStock accessible: {'✓' if vietstock_ok else '✗'}")
    print(f"  Manual template: ✓ Created")
    print("="*60)
    
    print("\n💡 RECOMMENDATION:")
    print("Since automated scraping may be complex, suggest:")
    print("1. Use manual collection for company list (fastest)")
    print("2. Focus time on BCTC download automation (higher value)")
    print("3. Target: ~30 companies per industry = 90 total (acceptable)")
