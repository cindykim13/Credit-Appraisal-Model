"""
Script: Generate Comprehensive Dataset Documentation
- Data Dictionary (67 features)
- Statistical Summary (mean, std, min, max, percentiles)
- Distribution Analysis
- Correlation Analysis
- Module-wise Statistics
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_data_dictionary():
    """Create comprehensive data dictionary for all 67 features"""
    
    features = {
        # Module 1: Financial Health (22)
        'Module 1: Financial Health': [
            ('revenue_growth', 'float', 'Year-over-year revenue growth rate', '[-0.5, 1.0]', '%'),
            ('profit_margin', 'float', 'Net profit margin ratio', '[0.02, 0.15]', 'ratio'),
            ('roa', 'float', 'Return on Assets', '[0.01, 0.12]', 'ratio'),
            ('roe', 'float', 'Return on Equity', '[0.02, 0.25]', 'ratio'),
            ('current_ratio', 'float', 'Current Assets / Current Liabilities', '[0.5, 2.5]', 'ratio'),
            ('quick_ratio', 'float', 'Quick Assets / Current Liabilities', '[0.4, 2.0]', 'ratio'),
            ('debt_to_equity', 'float', 'Total Debt / Total Equity', '[0.3, 1.5]', 'ratio'),
            ('debt_to_asset', 'float', 'Total Debt / Total Assets', '[0.2, 0.7]', 'ratio'),
            ('dscr', 'float', 'Debt Service Coverage Ratio', '[0.3, 3.5]', 'ratio'),
            ('inventory_turnover', 'float', 'Cost of Goods Sold / Average Inventory', '[2, 13]', 'times/year'),
            ('receivable_turnover', 'float', 'Revenue / Average Receivables', '[2, 11]', 'times/year'),
            ('asset_turnover', 'float', 'Revenue / Total Assets', '[0.3, 1.5]', 'times/year'),
            ('free_cash_flow', 'float', 'Operating Cash Flow - CapEx', '[-50M, 500M]', 'VND'),
            ('operating_cash_flow_ratio', 'float', 'OCF / Current Liabilities', '[0.3, 1.5]', 'ratio'),
            ('cash_conversion_cycle', 'float', 'DSO + DIO - DPO', '[-30, 180]', 'days'),
            ('days_sales_outstanding', 'float', '365 / Receivable Turnover', '[10, 120]', 'days'),
            ('days_payables_outstanding', 'float', 'Days to pay suppliers', '[30, 60]', 'days'),
            ('avg_daily_balance', 'float', 'Average daily account balance', 'varies', 'VND'),
            ('min_balance_3m', 'float', 'Minimum balance in last 3 months', 'varies', 'VND'),
            ('cash_flow_volatility', 'float', 'Standard deviation of cash flows', 'varies', '%'),
            ('net_cash_flow', 'float', 'Deposits - Withdrawals', 'varies', 'VND'),
            ('overdraft_count', 'int', 'Number of overdraft incidents', '[0, 10]', 'count'),
        ],
        
        # Module 2: Credit Behavior (13)
        'Module 2: Credit Behavior': [
            ('cic_score', 'int', 'Credit Information Center score', '[300, 850]', 'score'),
            ('num_active_loans', 'int', 'Number of active credit facilities', '[0, 5]', 'count'),
            ('total_outstanding_debt', 'float', 'Total outstanding debt amount', 'varies', 'VND'),
            ('max_past_due_days', 'int', 'Maximum days past due', '[0, 180]', 'days'),
            ('num_past_due_30d', 'int', 'Number of 30+ days past due', '[0, 10]', 'count'),
            ('num_past_due_90d', 'int', 'Number of 90+ days past due', '[0, 5]', 'count'),
            ('debt_burden_ratio', 'float', 'Total Debt / Revenue', '[0, 2]', 'ratio'),
            ('credit_history_length', 'float', 'Years of credit history', '[0, 20]', 'years'),
            ('previous_default_history', 'int', 'Has previous default (0/1)', '[0, 1]', 'binary'),
            ('num_transactions_3m', 'int', 'Transaction count in 3 months', '[5, 500]', 'count'),
            ('avg_monthly_deposits', 'float', 'Average monthly deposits', 'varies', 'VND'),
            ('avg_monthly_withdrawals', 'float', 'Average monthly withdrawals', 'varies', 'VND'),
            ('transaction_regularity_score', 'float', 'Transaction pattern regularity', '[1, 10]', 'score'),
        ],
        
        # Module 3: Business Quality (14)
        'Module 3: Business Quality': [
            ('owner_age', 'int', 'Age of business owner', '[25, 70]', 'years'),
            ('owner_education', 'str', 'Education level', 'categorical', 'HS/Voc/Bach/Mast'),
            ('owner_experience', 'int', 'Years of industry experience', '[0, 40]', 'years'),
            ('product_differentiation_score', 'float', 'Product uniqueness score', '[1, 10]', 'score'),
            ('industry_competition_intensity', 'float', 'Market competition level', '[1, 10]', 'score'),
            ('supplier_relationships', 'float', 'Supplier relationship quality', '[1, 10]', 'score'),
            ('customer_concentration', 'float', '% revenue from top 3 customers', '[0, 1]', 'ratio'),
            ('digital_footprint', 'float', 'Online presence score', '[1, 10]', 'score'),
            ('business_age', 'int', 'Years since founding', '[1, 50]', 'years'),
            ('num_employees', 'int', 'Number of employees', '[5, 200]', 'count'),
            ('employee_productivity_score', 'float', 'Revenue per employee (normalized)', '[1, 10]', 'score'),
            ('revenue_capacity_ratio', 'float', 'Actual / Potential revenue', '[0, 1]', 'ratio'),
            ('financial_data_reliability_score', 'float', 'Data quality score', '[1, 10]', 'score'),
            ('business_certification_count', 'int', 'Number of certifications (ISO, HACCP)', '[0, 5]', 'count'),
        ],
        
        # Module 4: Stability & Risk (18)
        'Module 4: Stability & Risk': [
            ('founding_year', 'int', 'Year business was founded', '[1980, 2024]', 'year'),
            ('years_at_current_location', 'int', 'Years at current address', '[0, 40]', 'years'),
            ('industry_changes_count', 'int', 'Number of industry changes', '[0, 5]', 'count'),
            ('past_bankruptcy', 'int', 'Has bankruptcy history (0/1)', '[0, 1]', 'binary'),
            ('ownership_stability', 'float', 'Ownership stability score', '[1, 10]', 'score'),
            ('industry_code', 'int', 'VSIC industry code', '{10, 46, 56}', 'categorical'),
            ('industry_risk_score', 'int', 'Industry-level risk', '[4, 8]', 'score'),
            ('industry_lifecycle_stage', 'str', 'Industry maturity', 'categorical', 'Growth/Mature'),
            ('district_code', 'int', 'District code (1-24)', '[1, 24]', 'categorical'),
            ('district_risk_score', 'float', 'District-level risk', '[3, 7]', 'score'),
            ('business_zone', 'str', 'Business location zone', 'categorical', 'CBD/Urban/Suburb/Indust'),
            ('district_business_density', 'float', 'Businesses per km²', '[100, 1200]', 'count/km²'),
            ('avg_income_district', 'float', 'Average district income', '[8, 20]', 'M VND/month'),
            ('has_collateral', 'int', 'Has collateral (0/1)', '[0, 1]', 'binary'),
            ('collateral_value', 'float', 'Collateral value', 'varies', 'VND'),
            ('loan_to_value', 'float', 'Loan / Collateral Value', '[0, 1]', 'ratio'),
            ('collateral_liquidity_score', 'float', 'Collateral marketability', '[1, 10]', 'score'),
            ('collateral_type', 'str', 'Type of collateral', 'categorical', 'Real Estate/Machinery/Inventory'),
        ],
    }
    
    # Create markdown table
    lines = []
    lines.append("# Data Dictionary\n\n")
    lines.append("**Total Features**: 67\n\n")
    lines.append("---\n\n")
    
    for module, feature_list in features.items():
        lines.append(f"## {module}\n\n")
        lines.append("| Feature | Type | Description | Range | Unit |\n")
        lines.append("|---------|------|-------------|-------|------|\n")
        
        for feat_name, feat_type, description, range_val, unit in feature_list:
            lines.append(f"| `{feat_name}` | {feat_type} | {description} | {range_val} | {unit} |\n")
        
        lines.append("\n")
    
    return lines

def generate_statistical_summary(df):
    """Generate detailed statistical summary"""
    
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)
    
    # Numeric features only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ['sample_id', 'default']]
    
    stats = []
    stats.append("\n# Statistical Summary\n\n")
    stats.append(f"**Dataset**: {len(df):,} samples\n\n")
    
    # Overall statistics
    stats.append("## Overall Statistics\n\n")
    stats.append("| Metric | Value |\n")
    stats.append("|--------|-------|\n")
    stats.append(f"| Total Samples | {len(df):,} |\n")
    stats.append(f"| Numeric Features | {len(feature_cols)} |\n")
    stats.append(f"| Missing Values | {df[feature_cols].isnull().sum().sum()} |\n")
    stats.append(f"| Default Rate | {df['default'].mean()*100:.2f}% |\n\n")
    
    # Industry breakdown
    stats.append("## Industry Distribution\n\n")
    stats.append("| Industry Code | Industry Name | Count | Percentage | Default Rate |\n")
    stats.append("|---------------|---------------|-------|------------|-------------|\n")
    
    industry_names = {10: 'Food Manufacturing', 46: 'Wholesale', 56: 'F&B'}
    
    for ind in sorted(df['industry_code'].unique()):
        ind_df = df[df['industry_code'] == ind]
        count = len(ind_df)
        pct = count / len(df) * 100
        def_rate = ind_df['default'].mean() * 100
        name = industry_names.get(ind, f'Industry {ind}')
        stats.append(f"| {ind} | {name} | {count:,} | {pct:.1f}% | {def_rate:.2f}% |\n")
    
    stats.append("\n")
    
    # Feature statistics table
    stats.append("## Feature Statistics\n\n")
    stats.append("| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |\n")
    stats.append("|---------|------|-----|-----|-----|-----|-----|-----|\n")
    
    for col in feature_cols[:30]:  # Top 30 features
        if col in df.columns:
            desc = df[col].describe()
            stats.append(f"| `{col}` | {desc['mean']:.2f} | {desc['std']:.2f} | {desc['min']:.2f} | {desc['25%']:.2f} | {desc['50%']:.2f} | {desc['75%']:.2f} | {desc['max']:.2f} |\n")
    
    stats.append("\n...(showing first 30 features)\n\n")
    
    print("✅ Statistical summary generated")
    return stats

def generate_module_analysis(df):
    """Analyze each module separately"""
    
    module_features = {
        'Module 1: Financial Health': [
            'roa', 'roe', 'profit_margin', 'revenue_growth', 'current_ratio', 'quick_ratio',
            'debt_to_equity', 'debt_to_asset', 'dscr', 'inventory_turnover', 'receivable_turnover',
            'asset_turnover', 'free_cash_flow', 'operating_cash_flow_ratio', 'cash_conversion_cycle',
            'days_sales_outstanding', 'days_payables_outstanding', 'avg_daily_balance',
            'min_balance_3m', 'cash_flow_volatility', 'net_cash_flow', 'overdraft_count'
        ],
        'Module 2: Credit Behavior': [
            'cic_score', 'num_active_loans', 'total_outstanding_debt', 'max_past_due_days',
            'num_past_due_30d', 'num_past_due_90d', 'debt_burden_ratio', 'credit_history_length',
            'previous_default_history', 'num_transactions_3m', 'avg_monthly_deposits',
            'avg_monthly_withdrawals', 'transaction_regularity_score'
        ],
        'Module 3: Business Quality': [
            'owner_age', 'owner_education', 'owner_experience', 'product_differentiation_score',
            'industry_competition_intensity', 'supplier_relationships', 'customer_concentration',
            'digital_footprint', 'business_age', 'num_employees', 'employee_productivity_score',
            'revenue_capacity_ratio', 'financial_data_reliability_score', 'business_certification_count'
        ],
        'Module 4: Stability & Risk': [
            'founding_year', 'years_at_current_location', 'industry_changes_count',
            'past_bankruptcy', 'ownership_stability', 'industry_code', 'industry_risk_score',
            'industry_lifecycle_stage', 'district_code', 'district_risk_score', 'business_zone',
            'district_business_density', 'avg_income_district', 'has_collateral', 'collateral_value',
            'loan_to_value', 'collateral_liquidity_score', 'collateral_type'
        ]
    }
    
    lines = []
    lines.append("\n# Module-wise Analysis\n\n")
    
    for module_name, features in module_features.items():
        lines.append(f"## {module_name}\n\n")
        
        # Get features that exist in df
        existing_features = [f for f in features if f in df.columns]
        
        lines.append(f"**Features**: {len(existing_features)}\n\n")
        
        # Numeric features only
        numeric_features = [f for f in existing_features if df[f].dtype in [np.int64, np.float64]]
        
        if numeric_features:
            lines.append("### Key Statistics\n\n")
            lines.append("| Feature | Mean | Std | Range |\n")
            lines.append("|---------|------|-----|-------|\n")
            
            for feat in numeric_features[:10]:  # Top 10
                mean_val = df[feat].mean()
                std_val = df[feat].std()
                min_val = df[feat].min()
                max_val = df[feat].max()
                lines.append(f"| `{feat}` | {mean_val:.2f} | {std_val:.2f} | [{min_val:.2f}, {max_val:.2f}] |\n")
            
            if len(numeric_features) > 10:
                lines.append(f"\n...(showing top 10 of {len(numeric_features)} numeric features)\n")
        
        lines.append("\n")
    
    print("✅ Module analysis generated")
    return lines

def main():
    print("\n" + "=" * 80)
    print("DATASET DOCUMENTATION GENERATOR")
    print("=" * 80 + "\n")
    
    # Load dataset
    base_dir = Path("d:/SystemFolders/Desktop/NCKH/data_organized/03_final")
    dataset_path = base_dir / "dataset_67_features_balanced.csv"
    
    print(f"📂 Loading: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"   Shape: {df.shape}")
    
    # Generate components
    print("\nGenerating documentation components...")
    
    dict_lines = create_data_dictionary()
    stats_lines = generate_statistical_summary(df)
    module_lines = generate_module_analysis(df)
    
    # Combine all
    all_lines = []
    all_lines.append("# SME Credit Scoring Dataset Documentation\n\n")
    all_lines.append(f"**Version**: 1.0\n")
    all_lines.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
    all_lines.append(f"**Total Samples**: {len(df):,}\n")
    all_lines.append(f"**Total Features**: 67\n\n")
    all_lines.append("---\n\n")
    
    # Table of contents
    all_lines.append("## Table of Contents\n\n")
    all_lines.append("1. [Data Dictionary](#data-dictionary)\n")
    all_lines.append("2. [Statistical Summary](#statistical-summary)\n")
    all_lines.append("3. [Module-wise Analysis](#module-wise-analysis)\n\n")
    all_lines.append("---\n\n")
    
    # Append all sections
    all_lines.extend(dict_lines)
    all_lines.append("---\n\n")
    all_lines.extend(stats_lines)
    all_lines.append("---\n\n")
    all_lines.extend(module_lines)
    
    # Save
    output_path = base_dir / "DATASET_DOCUMENTATION.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(all_lines)
    
    print(f"\n💾 Documentation saved: {output_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ DOCUMENTATION COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated: {output_path}")
    print(f"Size: {len(all_lines):,} lines")

if __name__ == "__main__":
    main()
