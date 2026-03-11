"""
Script: Generate Advanced Statistical Analysis
- Distribution analysis
- Correlation matrices
- Default vs Non-default comparisons
"""
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_distributions(df):
    """Analyze feature distributions"""
    
    lines = []
    lines.append("# Distribution Analysis\n\n")
    
    # Numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ['sample_id', 'default']]
    
    lines.append("## Skewness Analysis\n\n")
    lines.append("Features with high skewness (|skew| > 1.0):\n\n")
    lines.append("| Feature | Skewness | Interpretation |\n")
    lines.append("|---------|----------|----------------|\n")
    
    for col in feature_cols:
        if col in df.columns:
            skew = df[col].skew()
            if abs(skew) > 1.0:
                interp = "Right-skewed" if skew > 0 else "Left-skewed"
                lines.append(f"| `{col}` | {skew:.2f} | {interp} |\n")
    
    lines.append("\n")
    
    return lines

def compare_default_nondefault(df):
    """Compare features between default and non-default"""
    
    lines = []
    lines.append("\n# Default vs Non-Default Comparison\n\n")
    
    default_df = df[df['default'] == 1]
    nondefault_df = df[df['default'] == 0]
    
    lines.append(f"**Default cases**: {len(default_df):,} ({len(default_df)/len(df)*100:.2f}%)\n")
    lines.append(f"**Non-default cases**: {len(nondefault_df):,} ({len(nondefault_df)/len(df)*100:.2f}%)\n\n")
    
    # Key differentiators
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ['sample_id', 'default']]
    
    differences = []
    for col in feature_cols[:30]:  # Top 30
        if col in df.columns:
            mean_default = default_df[col].mean()
            mean_nondefault = nondefault_df[col].mean()
            
            if mean_nondefault != 0:
                pct_diff = abs((mean_default - mean_nondefault) / mean_nondefault * 100)
                differences.append((col, mean_default, mean_nondefault, pct_diff))
    
    # Sort by percentage difference
    differences.sort(key=lambda x: x[3], reverse=True)
    
    lines.append("## Top Differentiating Features\n\n")
    lines.append("Features with largest difference between default and non-default:\n\n")
    lines.append("| Feature | Default Mean | Non-Default Mean | % Difference |\n")
    lines.append("|---------|--------------|------------------|-------------|\n")
    
    for feat, def_mean, nondef_mean, pct_diff in differences[:20]:
        lines.append(f"| `{feat}` | {def_mean:.2f} | {nondef_mean:.2f} | {pct_diff:.1f}% |\n")
    
    lines.append("\n")
    
    return lines

def correlation_analysis(df):
    """Analyze correlations"""
    
    lines = []
    lines.append("\n# Correlation Analysis\n\n")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ['sample_id', 'default']]
    
    corr_matrix = df[feature_cols].corr()
    
    # Find highly correlated pairs
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.70:
                high_corr.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
    
    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    
    lines.append("## Highly Correlated Feature Pairs (|r| > 0.70)\n\n")
    
    if high_corr:
        lines.append("| Feature 1 | Feature 2 | Correlation |\n")
        lines.append("|-----------|-----------|-------------|\n")
        
        for f1, f2, corr in high_corr[:30]:
            lines.append(f"| `{f1}` | `{f2}` | {corr:.3f} |\n")
    else:
        lines.append("✅ No highly correlated pairs found!\n")
    
    lines.append("\n")
    
    # Correlation with target
    lines.append("## Correlation with Default Target\n\n")
    lines.append("Features most correlated with default status:\n\n")
    lines.append("| Feature | Correlation | Strength |\n")
    lines.append("|---------|-------------|----------|\n")
    
    target_corrs = []
    for col in feature_cols:
        if col in df.columns:
            corr = df[[col, 'default']].corr().iloc[0, 1]
            target_corrs.append((col, corr))
    
    target_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for feat, corr in target_corrs[:20]:
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        lines.append(f"| `{feat}` | {corr:.3f} | {strength} |\n")
    
    lines.append("\n")
    
    return lines

def categorical_analysis(df):
    """Analyze categorical features"""
    
    lines = []
    lines.append("\n# Categorical Features Analysis\n\n")
    
    # Industry
    lines.append("## Industry Code Distribution\n\n")
    lines.append("| Industry | Name | Count | % | Default Rate |\n")
    lines.append("|----------|------|-------|---|-------------|\n")
    
    industry_names = {10: 'Food Manufacturing', 46: 'Wholesale', 56: 'F&B'}
    
    for ind in sorted(df['industry_code'].unique()):
        ind_df = df[df['industry_code'] == ind]
        count = len(ind_df)
        pct = count / len(df) * 100
        def_rate = ind_df['default'].mean() * 100
        name = industry_names.get(ind, f'Industry {ind}')
        lines.append(f"| {ind} | {name} | {count:,} | {pct:.1f}% | {def_rate:.2f}% |\n")
    
    lines.append("\n")
    
    # Owner education
    if 'owner_education' in df.columns:
        lines.append("## Owner Education Distribution\n\n")
        lines.append("| Education Level | Count | % | Default Rate |\n")
        lines.append("|----------------|-------|---|-------------|\n")
        
        for edu in df['owner_education'].value_counts().index:
            edu_df = df[df['owner_education'] == edu]
            count = len(edu_df)
            pct = count / len(df) * 100
            def_rate = edu_df['default'].mean() * 100
            lines.append(f"| {edu} | {count:,} | {pct:.1f}% | {def_rate:.2f}% |\n")
        
        lines.append("\n")
    
    # Business zone
    if 'business_zone' in df.columns:
        lines.append("## Business Zone Distribution\n\n")
        lines.append("| Zone | Count | % | Default Rate |\n")
        lines.append("|------|-------|---|-------------|\n")
        
        for zone in df['business_zone'].value_counts().index:
            zone_df = df[df['business_zone'] == zone]
            count = len(zone_df)
            pct = count / len(df) * 100
            def_rate = zone_df['default'].mean() * 100
            lines.append(f"| {zone} | {count:,} | {pct:.1f}% | {def_rate:.2f}% |\n")
        
        lines.append("\n")
    
    # Collateral type
    if 'collateral_type' in df.columns:
        lines.append("## Collateral Type Distribution\n\n")
        lines.append("| Type | Count | % | Default Rate |\n")
        lines.append("|------|-------|---|-------------|\n")
        
        for ctype in df['collateral_type'].value_counts().index:
            ct_df = df[df['collateral_type'] == ctype]
            count = len(ct_df)
            pct = count / len(df) * 100
            def_rate = ct_df['default'].mean() * 100
            lines.append(f"| {ctype} | {count:,} | {pct:.1f}% | {def_rate:.2f}% |\n")
        
        lines.append("\n")
    
    return lines

def main():
    print("\n" + "=" * 80)
    print("ADVANCED STATISTICAL ANALYSIS")
    print("=" * 80 + "\n")
    
    # Load
    base_dir = Path("d:/SystemFolders/Desktop/NCKH/data_organized/03_final")
    df = pd.read_csv(base_dir / "dataset_67_features_balanced.csv")
    
    print(f"📂 Dataset: {df.shape}")
    
    # Generate analyses
    print("\nGenerating analyses...")
    
    dist_lines = analyze_distributions(df)
    comp_lines = compare_default_nondefault(df)
    corr_lines = correlation_analysis(df)
    cat_lines = categorical_analysis(df)
    
    # Combine
    all_lines = []
    all_lines.append("# Advanced Statistical Analysis\n\n")
    all_lines.append(f"**Dataset**: {len(df):,} samples × {df.shape[1]} features\n")
    all_lines.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    all_lines.append("---\n\n")
    
    all_lines.extend(dist_lines)
    all_lines.append("---\n\n")
    all_lines.extend(comp_lines)
    all_lines.append("---\n\n")
    all_lines.extend(corr_lines)
    all_lines.append("---\n\n")
    all_lines.extend(cat_lines)
    
    # Save
    output_path = base_dir / "STATISTICAL_ANALYSIS.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(all_lines)
    
    print(f"\n💾 Saved: {output_path}")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
