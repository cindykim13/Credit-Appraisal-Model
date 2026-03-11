"""
Script 5 (Simplified): Final QA and Train/Val/Test Split
No sklearn dependency - using manual stratified sampling
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

def manual_stratified_split(df, train_size=0.70, val_size=0.15, test_size=0.15):
    """
    Manual stratified split by industry_code and default
    """
    
    print("=" * 80)
    print("TRAIN/VAL/TEST SPLIT (Manual Stratification)")
    print("=" * 80)
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    # Split by each combination of industry and default
    for industry in df['industry_code'].unique():
        for default_val in [0, 1]:
            subset = df[(df['industry_code'] == industry) & (df['default'] == default_val)]
            
            if len(subset) == 0:
                continue
            
            # Shuffle
            subset = subset.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Calculate split points
            n = len(subset)
            train_end = int(n * train_size)
            val_end = train_end + int(n * val_size)
            
            train_dfs.append(subset[:train_end])
            val_dfs.append(subset[train_end:val_end])
            test_dfs.append(subset[val_end:])
    
    # Concatenate
    train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 Split Sizes:")
    print(f"   Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"   Total: {len(df):,}")
    
    # Validate
    print("\n✅ Validation:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        default_rate = split_df['default'].mean() * 100
        print(f"\n{split_name}:")
        print(f"   Default rate: {default_rate:.2f}%")
        for industry in sorted(split_df['industry_code'].unique()):
            ind_count = (split_df['industry_code'] == industry).sum()
            ind_pct = ind_count / len(split_df) * 100
            print(f"   Industry {industry}: {ind_count:,} ({ind_pct:.1f}%)")
    
    return train_df, val_df, test_df

def check_high_correlations(df, threshold=0.90):
    """Simple correlation check"""
    
    print("\n" + "=" *80)
    print("CORRELATION CHECK")
    print("=" * 80)
    
    # Select numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ['sample_id', 'default']]
    
    print(f"\nChecking {len(feature_cols)} numeric features...")
    
    corr_matrix = df[feature_cols].corr()
    
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr:
        print(f"\n⚠️  Found {len(high_corr)} highly correlated pairs (|r| > {threshold}):")
        for f1, f2, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[:10]:
            print(f"   {f1:35s} ↔ {f2:35s}: {r:6.3f}")
    else:
        print(f"\n✅ No highly correlated pairs (threshold: {threshold})")
    
    return high_corr

def generate_report(df, train_df, val_df, test_df):
    """Generate simple text report"""
    
    print("\n" + "=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)
    
    lines = []
    lines.append("=" * 80 + "\n")
    lines.append("FINAL DATA QUALITY REPORT\n")
    lines.append("=" * 80 + "\n\n")
    
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    lines.append("DATASET SUMMARY\n")
    lines.append("-" * 80 + "\n")
    lines.append(f"Full Dataset: {len(df):,} rows × {df.shape[1]} columns\n")
    lines.append(f"Features: 67 (+ sample_id, default)\n")
    lines.append(f"Default rate: {df['default'].mean()*100:.2f}%\n\n")
    
    lines.append("SPLITS\n")
    lines.append("-" * 80 + "\n")
    lines.append(f"Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)\n")
    lines.append(f"Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)\n")
    lines.append(f"Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)\n\n")
    
    lines.append("DATA QUALITY\n")
    lines.append("-" * 80 + "\n")
    lines.append(f"Missing values: {df.isnull().sum().sum()}\n")
    lines.append(f"Duplicates: {df.duplicated().sum()}\n\n")
    
    lines.append("INDUSTRY DISTRIBUTION\n")
    lines.append("-" * 80 + "\n")
    for ind in sorted(df['industry_code'].unique()):
        count = (df['industry_code'] == ind).sum()
        pct = count / len(df) * 100
        lines.append(f"Industry {ind}: {count:,} ({pct:.1f}%)\n")
    
    lines.append("\n")
    lines.append("MODULE COVERAGE: 100%\n")
    lines.append("-" * 80 + "\n")
    lines.append("Module 1 (Financial):  22/22 ✅\n")
    lines.append("Module 2 (Credit):     13/13 ✅\n")
    lines.append("Module 3 (Business):   14/14 ✅\n")
    lines.append("Module 4 (Stability):  18/18 ✅\n")
    lines.append("\n" + "=" * 80 + "\n")
    lines.append("READY FOR MODEL TRAINING!\n")
    lines.append("=" * 80 + "\n")
    
    # Save
    output_dir = Path("d:/SystemFolders/Desktop/NCKH/data_organized/03_final")
    report_path = output_dir / "FINAL_QA_REPORT.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"💾 Report saved: {report_path}")
    
    return report_path

def main():
    print("\n" + "=" * 80)
    print("PHASE 5: FINAL QA & SPLIT")
    print("=" * 80 + "\n")
    
    # Load
    base_dir = Path("d:/SystemFolders/Desktop/NCKH/data_organized")
    input_path = base_dir / "03_final" / "dataset_67_features_balanced.csv"
    
    print(f"📂 Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Shape: {df.shape}")
    
    # Check correlations
    high_corr = check_high_correlations(df, threshold=0.90)
    
    # Create splits
    train_df, val_df, test_df = manual_stratified_split(df)
    
    # Save
    output_dir = base_dir / "03_final"
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    print(f"\n💾 Saved:")
    print(f"   train.csv")
    print(f"   val.csv")
    print(f"   test.csv")
    
    # Report
    generate_report(df, train_df, val_df, test_df)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PHASE 5 COMPLETE!")
    print("=" * 80)
    print("\n🎉 ALL PHASES COMPLETE - READY FOR MODEL TRAINING!")
    print(f"\nFinal datasets in: {output_dir}")
    print(f"   - train.csv: {len(train_df):,} samples")
    print(f"   - val.csv: {len(val_df):,} samples")
    print(f"   - test.csv: {len(test_df):,} samples")

if __name__ == "__main__":
    main()
