"""
Script 4: Adjust Default Rate from 5.6% to 8.5%
Method: Random Oversampling of minority class
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

def oversample_default_cases(df, target_rate=0.085):
    """
    Oversample minority class (default=1) to reach target default rate
    
    Args:
        df: Input dataframe
        target_rate: Target default rate (0.085 = 8.5%)
    
    Returns:
        Balanced dataframe
    """
    
    print("=" * 80)
    print("PHASE 4: ADJUSTING DEFAULT RATE")
    print("=" * 80)
    
    # Current state
    current_total = len(df)
    current_defaults = df['default'].sum()
    current_rate = current_defaults / current_total
    
    print(f"\n📊 Current State:")
    print(f"   Total samples: {current_total:,}")
    print(f"   Default cases (1): {current_defaults} ({current_rate*100:.2f}%)")
    print(f"   Non-default (0): {current_total - current_defaults} ({(1-current_rate)*100:.2f}%)")
    
    # Calculate needed samples
    # Let x = additional default cases needed
    # (current_defaults + x) / (current_total + x) = target_rate
    # Solving for x:
    # current_defaults + x = target_rate * (current_total + x)
    # current_defaults + x = target_rate * current_total + target_rate * x
    # x - target_rate * x = target_rate * current_total - current_defaults
    # x(1 - target_rate) = target_rate * current_total - current_defaults
    # x = (target_rate * current_total - current_defaults) / (1 - target_rate)
    
    additional_defaults_needed = int(
        (target_rate * current_total - current_defaults) / (1 - target_rate)
    )
    
    print(f"\n🎯 Target State:")
    print(f"   Target default rate: {target_rate*100:.1f}%")
    print(f"   Additional defaults needed: {additional_defaults_needed}")
    print(f"   New total: {current_total + additional_defaults_needed:,}")
    
    # Oversample
    if additional_defaults_needed <= 0:
        print("\n✅ Already at or above target rate!")
        return df
    
    # Get all default cases
    default_cases = df[df['default'] == 1]
    
    print(f"\n🔄 Oversampling...")
    print(f"   Sampling {additional_defaults_needed} cases from {len(default_cases)} default cases")
    
    # Random sample with replacement
    oversampled_defaults = default_cases.sample(
        n=additional_defaults_needed,
        replace=True,
        random_state=42
    )
    
    # Combine
    df_balanced = pd.concat([df, oversampled_defaults], ignore_index=True)
    
    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Verify
    new_total = len(df_balanced)
    new_defaults = df_balanced['default'].sum()
    new_rate = new_defaults / new_total
    
    print(f"\n✅ Result:")
    print(f"   Total samples: {new_total:,}")
    print(f"   Default cases: {new_defaults} ({new_rate*100:.2f}%)")
    print(f"   Non-default: {new_total - new_defaults} ({(1-new_rate)*100:.2f}%)")
    print(f"   Achieved rate: {new_rate*100:.2f}% (target: {target_rate*100:.1f}%)")
    
    return df_balanced

def validate_balance(df):
    """Validate class balance by industry"""
    
    print("\n" + "=" * 80)
    print("VALIDATION: Class Balance by Industry")
    print("=" * 80)
    
    for industry in sorted(df['industry_code'].unique()):
        industry_df = df[df['industry_code'] == industry]
        total = len(industry_df)
        defaults = industry_df['default'].sum()
        rate = defaults / total * 100
        
        print(f"\nIndustry {industry}:")
        print(f"   Total: {total:,}")
        print(f"   Defaults: {defaults} ({rate:.2f}%)")
        print(f"   Non-defaults: {total - defaults}")

def main():
    print("\n" + "=" * 80)
    print("DEFAULT RATE ADJUSTMENT SCRIPT")
    print("=" * 80 + "\n")
    
    # Load data
    base_dir = Path("d:/SystemFolders/Desktop/NCKH/data_organized")
    input_path = base_dir / "03_final" / "dataset_67_features_3k.csv"
    
    print(f"📂 Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Shape: {df.shape}")
    
    # Oversample
    df_balanced = oversample_default_cases(df, target_rate=0.085)
    
    # Validate
    validate_balance(df_balanced)
    
    # Save
    output_path = base_dir / "03_final" / "dataset_67_features_balanced.csv"
    df_balanced.to_csv(output_path, index=False)
    print(f"\n💾 Saved: {output_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PHASE 4 COMPLETE!")
    print("=" * 80)
    print(f"\nBalanced dataset: {df_balanced.shape[0]:,} rows × {df_balanced.shape[1]} columns")
    print(f"Default rate: {df_balanced['default'].mean()*100:.2f}%")
    print("\nNext: Phase 5 (Final QA & Train/Val/Test Split)")

if __name__ == "__main__":
    main()
