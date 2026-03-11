"""
Script 06: Tạo Dữ Liệu Synthetic
Mục tiêu: Sinh 3,000 mẫu (1,000 mỗi ngành) dựa trên phân phối đã phân tích
"""

import pandas as pd
import numpy as np
import json

def load_industry_params(industry_name):
    """Load tham số phân phối từ file JSON"""
    filename = f"data/processed/distributions/{industry_name}_params.json"
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_industry_samples(industry_code, industry_name, n_samples=1000):
    """
    Sinh mẫu synthetic cho một ngành sử dụng multivariate normal
    """
    print(f"\n🏭 Đang tạo {n_samples} mẫu cho {industry_name} (Mã {industry_code})...")
    
    # Load parameters
    params = load_industry_params(industry_name)
    stats = params['statistics']
    corr_matrix_dict = params['correlation_matrix']
    
    # Ratio columns
    ratio_cols = [
        'roa', 'roe', 'profit_margin', 'revenue_growth',
        'current_ratio', 'quick_ratio', 'debt_to_equity', 'debt_to_asset',
        'asset_turnover', 'inventory_turnover', 'receivable_turnover', 'dscr'
    ]
    
    # Extract means and stds
    means = np.array([stats[col]['mean'] for col in ratio_cols])
    stds = np.array([stats[col]['std'] for col in ratio_cols])
    
    # Convert correlation dict to matrix
    corr_matrix = pd.DataFrame(corr_matrix_dict)[ratio_cols].loc[ratio_cols].values
    
    # Build covariance matrix
    cov_matrix = np.outer(stds, stds) * corr_matrix
    
    # Generate samples
    samples = np.random.multivariate_normal(means, cov_matrix, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(samples, columns=ratio_cols)
    df['industry_code'] = industry_code
    df['sample_id'] = range(1, n_samples + 1)
    
    # Reorder columns
    cols = ['sample_id', 'industry_code'] + ratio_cols
    df = df[cols]
    
    print(f"  ✓ Đã tạo {len(df)} mẫu")
    print(f"  ✓ ROA trung bình: {df['roa'].mean()*100:.2f}%")
    print(f"  ✓ Current Ratio trung bình: {df['current_ratio'].mean():.2f}")
    
    return df

def validate_synthetic_data(df_synth, df_real):
    """
    Kiểm tra chất lượng dữ liệu synthetic
    """
    print(f"\n{'='*60}")
    print("KIỂM TRA CHẤT LƯỢNG DỮ LIỆU SYNTHETIC")
    print(f"{'='*60}")
    
    ratio_cols = [
        'roa', 'roe', 'profit_margin', 'revenue_growth',
        'current_ratio', 'quick_ratio', 'debt_to_equity', 'debt_to_asset',
        'asset_turnover', 'inventory_turnover', 'receivable_turnover', 'dscr'
    ]
    
    industries = {46: 'Bán buôn', 10: 'Sản xuất TP', 56: 'F&B'}
    
    print("\n📊 SO SÁNH TRUNG BÌNH (Real vs Synthetic):")
    print(f"{'Ngành':<15} {'Chỉ số':<20} {'Real':>12} {'Synthetic':>12} {'Diff%':>10}")
    print("-" * 70)
    
    max_diff = 0
    for industry_code, industry_name in industries.items():
        real_ind = df_real[df_real['industry_code'] == industry_code]
        synth_ind = df_synth[df_synth['industry_code'] == industry_code]
        
        for ratio in ['roa', 'current_ratio', 'debt_to_equity']:  # Sample ratios
            real_mean = real_ind[ratio].mean()
            synth_mean = synth_ind[ratio].mean()
            diff_pct = abs(synth_mean - real_mean) / abs(real_mean) * 100
            max_diff = max(max_diff, diff_pct)
            
            print(f"{industry_name:<15} {ratio:<20} {real_mean:>12.4f} "
                  f"{synth_mean:>12.4f} {diff_pct:>9.1f}%")
    
    print(f"\n✅ Sai số lớn nhất: {max_diff:.1f}%")
    
    if max_diff < 15:
        print("✅ PASS: Tất cả chỉ số trong phạm vi ±15%")
        return True
    else:
        print("⚠️  WARNING: Một số chỉ số sai lệch > 15%")
        return False

def create_final_dataset():
    """
    Tạo dataset synthetic cuối cùng
    """
    print(f"\n{'='*60}")
    print("PHASE 4: TẠO DỮ LIỆU SYNTHETIC")
    print(f"{'='*60}")
    
    # Generate for each industry
    industries = {
        46: 'Wholesale',
        10: 'Food_Manufacturing',
        56: 'FnB'
    }
    
    dfs = []
    for industry_code, industry_name in industries.items():
        df = generate_industry_samples(industry_code, industry_name, n_samples=1000)
        dfs.append(df)
    
    # Concatenate
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Reset sample_id to be continuous
    final_df['sample_id'] = range(1, len(final_df) + 1)
    
    print(f"\n{'='*60}")
    print("TỔNG KẾT DATASET CUỐI CÙNG")
    print(f"{'='*60}")
    print(f"Tổng số mẫu: {len(final_df)}")
    print(f"Phân bố theo ngành:")
    for industry_code, industry_name in industries.items():
        count = len(final_df[final_df['industry_code'] == industry_code])
        print(f"  • {industry_name} (Mã {industry_code}): {count} mẫu")
    
    print(f"\nSố cột: {len(final_df.columns)}")
    print(f"Cột: {', '.join(final_df.columns[:5])}...")
    print(f"\nMissing values: {final_df.isnull().sum().sum()}")
    
    # Save
    output_file = 'data/output/synthetic_financial_3k.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\n✅ Đã lưu: {output_file}")
    
    return final_df

def main():
    # Create synthetic data
    df_synth = create_final_dataset()
    
    # Load real data for validation
    df_real = pd.read_csv('data/processed/financial_features.csv')
    
    # Validate
    is_valid = validate_synthetic_data(df_synth, df_real)
    
    print(f"\n{'='*60}")
    print("✅ PHASE 4 HOÀN THÀNH!")
    print(f"{'='*60}")
    print(f"\n🎉 Đã tạo thành công dataset synthetic:")
    print(f"   • File: data/output/synthetic_financial_3k.csv")
    print(f"   • Tổng mẫu: 3,000")
    print(f"   • Ngành: 3 (mỗi ngành 1,000 mẫu)")
    print(f"   • Đặc trưng: 12 tỷ số tài chính")
    print(f"   • Chất lượng: {'✅ Tốt' if is_valid else '⚠️  Cần xem xét'}")

if __name__ == "__main__":
    main()
