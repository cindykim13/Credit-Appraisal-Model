"""
Script 05: Phân Tích Phân Phối
Mục tiêu: Phân tích phân phối thống kê của 12 tỷ số tài chính theo từng ngành
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

# Tạo thư mục outputs
os.makedirs('data/processed/distributions', exist_ok=True)

def analyze_industry_distribution(df, industry_code, industry_name):
    """
    Phân tích phân phối cho một ngành
    """
    print(f"\n{'='*60}")
    print(f"PHÂN TÍCH NGÀNH: {industry_name} (Mã {industry_code})")
    print(f"{'='*60}")
    
    # Filter data
    industry_data = df[df['industry_code'] == industry_code].copy()
    print(f"Số bản ghi: {len(industry_data)}")
    
    # Ratios to analyze
    ratio_cols = [
        'roa', 'roe', 'profit_margin', 'revenue_growth',
        'current_ratio', 'quick_ratio', 'debt_to_equity', 'debt_to_asset',
        'asset_turnover', 'inventory_turnover', 'receivable_turnover', 'dscr'
    ]
    
    # 1. Descriptive statistics
    stats_dict = {}
    dist_params = {}
    
    print(f"\n📊 THỐNG KÊ MÔ TẢ:")
    print(f"{'Chỉ số':<20} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)
    
    for ratio in ratio_cols:
        data = industry_data[ratio].dropna()
        
        stats_dict[ratio] = {
            'mean': float(data.mean()),
            'median': float(data.median()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'q25': float(data.quantile(0.25)),
            'q75': float(data.quantile(0.75)),
            'count': int(len(data))
        }
        
        # Try to fit normal distribution
        try:
            mu, sigma = stats.norm.fit(data)
            dist_params[ratio] = {
                'distribution': 'normal',
                'mu': float(mu),
                'sigma': float(sigma)
            }
        except:
            dist_params[ratio] = {
                'distribution': 'empirical',
                'mu': float(data.mean()),
                'sigma': float(data.std())
            }
        
        print(f"{ratio:<20} {stats_dict[ratio]['mean']:>10.4f} "
              f"{stats_dict[ratio]['median']:>10.4f} "
              f"{stats_dict[ratio]['std']:>10.4f} "
              f"{stats_dict[ratio]['min']:>10.4f} "
              f"{stats_dict[ratio]['max']:>10.4f}")
    
    # 2. Correlation matrix
    print(f"\n📐 MA TRẬN TƯƠNG QUAN:")
    corr_matrix = industry_data[ratio_cols].corr()
    
    # Hiển thị top 5 correlations
    corr_pairs = []
    for i in range(len(ratio_cols)):
        for j in range(i+1, len(ratio_cols)):
            corr_pairs.append((ratio_cols[i], ratio_cols[j], corr_matrix.iloc[i, j]))
    
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"{'Chỉ số 1':<20} {'Chỉ số 2':<20} {'Correlation':>15}")
    print("-" * 55)
    for r1, r2, corr_val in corr_pairs[:5]:
        print(f"{r1:<20} {r2:<20} {corr_val:>15.3f}")
    
    # 3. Save parameters
    output = {
        'industry_code': int(industry_code),
        'industry_name': industry_name,
        'n_samples': int(len(industry_data)),
        'statistics': stats_dict,
        'distributions': dist_params,
        'correlation_matrix': corr_matrix.to_dict()
    }
    
    filename = f"data/processed/distributions/{industry_name.lower()}_params.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Đã lưu tham số: {filename}")
    
    return stats_dict, corr_matrix, dist_params

def create_visualization(df):
    """
    Tạo visualization tổng hợp
    """
    print(f"\n{'='*60}")
    print("TẠO BIỂU ĐỒ TRỰC QUAN")
    print(f"{'='*60}")
    
    ratio_cols = [
        'roa', 'roe', 'profit_margin', 'revenue_growth',
        'current_ratio', 'quick_ratio', 'debt_to_equity', 'debt_to_asset',
        'asset_turnover', 'inventory_turnover', 'receivable_turnover', 'dscr'
    ]
    
    industries = {
        46: ('Bán buôn', 'blue'),
        10: ('Sản xuất TP', 'green'),
        56: ('F&B', 'red')
    }
    
    # Create grid 4x3
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Phân Phối 12 Tỷ Số Tài Chính Theo Ngành', fontsize=16, fontweight='bold')
    
    for idx, ratio in enumerate(ratio_cols):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        for industry_code, (industry_name, color) in industries.items():
            data = df[df['industry_code'] == industry_code][ratio].dropna()
            ax.hist(data, bins=20, alpha=0.5, label=industry_name, color=color, edgecolor='black')
        
        ax.set_title(ratio.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlabel('Giá trị')
        ax.set_ylabel('Tần suất')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/distributions/all_distributions.png', dpi=150, bbox_inches='tight')
    print("✓ Đã lưu: data/processed/distributions/all_distributions.png")
    
    # Correlation heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Ma Trận Tương Quan Theo Ngành', fontsize=16, fontweight='bold')
    
    for idx, (industry_code, (industry_name, _)) in enumerate(industries.items()):
        industry_data = df[df['industry_code'] == industry_code]
        corr = industry_data[ratio_cols].corr()
        
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                   vmin=-1, vmax=1, ax=axes[idx],
                   cbar_kws={'label': 'Correlation'})
        axes[idx].set_title(f'{industry_name} (Mã {industry_code})', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data/processed/distributions/correlation_heatmaps.png', dpi=150, bbox_inches='tight')
    print("✓ Đã lưu: data/processed/distributions/correlation_heatmaps.png")

def main():
    print("="*60)
    print("PHASE 3: PHÂN TÍCH PHÂN PHỐI THỐNG KÊ")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/processed/financial_features.csv')
    print(f"\n✓ Đã load {len(df)} bản ghi")
    
    # Analyze each industry
    industries = {
        46: 'Wholesale',
        10: 'Food_Manufacturing',
        56: 'FnB'
    }
    
    all_stats = {}
    for industry_code, industry_name in industries.items():
        stats, corr, dist = analyze_industry_distribution(df, industry_code, industry_name)
        all_stats[industry_name] = {
            'stats': stats,
            'correlation': corr,
            'distributions': dist
        }
    
    # Create visualizations
    create_visualization(df)
    
    # Create summary report
    print(f"\n{'='*60}")
    print("TẠO BÁO CÁO TỔNG KẾT")
    print(f"{'='*60}")
    
    summary = {
        'total_samples': int(len(df)),
        'industries_analyzed': 3,
        'ratios_analyzed': 12,
        'files_generated': [
            'distributions/Wholesale_params.json',
            'distributions/Food_Manufacturing_params.json',
            'distributions/FnB_params.json',
            'distributions/all_distributions.png',
            'distributions/correlation_heatmaps.png'
        ]
    }
    
    with open('data/processed/distributions/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✓ Đã lưu: data/processed/distributions/summary.json")
    
    print(f"\n{'='*60}")
    print("✅ PHASE 3 HOÀN THÀNH!")
    print(f"{'='*60}")
    print(f"\nĐã phân tích:")
    print(f"  • {summary['total_samples']} bản ghi")
    print(f"  • {summary['industries_analyzed']} ngành")
    print(f"  • {summary['ratios_analyzed']} tỷ số tài chính")
    print(f"\nFiles đã tạo: {len(summary['files_generated'])}")
    for f in summary['files_generated']:
        print(f"  ✓ {f}")
    print(f"\n🚀 SẴN SÀNG cho Phase 4: Tạo Dữ Liệu Synthetic!")

if __name__ == "__main__":
    main()
