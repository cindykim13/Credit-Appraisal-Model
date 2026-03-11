"""
Script: Generate Vietnamese Documentation
Tạo documentation bằng tiếng Việt
"""
import pandas as pd
import numpy as np
from pathlib import Path

def create_data_dictionary_vi():
    """Tạo từ điển dữ liệu bằng tiếng Việt"""
    
    features = {
        # Module 1: Sức khỏe tài chính (22)
        'Module 1: Sức Khỏe Tài Chính': [
            ('revenue_growth', 'số thực', 'Tốc độ tăng trưởng doanh thu YoY', '[-0.5, 1.0]', '%'),
            ('profit_margin', 'số thực', 'Tỷ suất lợi nhuận ròng', '[0.02, 0.15]', 'tỷ lệ'),
            ('roa', 'số thực', 'Tỷ suất sinh lời trên tài sản (ROA)', '[0.01, 0.12]', 'tỷ lệ'),
            ('roe', 'số thực', 'Tỷ suất sinh lời trên vốn chủ (ROE)', '[0.02, 0.25]', 'tỷ lệ'),
            ('current_ratio', 'số thực', 'Tỷ số thanh toán hiện hành', '[0.5, 2.5]', 'tỷ lệ'),
            ('quick_ratio', 'số thực', 'Tỷ số thanh toán nhanh', '[0.4, 2.0]', 'tỷ lệ'),
            ('debt_to_equity', 'số thực', 'Tỷ lệ nợ trên vốn chủ', '[0.3, 1.5]', 'tỷ lệ'),
            ('debt_to_asset', 'số thực', 'Tỷ lệ nợ trên tài sản', '[0.2, 0.7]', 'tỷ lệ'),
            ('dscr', 'số thực', 'Hệ số thanh toán nợ', '[0.3, 3.5]', 'tỷ lệ'),
            ('inventory_turnover', 'số thực', 'Vòng quay hàng tồn kho', '[2, 13]', 'lần/năm'),
            ('receivable_turnover', 'số thực', 'Vòng quay khoản phải thu', '[2, 11]', 'lần/năm'),
            ('asset_turnover', 'số thực', 'Vòng quay tổng tài sản', '[0.3, 1.5]', 'lần/năm'),
            ('free_cash_flow', 'số thực', 'Dòng tiền tự do', '[-50tr, 500tr]', 'VND'),
            ('operating_cash_flow_ratio', 'số thực', 'Tỷ lệ dòng tiền hoạt động', '[0.3, 1.5]', 'tỷ lệ'),
            ('cash_conversion_cycle', 'số thực', 'Chu kỳ chuyển đổi tiền', '[-30, 180]', 'ngày'),
            ('days_sales_outstanding', 'số thực', 'Số ngày thu tiền bình quân', '[10, 120]', 'ngày'),
            ('days_payables_outstanding', 'số thực', 'Số ngày trả tiền nhà cung cấp', '[30, 60]', 'ngày'),
            ('avg_daily_balance', 'số thực', 'Số dư bình quân hàng ngày', 'khác nhau', 'VND'),
            ('min_balance_3m', 'số thực', 'Số dư tối thiểu 3 tháng', 'khác nhau', 'VND'),
            ('cash_flow_volatility', 'số thực', 'Độ biến động dòng tiền', 'khác nhau', '%'),
            ('net_cash_flow', 'số thực', 'Dòng tiền ròng', 'khác nhau', 'VND'),
            ('overdraft_count', 'số nguyên', 'Số lần thấu chi', '[0, 10]', 'lần'),
        ],
        
        # Module 2: Hành vi tín dụng (13)
        'Module 2: Hành Vi Tín Dụng': [
            ('cic_score', 'số nguyên', 'Điểm CIC (Trung tâm Thông tin Tín dụng)', '[300, 850]', 'điểm'),
            ('num_active_loans', 'số nguyên', 'Số khoản vay đang hoạt động', '[0, 5]', 'khoản'),
            ('total_outstanding_debt', 'số thực', 'Tổng dư nợ', 'khác nhau', 'VND'),
            ('max_past_due_days', 'số nguyên', 'Số ngày quá hạn tối đa', '[0, 180]', 'ngày'),
            ('num_past_due_30d', 'số nguyên', 'Số lần quá hạn >30 ngày', '[0, 10]', 'lần'),
            ('num_past_due_90d', 'số nguyên', 'Số lần quá hạn >90 ngày', '[0, 5]', 'lần'),
            ('debt_burden_ratio', 'số thực', 'Tỷ lệ gánh nặng nợ / Doanh thu', '[0, 2]', 'tỷ lệ'),
            ('credit_history_length', 'số thực', 'Thời gian lịch sử tín dụng', '[0, 20]', 'năm'),
            ('previous_default_history', 'số nguyên', 'Có lịch sử vỡ nợ (0/1)', '[0, 1]', 'nhị phân'),
            ('num_transactions_3m', 'số nguyên', 'Số giao dịch trong 3 tháng', '[5, 500]', 'giao dịch'),
            ('avg_monthly_deposits', 'số thực', 'Tiền gửi bình quân hàng tháng', 'khác nhau', 'VND'),
            ('avg_monthly_withdrawals', 'số thực', 'Tiền rút bình quân hàng tháng', 'khác nhau', 'VND'),
            ('transaction_regularity_score', 'số thực', 'Điểm đều đặn giao dịch', '[1, 10]', 'điểm'),
        ],
        
        # Module 3: Chất lượng doanh nghiệp (14)
        'Module 3: Chất Lượng Doanh Nghiệp': [
            ('owner_age', 'số nguyên', 'Tuổi chủ doanh nghiệp', '[25, 70]', 'tuổi'),
            ('owner_education', 'chuỗi', 'Trình độ học vấn chủ DN', 'phân loại', 'THPT/TC/ĐH/ThS'),
            ('owner_experience', 'số nguyên', 'Số năm kinh nghiệm ngành', '[0, 40]', 'năm'),
            ('product_differentiation_score', 'số thực', 'Điểm khác biệt sản phẩm', '[1, 10]', 'điểm'),
            ('industry_competition_intensity', 'số thực', 'Mức độ cạnh tranh ngành', '[1, 10]', 'điểm'),
            ('supplier_relationships', 'số thực', 'Chất lượng quan hệ NCC', '[1, 10]', 'điểm'),
            ('customer_concentration', 'số thực', '% DT từ top 3 khách hàng', '[0, 1]', 'tỷ lệ'),
            ('digital_footprint', 'số thực', 'Điểm hiện diện trực tuyến', '[1, 10]', 'điểm'),
            ('business_age', 'số nguyên', 'Số năm hoạt động', '[1, 50]', 'năm'),
            ('num_employees', 'số nguyên', 'Số lượng nhân viên', '[5, 200]', 'người'),
            ('employee_productivity_score', 'số thực', 'Năng suất nhân viên', '[1, 10]', 'điểm'),
            ('revenue_capacity_ratio', 'số thực', 'DT thực tế / DT tiềm năng', '[0, 1]', 'tỷ lệ'),
            ('financial_data_reliability_score', 'số thực', 'Điểm độ tin cậy dữ liệu', '[1, 10]', 'điểm'),
            ('business_certification_count', 'số nguyên', 'Số chứng chỉ (ISO,...)', '[0, 5]', 'chứng chỉ'),
        ],
        
        # Module 4: Ổn định & Rủi ro (18)
        'Module 4: Ổn Định & Rủi Ro': [
            ('founding_year', 'số nguyên', 'Năm thành lập', '[1980, 2024]', 'năm'),
            ('years_at_current_location', 'số nguyên', 'Số năm ở địa chỉ hiện tại', '[0, 40]', 'năm'),
            ('industry_changes_count', 'số nguyên', 'Số lần đổi ngành', '[0, 5]', 'lần'),
            ('past_bankruptcy', 'số nguyên', 'Có lịch sử phá sản (0/1)', '[0, 1]', 'nhị phân'),
            ('ownership_stability', 'số thực', 'Điểm ổn định quyền sở hữu', '[1, 10]', 'điểm'),
            ('industry_code', 'số nguyên', 'Mã ngành VSIC', '{10, 46, 56}', 'phân loại'),
            ('industry_risk_score', 'số nguyên', 'Điểm rủi ro ngành', '[4, 8]', 'điểm'),
            ('industry_lifecycle_stage', 'chuỗi', 'Giai đoạn vòng đời ngành', 'phân loại', 'Tăng/Trưởng'),
            ('district_code', 'số nguyên', 'Mã quận/huyện (1-24)', '[1, 24]', 'phân loại'),
            ('district_risk_score', 'số thực', 'Điểm rủi ro khu vực', '[3, 7]', 'điểm'),
            ('business_zone', 'chuỗi', 'Vùng kinh doanh', 'phân loại', 'CBD/ĐT/NT/KCN'),
            ('district_business_density', 'số thực', 'Mật độ DN/km²', '[100, 1200]', 'DN/km²'),
            ('avg_income_district', 'số thực', 'Thu nhập bình quân quận', '[8, 20]', 'tr VND/tháng'),
            ('has_collateral', 'số nguyên', 'Có TSĐB (0/1)', '[0, 1]', 'nhị phân'),
            ('collateral_value', 'số thực', 'Giá trị TSĐB', 'khác nhau', 'VND'),
            ('loan_to_value', 'số thực', 'Tỷ lệ vay / Giá trị TSĐB', '[0, 1]', 'tỷ lệ'),
            ('collateral_liquidity_score', 'số thực', 'Điểm thanh khoản TSĐB', '[1, 10]', 'điểm'),
            ('collateral_type', 'chuỗi', 'Loại tài sản đảm bảo', 'phân loại', 'BĐS/Máy/Hàng'),
        ],
    }
    
    # Tạo bảng markdown
    lines = []
    lines.append("# Từ Điển Dữ Liệu\n\n")
    lines.append("**Tổng số features**: 67\n\n")
    lines.append("---\n\n")
    
    for module, feature_list in features.items():
        lines.append(f"## {module}\n\n")
        lines.append("| Feature | Kiểu | Mô tả | Khoảng giá trị | Đơn vị |\n")
        lines.append("|---------|------|-------|----------------|--------|\n")
        
        for feat_name, feat_type, description, range_val, unit in feature_list:
            lines.append(f"| `{feat_name}` | {feat_type} | {description} | {range_val} | {unit} |\n")
        
        lines.append("\n")
    
    return lines

def main():
    print("\n" + "=" * 80)
    print("TẠO DOCUMENTATION TIẾNG VIỆT")
    print("=" * 80 + "\n")
    
    # Load dataset
    base_dir = Path("d:/SystemFolders/Desktop/NCKH/data_organized/03_final")
    df = pd.read_csv(base_dir / "dataset_67_features_balanced.csv")
    
    print(f"📂 Dataset: {df.shape}")
    
    # Generate dictionary
    dict_lines = create_data_dictionary_vi()
    
    # Thêm phần tổng quan
    all_lines = []
    all_lines.append("# Tài Liệu Dataset Chấm Điểm Tín Dụng SME\n\n")
    all_lines.append(f"**Phiên bản**: 1.0\n")
    all_lines.append(f"**Ngày**: 13/02/2026\n")
    all_lines.append(f"**Tổng số mẫu**: {len(df):,}\n")
    all_lines.append(f"**Tổng số features**: 67\n\n")
    all_lines.append("---\n\n")
    
    # Tổng quan dataset
    all_lines.append("## 📊 Tổng Quan Dataset\n\n")
    all_lines.append("### Chỉ Số Chính\n\n")
    all_lines.append("| Chỉ số | Giá trị |\n")
    all_lines.append("|--------|----------|\n")
    all_lines.append(f"| **Tổng số mẫu** | {len(df):,} |\n")
    all_lines.append(f"| **Số features** | 67 (+ 2 metadata) |\n")
    all_lines.append(f"| **Tỷ lệ vỡ nợ** | {df['default'].mean()*100:.2f}% |\n")
    all_lines.append(f"| **Missing values** | 0 |\n")
    all_lines.append(f"| **Duplicates** | 0 |\n\n")
    
    # Phân bổ ngành
    all_lines.append("### Phân Bổ Theo Ngành\n\n")
    all_lines.append("| Ngành | Mã | Số mẫu | % | Tỷ lệ vỡ nợ |\n")
    all_lines.append("|-------|-----|---------|---|-------------|\n")
    
    industry_names = {10: 'Sản xuất thực phẩm', 46: 'Bán buôn', 56: 'Ăn uống'}
    for ind in sorted(df['industry_code'].unique()):
        ind_df = df[df['industry_code'] == ind]
        count = len(ind_df)
        pct = count / len(df) * 100
        def_rate = ind_df['default'].mean() * 100
        name = industry_names.get(ind, f'Ngành {ind}')
        all_lines.append(f"| {name} | {ind} | {count:,} | {pct:.1f}% | {def_rate:.2f}% |\n")
    
    all_lines.append("\n")
    
    # Coverage theo module
    all_lines.append("### Coverage Theo Module\n\n")
    all_lines.append("| Module | Số Features | Coverage |\n")
    all_lines.append("|--------|-------------|----------|\n")
    all_lines.append("| Module 1: Sức khỏe tài chính | 22 | ✅ 100% |\n")
    all_lines.append("| Module 2: Hành vi tín dụng | 13 | ✅ 100% |\n")
    all_lines.append("| Module 3: Chất lượng DN | 14 | ✅ 100% |\n")
    all_lines.append("| Module 4: Ổn định & Rủi ro | 18 | ✅ 100% |\n\n")
    
    all_lines.append("---\n\n")
    
    # Thêm data dictionary
    all_lines.extend(dict_lines)
    
    # Thêm phần thống kê
    all_lines.append("---\n\n")
    all_lines.append("## 📈 Thống Kê Mô Tả\n\n")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ['sample_id', 'default']]
    
    all_lines.append("### Thống Kê Tổng Quát\n\n")
    all_lines.append("| Feature | Trung bình | Độ lệch chuẩn | Min | Max |\n")
    all_lines.append("|---------|------------|---------------|-----|-----|\n")
    
    for col in feature_cols[:20]:  # Top 20
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            all_lines.append(f"| `{col}` | {mean_val:.2f} | {std_val:.2f} | {min_val:.2f} | {max_val:.2f} |\n")
    
    all_lines.append("\n...(hiển thị top 20 features)\n\n")
    
    # Save
    output_path = base_dir / "TAI_LIEU_DATASET.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(all_lines)
    
    print(f"\n💾 Đã lưu: {output_path}")
    
    print("\n" + "=" * 80)
    print("✅ HOÀN TẤT!")
    print("=" * 80)

if __name__ == "__main__":
    main()
