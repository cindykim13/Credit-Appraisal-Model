# Danh Mục Script Train Mô Hình & File Kết Quả

> **Cập nhật**: 2026-03-05  
> **Mục đích**: Liệt kê vị trí các script huấn luyện mô hình và các file ghi nhận kết quả tương ứng trong toàn bộ dự án.  
> **Gốc thư mục dự án**: `d:\SystemFolders\Desktop\NCKH\`

---

## Tổng quan kiến trúc pipeline

```
[Tầng 1] 4 Module chuyên biệt → scores S1, S2, S3, S4
              ↓  score_collector.py
[Tầng 2] Meta-Model → PD score (Probability of Default)
```

---

## 1. Tầng 1 — Các Module Scoring

### 1.1 Module 1 — Financial Health

| Loại | Đường dẫn |
|------|-----------|
| **Script train** | `scripts/train_module1.py` |
| **Logic module** | `src/modules/module1_financial.py` |
| **Model đã train** | `models/module1.pkl` |
| **Kết quả ghi nhận** | `baocao/phase3_results.md` → mục *"Module 1: Financial Health"* |

**Mô tả script**: Huấn luyện Ensemble XGBoost + LightGBM với semi-supervised blend  
`target = 0.70 × domain_proxy + 0.30 × (1 − default)`. Validate bằng R², MAE và sanity check.

**Kết quả chính** (post-blend):

| Metric | Giá trị |
|--------|---------|
| R² (val) | 0.7014 |
| Sanity gap | +0.063 ✅ |
| target_corr | −0.130 |

---

### 1.2 Module 2 — Credit Behavior

| Loại | Đường dẫn |
|------|-----------|
| **Script train** | *(tích hợp trong pipeline — không có file train riêng)* |
| **Logic module** | `src/modules/module2_credit.py` |
| **Model đã train** | `models/module2.pkl` |
| **Test suite** | `tests/test_module2.py` |
| **Kết quả ghi nhận** | `baocao/phase3_results.md` → mục *"Module 2: Credit Behavior"* |

**Mô tả**: Ensemble XGBoost + Random Forest với target proxy dựa trên lịch sử quá hạn.  
Module có correlation tự nhiên với default cao nhất (`corr = −0.202`), không cần semi-supervised blend.

**Kết quả chính**:

| Metric | Giá trị |
|--------|---------|
| R² (val) | 0.9705 |
| MAE (val) | 0.0152 |
| Sanity gap | +0.043 ✅ |

---

### 1.3 Module 3 — Business Quality

| Loại | Đường dẫn |
|------|-----------|
| **Script train** | `scripts/train_module3.py` |
| **Logic module** | `src/modules/module3_business.py` |
| **Model đã train** | `models/module3.pkl` |
| **Kết quả ghi nhận** | `baocao/phase3_results.md` → mục *"Module 3: Business Quality"* |

**Mô tả script**: Huấn luyện Single LightGBM (vượt trội Random Forest với ΔR²=0.1317) với semi-supervised blend.  
`target = 0.70 × domain_proxy + 0.30 × (1 − default)`.

**Kết quả chính** (post-blend):

| Metric | Giá trị |
|--------|---------|
| R² (val) | 0.6717 |
| Sanity gap | +0.080 ✅ |
| target_corr | −0.229 |

---

### 1.4 Module 4 — Stability & Risk

| Loại | Đường dẫn |
|------|-----------|
| **Script train** | `scripts/train_module4.py` |
| **Logic module** | `src/modules/module4_stability.py` |
| **Model đã train** | `models/module4.pkl` |
| **Kết quả ghi nhận** | `baocao/phase3_results.md` → mục *"Module 4: Stability & Risk"* |

**Mô tả script**: Huấn luyện Ensemble XGBoost + LightGBM với semi-supervised blend (áp dụng từ đầu).  
Có post-prediction hard rule: `past_bankruptcy == 1 → S₄ = min(S₄, 0.30)`.

**Kết quả chính**:

| Metric | Giá trị |
|--------|---------|
| R² (val) | 0.5739 |
| MAE (val) | 0.0717 |
| Sanity gap | +0.057 ✅ |
| target_corr | −0.537 |

---

## 2. Bước Trung Gian — Thu thập Scores Tầng 1

| Loại | Đường dẫn |
|------|-----------|
| **Script** | `scripts/score_collector.py` |
| **Output train** | `models/scores_train.csv` |
| **Output val** | `models/scores_val.csv` |
| **Output test** | `models/scores_test.csv` |

**Mô tả**: Chạy sau khi 4 modules đã được train. Dùng `CreditScoringPipeline` để generate S1–S4 scores cho toàn bộ 3 tập dữ liệu (train/val/test) dưới dạng CSV. Các file CSV này là **đầu vào** của Meta-Model.

**Cấu trúc mỗi file CSV**:

| Cột | Mô tả |
|-----|-------|
| `S1` | Điểm Module 1 — Financial Health ∈ [0, 1] |
| `S2` | Điểm Module 2 — Credit Behavior ∈ [0, 1] |
| `S3` | Điểm Module 3 — Business Quality ∈ [0, 1] |
| `S4` | Điểm Module 4 — Stability & Risk ∈ [0, 1] |
| `default` | Nhãn thực tế (0 = không vỡ nợ, 1 = vỡ nợ) |

---

## 3. Tầng 2 — Meta-Model (PD Scoring)

| Loại | Đường dẫn |
|------|-----------|
| **Script train** | `scripts/train_meta_model.py` |
| **Logic module** | `src/modules/meta_model.py` |
| **Model đã train** | `models/meta_model.pkl` |
| **Kết quả kỹ thuật** | `models/phase4_results.md` |
| **Kết quả tổng hợp** | `baocao/phase4_results.md` |

**Mô tả script**: Train XGBoost Meta-Model trên 8 features (4 scores gốc + 4 interaction features: `min_score`, `S1_x_S2`, `score_std`, `weighted_avg`). Threshold được tối ưu bằng PR curve (threshold = 0.8451).

**Kết quả cuối cùng** (ALL TARGETS MET ✅):

| Metric | Val | Test | Target | Trạng thái |
|--------|-----|------|--------|-----------|
| AUC-ROC | **0.8773** | 0.7955 | > 0.87 | ✅ PASS |
| KS Statistic | **0.7338** | 0.5329 | > 0.70 | ✅ PASS |
| Brier Score | **0.0637** | 0.0781 | < 0.09 | ✅ PASS |
| F1-Score | **0.7436** | 0.5753 | > 0.65 | ✅ PASS |

---

## 4. Bảng Tổng Hợp Toàn Bộ

| Thứ tự chạy | Script | Đầu ra chính | File kết quả |
|-------------|--------|-------------|--------------|
| 1️⃣ | `scripts/train_module1.py` | `models/module1.pkl` | `baocao/phase3_results.md` §Module1 |
| 1️⃣ | *(pipeline tích hợp)* | `models/module2.pkl` | `baocao/phase3_results.md` §Module2 |
| 1️⃣ | `scripts/train_module3.py` | `models/module3.pkl` | `baocao/phase3_results.md` §Module3 |
| 1️⃣ | `scripts/train_module4.py` | `models/module4.pkl` | `baocao/phase3_results.md` §Module4 |
| 2️⃣ | `scripts/score_collector.py` | `models/scores_*.csv` | *(log trực tiếp ra stdout)* |
| 3️⃣ | `scripts/train_meta_model.py` | `models/meta_model.pkl` | `baocao/phase4_results.md`, `models/phase4_results.md` |

---

## 5. Các File Liên Quan Khác

### Source code modules

| File | Mô tả |
|------|-------|
| `src/modules/base_module.py` | Abstract base class — interface chung cho 4 modules |
| `src/modules/meta_model.py` | MetaModel với feature engineering + threshold optimization + SHAP |
| `src/pipeline.py` | `CreditScoringPipeline` — orchestrate 4 modules end-to-end |
| `src/config.py` | Cấu hình đường dẫn, hyperparameters |
| `src/utils/data_loader.py` | Load train/val/test splits |
| `src/utils/preprocessor.py` | `DataPreprocessor` — normalize, encode, drop low-variance features |

### Test suite

| File | Phạm vi kiểm thử |
|------|-----------------|
| `tests/test_module1.py` | 5 unit tests cho Module 1 |
| `tests/test_module2.py` | 5 unit tests cho Module 2 |
| `tests/test_module3.py` | 5 unit tests cho Module 3 |
| `tests/test_module4.py` | 5 unit tests cho Module 4 |
| `tests/test_meta_model.py` | 5 unit tests cho Meta-Model |

> **Kết quả test tổng thể**: 25/25 passed ✅ (`pytest tests/`)

### Các file kết quả & báo cáo

| File | Nội dung |
|------|----------|
| `baocao/phase3_results.md` | Kết quả chi tiết 4 modules — target proxy design, model comparison, metrics, lessons learned |
| `baocao/phase4_results.md` | Kết quả Meta-Model — kiến trúc, tuning journey (Steps 1→3), SHAP analysis, hạn chế, hướng cải thiện |
| `models/phase4_results.md` | Kết quả kỹ thuật raw của Phase 4 (metrics bảng, tuning steps) |
| `baocao/bao_cao.md` | Báo cáo NCKH hoàn chỉnh (văn bản học thuật) |

---

## 6. Thứ Tự Chạy Lại Toàn Bộ Pipeline

```bash
# Bước 1: Train 4 modules (thứ tự tùy ý, độc lập nhau)
python scripts/train_module1.py
python scripts/train_module3.py
python scripts/train_module4.py
# Module 2 được train qua pipeline.fit() trong bước 2

# Bước 2: Thu thập scores trung gian (cần module1/2/3/4.pkl)
python scripts/score_collector.py

# Bước 3: Train Meta-Model (cần scores_train/val/test.csv)
python scripts/train_meta_model.py

# Kiểm tra toàn bộ
pytest tests/
```
