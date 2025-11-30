# 快速開始指南

## 如何調整參數？

### 步驟1: 打開文件
用編輯器打開 `defog_proposed_atmo_section_claude.py`

### 步驟2: 找到參數設定區域（第26-35行）
```python
# ========== 全域參數設定 ==========
# 分區段處理參數
SECTION_COUNT = 20          # 將圖片從上到下切成的區段數量
PADDING_LENGTH = 50         # 區段交界處的padding長度（像素）
A_CHANGE_N = 1              # 每n個pixel的A變化（區段內）
A_CHANGE_LIMIT = 1          # padding區域內A值變化的步進（每n個pixel）

# 是否使用分區段處理（True: 使用分區段, False: 使用傳統單一A值）
USE_SECTIONS = True
# ==================================
```

### 步驟3: 修改數值
例如，想要更多區段和更長的平滑過渡：
```python
SECTION_COUNT = 30          # 改成30個區段
PADDING_LENGTH = 80         # 改成80像素的padding
```

### 步驟4: 保存文件

### 步驟5: 執行程式
```bash
python main_score.py
```

就這麼簡單！

---

## 四個核心參數

| 參數名稱 | 預設值 | 說明 | 常用範圍 |
|---------|-------|------|---------|
| `SECTION_COUNT` | 20 | 圖片切成幾個區段 | 10-30 |
| `PADDING_LENGTH` | 50 | 區段交界處平滑過渡長度 | 30-100 |
| `A_CHANGE_N` | 1 | 區段內變化步進（保留參數） | 1 |
| `A_CHANGE_LIMIT` | 1 | Padding內變化步進 | 1-3 |

---

## 常見調整情境

### 情境1: 圖片出現橫向條紋
**問題**: 區段交界處有明顯分界線
**解決**: 增加 `PADDING_LENGTH`
```python
PADDING_LENGTH = 100   # 從50改成100
```

### 情境2: 除霧效果不均勻
**問題**: 有些地方霧霾去除不夠或過度
**解決**: 增加 `SECTION_COUNT`
```python
SECTION_COUNT = 30     # 從20改成30
```

### 情境3: 處理速度太慢
**問題**: 大量圖片處理時間過長
**解決**: 增加 `A_CHANGE_LIMIT`
```python
A_CHANGE_LIMIT = 2     # 從1改成2，可加速約1倍
```

### 情境4: 想要關閉分區段功能
**問題**: 想對比傳統方法
**解決**: 設定 `USE_SECTIONS = False`
```python
USE_SECTIONS = False   # 改成False使用傳統方法
```

---

## 更多資訊

- 詳細參數說明：請看 `PARAMETERS_GUIDE.md`
- 完整功能介紹：請看 `README_SECTION.md`
- 測試程式：`test_section_defog.py`

---

## 重要提醒

1. **Padding不能太長**：確保 `PADDING_LENGTH < (圖像高度 / SECTION_COUNT)`
2. **每次只改一個參數**：方便觀察效果
3. **記錄實驗結果**：找到最適合你數據集的參數

祝你調參順利！
