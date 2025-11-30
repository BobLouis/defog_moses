# 分區段除霧功能說明

## 概述
`defog_proposed_atmo_section_claude.py` 中的 `defog_img()` 函數已經整合了分區段處理功能，可以根據圖像的不同區域使用不同的大氣光值A進行除霧，並在區段交界處使用padding進行平滑過渡，避免分界感。

## 全域參數配置

**在文件開頭定義了以下全域變數，可直接修改以調整行為：**

```python
# ========== 全域參數設定 ==========
SECTION_COUNT = 20          # 將圖片從上到下切成的區段數量
PADDING_LENGTH = 50         # 區段交界處的padding長度（像素）
A_CHANGE_N = 1              # 每n個pixel的A變化（區段內）
A_CHANGE_LIMIT = 1          # padding區域內A值變化的步進（每n個pixel）

# 是否使用分區段處理（True: 使用分區段, False: 使用傳統單一A值）
USE_SECTIONS = True
# ==================================
```

## 使用方法

### 方法1: 直接調整全域變數（推薦）
在 `defog_proposed_atmo_section_claude.py` 文件開頭修改參數：

```python
# 修改區段數量
SECTION_COUNT = 30          # 改成30個區段

# 修改padding長度
PADDING_LENGTH = 100        # 改成100像素的平滑過渡

# 關閉分區段處理，使用傳統模式
USE_SECTIONS = False
```

然後直接使用：
```python
from defog_proposed_atmo_section_claude import defog_img
import numpy as np
from PIL import Image

# 讀取圖像
img = Image.open('hazy_image.jpg').convert('RGB')
H = np.array(img)

# 直接調用，會自動使用全域變數
defog_output, A, BestPsi = defog_img(H)

# 保存結果
Image.fromarray(defog_output).save('result.png')
```

### 方法2: 與 main_score.py 配合使用
`main_score.py` 已經導入了這個版本的 `defog_img`：
```python
from defog_proposed_atmo_section_claude import defog_img
```

**使用步驟：**
1. 在 `defog_proposed_atmo_section_claude.py` 開頭調整全域變數
2. 直接執行：
```bash
python main_score.py
```

預設會使用分區段處理模式。

## 四個關鍵全域參數

### 1. `SECTION_COUNT` (預設: 20)
- **說明**: 將圖片從上到下切成的區段數量
- **作用**: 決定圖像垂直方向上分成多少個區域，每個區域計算獨立的大氣光值A
- **建議值**: 10-30，根據圖像高度和霧霾變化調整
- **調整方式**: 直接修改 `SECTION_COUNT = 20` 這一行

### 2. `PADDING_LENGTH` (預設: 50)
- **說明**: 區段交界處的padding長度（像素）
- **作用**: 在相鄰區段交界處創建平滑過渡區域，避免明顯的分界線
- **建議值**: 30-100像素，視圖像解析度而定
- **調整方式**: 直接修改 `PADDING_LENGTH = 50` 這一行

### 3. `A_CHANGE_N` (預設: 1)
- **說明**: 區段內每n個pixel的A變化量
- **作用**: 目前設為1，表示每個pixel都使用該區段的A值
- **建議值**: 通常保持為1
- **備註**: 此參數保留供未來擴展使用

### 4. `A_CHANGE_LIMIT` (預設: 1)
- **說明**: padding區域每n個pixel的A變化步進
- **作用**: 控制padding區域內線性插值的步進
  - 設為1：每個pixel都單獨計算插值
  - 設為2：每2個pixel使用相同的插值結果（可加速但稍微降低平滑度）
- **建議值**: 1-3，視性能需求而定
- **調整方式**: 直接修改 `A_CHANGE_LIMIT = 1` 這一行

## 工作原理

### 1. 分區段計算A值
- 圖像被垂直分成 `num_sections` 個區段
- 每個區段獨立計算自己的大氣光值 A = (Ar, Ag, Ab)
- 這樣可以適應圖像不同區域的霧霾濃度變化

### 2. 創建A值空間分布圖 (A_map)
- A_map 是一個與原圖相同大小的3通道陣列
- 每個像素位置都有對應的A值
- 在區段主體部分，使用該區段的固定A值
- 在padding區域，使用線性插值進行平滑過渡

### 3. Padding平滑過渡
在每個區段的末端 `padding_length` 像素範圍內：
```
A(row) = (1 - α) * A_current + α * A_next
其中 α = (row - main_end) / padding_length
```
這確保A值從當前區段平滑過渡到下一個區段。

### 4. 使用A_map除霧
- 每個像素使用其對應的A值進行歸一化
- 計算傳輸圖 t(x)
- 恢復無霧圖像

## 優勢

1. **自適應處理**: 不同區域使用不同的A值，更好地處理霧霾分布不均的情況
2. **無分界感**: padding機制消除了區段之間的明顯分界線
3. **向後兼容**: 可以通過 `use_sections=False` 切換回傳統模式
4. **參數可調**: 四個參數可以靈活調整以適應不同圖像

## 範例比較

### 傳統方法（單一A值）
- 整張圖片使用同一個A值
- 對於霧霾分布不均的圖像效果不佳
- 可能導致某些區域過度除霧或除霧不足

### 分區段方法（多A值 + Padding）
- 每個區段使用適合該區域的A值
- Padding確保平滑過渡，無分界感
- 更好地處理複雜霧霾場景

## 注意事項

1. **區段數量**: 不要設置過多區段（建議不超過50），否則可能導致過度分割
2. **Padding長度**: 確保 `padding_length < (圖像高度 / num_sections)`，否則會重疊
3. **記憶體使用**: 分區段模式需要額外存儲A_map，對於超大圖像需注意記憶體
4. **返回值**: 為保持向後兼容，函數仍返回第一個區段的A值

## 測試

可以使用提供的測試程式：
```bash
python test_section_defog.py
```

這將生成視覺化結果，包括：
- 原始圖像 vs 除霧結果
- A_map各通道的熱力圖
- 每個區段A值的變化圖
- A值沿垂直方向的變化曲線（驗證平滑過渡）
