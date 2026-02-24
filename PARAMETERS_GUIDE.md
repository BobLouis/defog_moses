# 參數調整指南

## 快速開始

在 `defog_proposed_atmo_section_claude.py` 文件的第 26-35 行，你會看到：

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

**直接修改這些數值，然後執行 `python main_score.py` 即可！**

## 參數說明與調整建議

### 1. SECTION_COUNT（區段數量）

**這是什麼？**
- 把圖片從上到下切成幾等份
- 每一份都有自己的大氣光值 A

**怎麼調？**
```python
SECTION_COUNT = 10   # 少一點區段，適合霧霾分布較均勻的圖像
SECTION_COUNT = 20   # 預設值，適合大部分情況
SECTION_COUNT = 30   # 多一點區段，適合霧霾分布變化大的圖像
SECTION_COUNT = 40   # 很細緻的分區，但padding要相應增加
```

**注意事項：**
- 區段太少（<10）：可能無法適應霧霾變化
- 區段太多（>50）：可能導致過度分割，且需要更長的padding
- 建議範圍：**10-30**

---

### 2. PADDING_LENGTH（平滑過渡長度）

**這是什麼？**
- 在相鄰區段交界處的平滑過渡區域長度（像素）
- 避免出現明顯的分界線

**怎麼調？**
```python
PADDING_LENGTH = 30    # 短一點，過渡快速但可能有輕微分界感
PADDING_LENGTH = 50    # 預設值，適合大部分圖像
PADDING_LENGTH = 100   # 長一點，過渡非常平滑
PADDING_LENGTH = 150   # 很長的過渡，適合大尺寸圖像
```

**計算公式（重要！）：**
```
PADDING_LENGTH 應該 < (圖像高度 / SECTION_COUNT)

例如：圖像高度 = 1000 像素，SECTION_COUNT = 20
每個區段高度 = 1000 / 20 = 50 像素
則 PADDING_LENGTH 應該 < 50
```

**注意事項：**
- Padding太短（<30）：可能出現分界線
- Padding太長：可能造成區段重疊
- 建議範圍：**30-100** 像素（視圖像解析度調整）

---

### 3. A_CHANGE_N（區段內變化步進）

**這是什麼？**
- 區段內每n個pixel使用相同的A值
- 目前設為1，表示每個pixel都有A值

**怎麼調？**
```python
A_CHANGE_N = 1   # 預設值，每個pixel都使用該區段的A（最精確）
A_CHANGE_N = 2   # 每2個pixel使用相同A（目前未實現此功能）
```

**注意事項：**
- **目前建議保持為 1**
- 此參數保留供未來擴展使用

---

### 4. A_CHANGE_LIMIT（Padding內變化步進）

**這是什麼？**
- 控制padding區域內線性插值的步進
- 可以用來加速計算

**怎麼調？**
```python
A_CHANGE_LIMIT = 1   # 預設值，每個pixel都單獨計算插值（最平滑）
A_CHANGE_LIMIT = 2   # 每2個pixel使用相同插值（稍快，輕微降低平滑度）
A_CHANGE_LIMIT = 3   # 每3個pixel使用相同插值（更快，但可能有細微階梯感）
```

**注意事項：**
- 設為1：最平滑，但計算稍慢
- 設為2-3：可加速計算，對視覺效果影響很小
- 不建議 > 5：會出現明顯的階梯效果
- 建議範圍：**1-3**

---

### 5. USE_SECTIONS（啟用/關閉分區段）

**這是什麼？**
- 控制是否使用分區段處理

**怎麼調？**
```python
USE_SECTIONS = True    # 使用分區段處理（推薦）
USE_SECTIONS = False   # 使用傳統單一A值處理
```

**什麼時候用 False？**
- 想對比傳統方法與分區段方法的差異
- 圖像霧霾分布非常均勻
- 調試時需要快速測試

---

## 實際調整範例

### 範例1：高解析度圖像（1920x1080）
```python
SECTION_COUNT = 30          # 多一點區段以適應大圖
PADDING_LENGTH = 80         # 較長的padding保證平滑
A_CHANGE_N = 1
A_CHANGE_LIMIT = 2          # 稍微加速
USE_SECTIONS = True
```

### 範例2：低解析度圖像（640x480）
```python
SECTION_COUNT = 15          # 少一點區段
PADDING_LENGTH = 40         # 較短的padding
A_CHANGE_N = 1
A_CHANGE_LIMIT = 1
USE_SECTIONS = True
```

### 範例3：霧霾分布變化大的圖像
```python
SECTION_COUNT = 25          # 較多區段捕捉變化
PADDING_LENGTH = 60         # 適中的padding
A_CHANGE_N = 1
A_CHANGE_LIMIT = 1          # 保持最平滑
USE_SECTIONS = True
```

### 範例4：霧霾分布均勻的圖像
```python
SECTION_COUNT = 10          # 較少區段即可
PADDING_LENGTH = 50
A_CHANGE_N = 1
A_CHANGE_LIMIT = 2
USE_SECTIONS = True
```

---

## 調整流程建議

1. **第一次測試** - 使用預設值
   ```python
   SECTION_COUNT = 20
   PADDING_LENGTH = 50
   A_CHANGE_N = 1
   A_CHANGE_LIMIT = 1
   USE_SECTIONS = True
   ```

2. **觀察結果**
   - 如果有明顯分界線 → 增加 `PADDING_LENGTH`
   - 如果霧霾去除不均勻 → 增加 `SECTION_COUNT`
   - 如果處理時間太長 → 增加 `A_CHANGE_LIMIT` 到 2 或 3

3. **微調優化**
   - 每次只調整一個參數
   - 記錄每次調整的結果
   - 找到最適合你的數據集的參數組合

---

## 常見問題

**Q: 圖像出現橫向條紋/分界線？**
A: 增加 `PADDING_LENGTH`，例如從 50 改成 80 或 100

**Q: 除霧效果不均勻？**
A: 增加 `SECTION_COUNT`，例如從 20 改成 30

**Q: 處理速度太慢？**
A: 將 `A_CHANGE_LIMIT` 從 1 改成 2 或 3

**Q: 想對比傳統方法？**
A: 設定 `USE_SECTIONS = False`

**Q: Padding太長導致錯誤？**
A: 確保 `PADDING_LENGTH < (圖像高度 / SECTION_COUNT)`

---

## 修改後執行

修改參數後，直接執行：

```bash
python main_score.py
```

程式會自動使用你設定的全域變數進行處理。
