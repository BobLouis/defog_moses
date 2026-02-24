# defog_v5_template.py (基於 proposed_v5 邏輯的模板格式)
import numpy as np
from scipy.ndimage import minimum_filter

""" 公式整理
H(x) = D(x)*t(x) + A*(1-t(x))
D(x) = ((H(x) - A) / t(x)) + A
=================================================================
A 從暗通道中選擇最亮的像素作為 A = (Ar, Ag, Ab) (patch = 8x8, 無降採樣)
t(x) = 1 - w * (K_H(x) / A) * (1 - S_H(x)/S_D(x))
=================================================================
which
S(x) 為飽和度, K(x) 為像素強度值
w = psi (動態計算，範圍 0.5~1.2)
K_H(x) = Hr(x) + Hg(x) + Hb(x) / 3
    => (K_H(x) / A) = (Hr(x)/Ar + Hg(x)/Ag + Hb(x)/Ab) / 3
    => let H_norm[] = [Hr(x)/Ar, Hg(x)/Ag, Hb(x)/Ab]
    => (K_H(x) / A) = avg(H_norm)
S_D(x) = S_H(x) * (2 - S_H(x))
S_H(x) = 1 - (min_c(H_c(x)) / K_H(x)), which c is rgb
=================================================================
    => t(x) = 1 - w * (K_H(x) / A) * (1 - 1/(2 - S_H(x)) )
簡化後：
    t(x) = (temp - psi * 3 * K * min_norm) / temp
    where temp = 3*K + 3*min_norm
"""


def predict_psi(image):
    """
    基於硬體霧氣評分預測最佳 PSI 值（V5 版本：無降採樣）
    回歸公式：BestPsi = 0.009308 × HW_FogScore + 0.927009
    限制範圍：0.5 ~ 1.2
    
    參數:
    image: 輸入圖像（RGB，np.uint8 或 float32）
    
    返回:
    BestPsi: 最佳 PSI 值（float）
    """
    # V5 硬體：直接用 full resolution 取 R channel（不降採樣）
    gray = image[:, :, 0]
    height, width = gray.shape
    
    # 單次掃描找最大最小值
    max_val = 0
    min_val = 255
    
    for i in range(height):
        for j in range(width):
            pixel = int(gray[i, j])
            if pixel > max_val:
                max_val = pixel
            if pixel < min_val:
                min_val = pixel
    
    # 計算動態範圍
    dynamic_range = max_val - min_val
    
    # 霧氣評分（用位移代替除法）
    if dynamic_range >= 240:
        fog_score = 0
    elif dynamic_range <= 100:
        fog_score = 100
    else:
        fog_score = (240 - dynamic_range) >> 1
    
    # 限制範圍
    fog_score = max(0, min(100, fog_score))
    
    # 套用回歸公式
    BestPsi = 0.009308 * fog_score + 0.927009
    
    # 限制範圍 0.5 ~ 1.2
    BestPsi = max(0.5, min(1.2, BestPsi))
    
    return BestPsi


def defog_img(hazy_image, psi=1, t0=0.2, window_size=8, epsilon=1e-6):
    """
    基於 proposed_v5 方法對輸入的 hazy 圖像進行去霧處理，返回無霧圖像、大氣光和最佳 PSI。
    
    參數:
    hazy_image: 輸入圖像（RGB，np.uint8）
    psi: 擬合係數（會被自動計算的 BestPsi 覆蓋）
    t0: 傳輸圖的下界（預設 0.2）
    window_size: 最小濾波器窗口大小（8x8，V5 版本）
    epsilon: 防止除零的小常數
    
    返回:
    D: 去霧後的圖像（np.uint8）
    A: 大氣光向量（3,）
    BestPsi: 自動計算的最佳 PSI 值
    """
    # 將輸入轉換為 float 型態以便計算
    H = hazy_image.astype(np.float32)
    
    # ========== 使用動態 PSI（V5 版本：無降採樣）==========
    BestPsi = predict_psi(H)
    psi = BestPsi
    
    # ========== 計算大氣光 A（V5 版本：直接對 full resolution 做處理）==========
    # 直接對 full resolution 計算暗通道
    dark_channel = np.min(H, axis=2)  # 取 RGB 最小值
    dark_min = minimum_filter(dark_channel, size=window_size)  # 8x8 window min
    
    # 找出最大的 dark channel 值對應的位置
    idx = np.argmax(dark_min)
    y, x = np.unravel_index(idx, dark_min.shape)
    A = H[y, x, :].copy()  # 大氣光向量
    
    # ========== 使用原始全解析度圖像進行去霧處理 ==========
    # 對每個通道進行歸一化（除以 A）
    H_norm = H / (A + epsilon)
    
    # 計算歸一化圖像的平均強度 K（每個像素的均值）
    K = np.mean(H_norm, axis=2)
    
    # 計算最小歸一化值
    min_norm = np.min(H_norm, axis=2)
    
    # 計算傳輸圖 t
    temp = 3 * K + 3 * min_norm
    t = (temp - psi * 3 * K * min_norm) / (temp + epsilon)
    
    # 限制傳輸圖的下界
    t = np.clip(t, t0, 1)
    
    # 利用傳輸圖恢復無霧圖像： D(x) = (H(x) - A) / t(x) + A
    t_expanded = t[:, :, np.newaxis]
    D = (H - A) / t_expanded + A
    D = np.clip(D, 0, 255).astype(np.uint8)

    return D, A, BestPsi