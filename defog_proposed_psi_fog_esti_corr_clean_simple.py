# defog_2023.py
import numpy as np
from scipy.ndimage import minimum_filter

""" 公式整理
H(x) = D(x)*t(x) + A*(1-t(x))
D(x) = ((H(x) - A) / t(x)) + A
=================================================================
A 從暗通道中選擇最亮的像素作為 A = (Ar, Ag, Ab) (patch = 15*15, downsample 2)
t(x) = 1 - w * (K_H(x) / A) * (1 - S_H(x)/S_D(x))
=================================================================
which
S(x) 為飽和度, K(x) 為像素強度值
w = 1.25
K_H(x) = Hr(x) + Hg(x) + Hb(x) / 3
	=> (K_H(x) / A) = (Hr(x)/Ar + Hg(x)/Ag + Hb(x)/Ab) / 3
	=> let H_norm[] = [Hr(x)/Ar, Hg(x)/Ag, Hb(x)/Ab]
	=> (K_H(x) / A) = avg(H_norm)
S_D(x) = S_H(x) * (2 - S_H(x))
S_H(x) = 1 - (min_c(H_c(x)) / K_H(x)), which c is rgb
=================================================================
	=> t(x) = 1 - w * (K_H(x) / A) * (1 - 1/(2 - S_H(x)) )
"""

def predict_psi(image):
    """
    基於硬體霧氣評分預測最佳 PSI 值
    回歸公式：BestPsi = 0.009308 × HW_FogScore + 0.927009
    限制範圍：0.5 ~ 1.2
    """
    # V5 硬體霧氣評分算法
    # Downsample by 2 + 取 R channel
    gray = image[::2, ::2, 0]
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
    
    fog_score = max(0, min(100, fog_score))
    
    # 套用回歸公式
    BestPsi = 0.009308 * fog_score + 0.927009
    
    # 限制範圍 0.5 ~ 1.2
    BestPsi = max(0.5, min(1.2, BestPsi))
    
    return BestPsi


def defog_img(hazy_image, psi=1, t0=0.2, window_size=8, epsilon=1e-6):
    """
    基於論文方法對輸入的 hazy 圖像進行去霧處理，返回無霧圖像、暗通道圖像、大氣光和傳輸圖。
    參數:
    hazy_image: 輸入圖像（RGB，np.uint8）
    psi: 擬合係數（論文中固定為 1.25）
    t0: 傳輸圖的下界（例如 0.2）
    window_size: 最小濾波器窗口大小（15x15）
    epsilon: 防止除零的小常數
    返回:
    D: 去霧後的圖像（np.uint8）
    dark_output: 暗通道圖像（灰階複製為三通道，np.uint8）
    A: 大氣光向量（3,）
    t: 傳輸圖（float32）
    """
    # 將輸入轉換為 float 型態以便計算
    H = hazy_image.astype(np.float32)
    
    # 根據論文描述，先對 hazy 圖像進行下採樣（因子為 2），用於大氣光 A 的估計
    H_ds = H[::2, ::2, :]
    
    # 計算下採樣圖像的暗通道：對每個像素在窗口內取三個通道的最小值，然後再做最小濾波
    dark_channel_ds = minimum_filter(np.min(H_ds, axis=2), size=window_size)
    
    # 選擇暗通道中最大值對應的像素作為大氣光 A（從下採樣圖像中取得）
    idx = np.argmax(dark_channel_ds)
    y, x = np.unravel_index(idx, dark_channel_ds.shape)
    A = H_ds[y, x, :]  # 大氣光向量
    
    # Calculate optimal PSI based on fog estimation
    BestPsi = predict_psi(hazy_image)
    psi = BestPsi
    
    # 使用原始全解析度圖像進行後續處理：對每個通道進行歸一化(除以 A)
    H_norm = np.empty_like(H, dtype=np.float32)
    for c in range(3):
        H_norm[:, :, c] = H[:, :, c] / (A[c] + epsilon)
    
    # 計算歸一化圖像的平均強度 K（每個像素的均值）
    K = np.mean(H_norm, axis=2)
    
    # 計算飽和度 S，公式：S = 1 - (min(R_norm, G_norm, B_norm) / (K + epsilon))
    min_norm = np.min(H_norm, axis=2)
    temp = 3*K + 3*min_norm
    t = (temp - psi*3*K*min_norm) / (temp + epsilon)
    
    # 限制傳輸圖的下界
    t = np.clip(t, t0, 1)
    
    # 利用傳輸圖恢復無霧圖像： D(x) = (H(x) - A) / t(x) + A
    t_expanded = t[:, :, np.newaxis]
    D = (H - A) / t_expanded + A
    D = np.clip(D, 0, 255).astype(np.uint8)

    return D, A, BestPsi
