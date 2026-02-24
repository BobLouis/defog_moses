import numpy as np

def defog_img(hazy_image, psi=1.0, t0=0.1, epsilon=1e-6):
    """
    基於 IEEE VLSI 2025 論文 "Efficient Pipelined Hardware Architecture for 
    Depth-Map-Based Image Dehazing System" 的去霧實作。
    
    主要模組:
    1. AALE: Saturation-Based Local Airlight Estimation (飽和度局部大氣光估計)
    2. DMTME: Depth-Map-Based Transmission Map Estimation (深度圖傳輸圖估計)

    參數:
    hazy_image: 輸入圖像 (RGB, np.uint8)
    psi: 論文中的 beta 係數，控制去霧程度 (默認 1.0，可調大以增強去霧)
    t0: 傳輸圖下界 (論文建議 0.1)
    epsilon: 防止除零的小常數

    返回:
    D: 去霧後的圖像 (np.uint8)
    A_global: 全域大氣光向量 (用於顯示或兼容性)
    """
    
    # 1. 預處理: 轉換為 0-1 float
    H = hazy_image.astype(np.float32) / 255.0
    h, w, c = H.shape
    
    # 2. 計算基礎統計量 (Pixel-wise, 無窗口操作)
    # I_min: 每個像素 RGB 中的最小值 (Eq. 1 的簡化硬體版)
    I_min = np.min(H, axis=2)
    # I_max: 每個像素 RGB 中的最大值 (即 V_I, Eq. 16)
    I_max = np.max(H, axis=2)
    # I_int: 每個像素的強度均值 (Eq. 10)
    I_int = np.mean(H, axis=2)
    
    # ---------------------------------------------------------
    # 3. AALE: 自適應大氣光估計 (Adaptive Atmospheric Light Estimation)
    # ---------------------------------------------------------
    
    # 3.1 全域大氣光 (Global Airlight) - Eq. 6
    # 論文使用 I_min 作為暗通道 I_dark
    I_dark = I_min 
    # 找到暗通道中最亮的位置
    flat_idx = np.argmax(I_dark)
    y_idx, x_idx = np.unravel_index(flat_idx, I_dark.shape)
    A_global = H[y_idx, x_idx, :] # (3,) vector
    
    # 3.2 局部大氣光 (Local Airlight) - Eq. 7, Eq. 12
    # 計算飽和度比率 I_sat (Eq. 12)
    # I_sat = I_dark / (I_int + I_dark)
    I_sat = I_dark / (I_int + I_dark + epsilon)
    
    # 計算 Eq. 7 中的修正項
    # Term X = [(I_int - I_dark) * (I_dark + I_int)] / I_sat
    # 也就是 (I_int^2 - I_dark^2) / I_sat
    term_numerator = (I_int - I_dark) * (I_dark + I_int)
    term_correction = term_numerator / (I_sat + epsilon)
    
    # 擴展維度以進行廣播運算 (H, W, 1)
    I_sat_exp = I_sat[:, :, np.newaxis]
    term_correction_exp = term_correction[:, :, np.newaxis]
    
    # 計算 A_local (Eq. 7)
    # A_local = A_global * I_sat - A_global * correction
    # 注意：根據公式特性，高對比區域(I_sat小)會被大幅減去，這符合物理特性(物體反射光不應包含大氣光)
    A_local = A_global * I_sat_exp - A_global * term_correction_exp
    
    # 限制範圍，避免負值 (因公式中的減法可能導致負值)
    A_local = np.clip(A_local, 0.0, 1.0)

    # ---------------------------------------------------------
    # 4. DMTME: 深度圖與傳輸圖估計
    # ---------------------------------------------------------
    
    # 4.1 深度圖估計 d(z) - Eq. 14, 15, 16
    # V_I = I_max (已計算)
    # S_I_depth = 1 - min/max (Eq. 15) - 注意這裡的 S_I 定義與 AALE 不同
    S_I_depth = 1.0 - (I_min / (I_max + epsilon))
    
    # 論文參數: theta0=0.12, theta1=0.96, theta2=-0.78
    theta_0 = 0.12
    theta_1 = 0.96
    theta_2 = -0.78
    
    d_z = theta_0 + theta_1 * I_max + theta_2 * S_I_depth
    
    # 4.2 傳輸圖估計 t(z) - Eq. 13
    # t(z) = exp(-beta * d(z))
    # 這裡將輸入參數 psi 作為 beta 使用
    beta = psi 
    t = np.exp(-beta * d_z)
    
    # 限制傳輸圖下界 (Eq. 5 context)
    t = np.clip(t, t0, 1.0)
    
    # ---------------------------------------------------------
    # 5. 場景恢復 (Scene Recovery)
    # ---------------------------------------------------------
    
    # J(z) = (I(z) - A_local) / t(z) + A_local (Eq. 5 變體)
    t_expanded = t[:, :, np.newaxis]
    
    D = (H - A_local) / t_expanded + A_local
    
    # 後處理: Clip 到 0-1 並轉回 uint8
    D = np.clip(D, 0.0, 1.0)
    D = (D * 255).astype(np.uint8)
    
    # 將 A_global 轉回 0-255 uint8 格式以便顯示
    A_global_uint8 = (A_global * 255).astype(np.uint8)

    return D, A_global_uint8, 1