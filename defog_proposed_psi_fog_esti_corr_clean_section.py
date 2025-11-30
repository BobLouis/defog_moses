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
	Calculate best PSI value based on fog score estimation.
	Parameters:
	image: Input image (RGB, np.uint8 or np.float32)
	Returns:
	BestPsi: Optimal PSI value calculated as 0.011099 × FogScore + 0.746386
	"""
	# Ensure image is in uint8 format
	if image.dtype == np.float32:
		img = np.clip(image, 0, 255).astype(np.uint8)
	else:
		img = image

	# 簡化霧氣估算：使用 R channel 作為灰階
	gray = img[:, :, 0]
	height, width = gray.shape
	total_pixels = height * width

	# 計算基本統計量
	max_val = np.max(gray)
	min_val = np.min(gray)
	avg_intensity = np.mean(gray)

	# 計算動態範圍
	dynamic_range = max_val - min_val

	# 計算平均偏差
	avg_deviation = np.mean(np.abs(gray - avg_intensity))

	# 計算局部差異（水平和垂直梯度）
	diff_h = np.abs(gray[:, :-1].astype(np.int16) - gray[:, 1:].astype(np.int16))
	diff_v = np.abs(gray[:-1, :].astype(np.int16) - gray[1:, :].astype(np.int16))
	avg_local_diff = (np.sum(diff_h) + np.sum(diff_v)) / (2 * total_pixels)

	# 計算霧氣評分
	# 基於動態範圍 (權重 50%)
	if dynamic_range >= 240:
		fog_score_range = 0
	elif dynamic_range <= 100:
		fog_score_range = 100
	else:
		fog_score_range = 100 - ((dynamic_range - 100) / 140.0) * 100

	# 基於平均偏差 (權重 25%)
	if avg_deviation >= 60:
		fog_score_deviation = 0
	elif avg_deviation <= 20:
		fog_score_deviation = 100
	else:
		fog_score_deviation = 100 - ((avg_deviation - 20) / 40.0) * 100

	# 基於局部差異 (權重 25%)
	if avg_local_diff >= 10:
		fog_score_edge = 0
	elif avg_local_diff <= 1:
		fog_score_edge = 100
	else:
		fog_score_edge = 100 - ((avg_local_diff - 1) / 9.0) * 100

	# 綜合評分
	fog_score = (fog_score_range * 2 + fog_score_deviation + fog_score_edge) / 4
	fog_score = np.clip(fog_score, 0, 100)

	# 計算最佳 PSI 值
	BestPsi = 0.011099 * fog_score + 0.746386
	BestPsi = np.clip(BestPsi, 0.7, 1.2)
	return BestPsi

def predict_psi_v2(image):
    """
    基於「平均霧濃度 (Density)」改進的 PSI 預測函數。
    改進點：
    1. 使用 Dark Channel Mean (暗通道均值) 判斷霧濃度，比單純對比度更準確。
    2. 引入 Saturation Mean (飽和度均值)，霧越濃飽和度越低。
    3. 對於極高濃度的霧 (O-Haze)，限制 Psi 的上限，防止過度除霧導致的破圖。
    
    Returns:
    BestPsi: 推薦的 Psi 值，範圍通常在 [0.85, 1.15] 之間。
    """
    # 1. 格式處理
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image

    # 2. 計算特徵圖
    # Dark Channel: 取 RGB 中最小的通道 (代表該像素最「不亮」的程度)
    # 霧越濃，dark_channel 數值越高 (被大氣光 A 填滿)
    dark_channel = np.min(img, axis=2)
    
    # Saturation: 飽和度 S = 1 - min/max
    # 霧越濃，RGB 數值越接近，飽和度越低
    max_ch = np.max(img, axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        saturation = 1.0 - (dark_channel / (max_ch + 1e-6))
        # 處理除零異常
        saturation[max_ch == 0] = 0

    # 3. 計算全域統計指標
    # avg_dark: 0.0 (無霧) ~ 1.0 (全白濃霧)
    # 一般有霧圖像約在 0.3 ~ 0.7
    avg_dark = np.mean(dark_channel)
    
    # avg_sat: 1.0 (鮮豔) ~ 0.0 (灰白)
    avg_sat = np.mean(saturation)

    # 4. 霧氣評分 (Fog Density Score)
    # 邏輯：暗通道越高 + 飽和度越低 = 霧越濃
    # 權重：暗通道佔 70%，飽和度佔 30%
    # 我們希望 score 落在 0~1 之間
    fog_density = (0.7 * avg_dark) + (0.3 * (1.0 - avg_sat))
    
    # 修正：O-Haze 的 fog_density 可能高達 0.6~0.8
    # 一般戶外無霧圖像 fog_density 約為 0.1~0.2

    print(f"Debug - Avg Dark: {avg_dark:.3f}, Avg Sat: {avg_sat:.3f}, Density: {fog_density:.3f}")

    # 5. 映射到 PSI
    # 您的原始邏輯是線性的，但對於 O-Haze 這種濃霧，
    # Psi 如果太大 (如 > 1.2) 會導致畫面變黑或噪點爆炸。
    # 建議策略：
    # - 輕霧 (Density < 0.3): Psi 約 0.85 (保留一點氛圍，不需強力除霧)
    # - 中霧 (Density 0.3~0.5): Psi 線性增加至 1.1
    # - 濃霧 (Density > 0.5): Psi 稍微回落或持平，避免過度增強噪點 (Soft Clipping)
    
    # 簡單線性映射 (可根據需求調整斜率)
    # 假設 Density 0.1 -> Psi 0.8
    # 假設 Density 0.6 -> Psi 1.1
    
    # 公式：Psi = 0.6 * density + 0.74
    raw_psi = 0.6 * fog_density + 0.74

    # 6. 安全限制 (關鍵)
    # O-Haze 這種濃霧，如果不希望過度除霧，要把上限壓低
    # 這裡設定上限為 1.05 或 1.1，而非 1.2
    # 下限設定為 0.85，避免對無霧圖像過度處理
    final_psi = np.clip(raw_psi, 0.85, 1.3)

    return final_psi


def get_section_psi_map(image, section_count=20, padding_length=50, psi_change_n=10, psi_change_limit=0.02):
    """
    Generate a Psi map with section-based estimation and smooth transitions.
    
    Parameters:
    - image: Input image.
    - section_count: Number of vertical sections.
    - padding_length: Length of the transition padding between sections (pixels).
    - psi_change_n: The 'n' in 'change per n pixels'.
    - psi_change_limit: Max allowed Psi change per n pixels.
    """
    h, w = image.shape[:2]
    section_height = h // section_count
    
    # 1. Calculate Psi for each section
    psi_values = []
    for i in range(section_count):
        start_y = i * section_height
        end_y = (i + 1) * section_height if i < section_count - 1 else h
        
        # Ensure we have a valid crop
        if start_y >= h:
            break
            
        section_img = image[start_y:end_y, :]
        # If section is too small, use global or previous
        if section_img.shape[0] == 0:
            psi_values.append(psi_values[-1] if psi_values else 1.0)
            continue
            
        psi = predict_psi_v2(section_img)
        psi_values.append(psi)
    
    # 2. Initialize map with block values
    psi_map = np.zeros((h, w), dtype=np.float32)
    
    for i in range(len(psi_values)):
        start_y = i * section_height
        end_y = (i + 1) * section_height if i < section_count - 1 else h
        psi_map[start_y:end_y, :] = psi_values[i]
        
    # 3. Smooth the boundaries (Padding)
    half_padding = padding_length // 2
    
    # Check slope constraint (informative for now, or could be used to adjust padding)
    limit_slope = psi_change_limit / psi_change_n
    
    for i in range(len(psi_values) - 1):
        # Boundary y coordinate
        b_y = (i + 1) * section_height
        
        # Define transition range
        t_start = max(0, b_y - half_padding)
        t_end = min(h, b_y + half_padding)
        t_len = t_end - t_start
        
        if t_len > 0:
            psi_prev = psi_values[i]
            psi_next = psi_values[i+1]
            
            # Check if transition is too steep
            actual_change = abs(psi_next - psi_prev)
            if t_len > 0:
                actual_slope = actual_change / t_len
                if actual_slope > limit_slope:
                    print(f"Warning: Psi change at section {i}-{i+1} exceeds limit! Slope: {actual_slope:.5f} > {limit_slope:.5f}")
            
            # Linear interpolation
            # Create a vertical gradient (column vector)
            gradient = np.linspace(psi_prev, psi_next, t_len).astype(np.float32)
            # Broadcast to width
            psi_map[t_start:t_end, :] = gradient[:, np.newaxis]
            
    return psi_map, psi_values


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
	# BestPsi = predict_psi_v2(hazy_image)
	# psi = BestPsi
	
	# Use Section-based Psi Map
	# Configurable variables
	SECTION_COUNT = 20
	PADDING_LENGTH = 50 # Adjust as needed
	PSI_CHANGE_N = 10
	PSI_CHANGE_LIMIT = 0.02
	
	psi_map, psi_list = get_section_psi_map(
		hazy_image, 
		section_count=SECTION_COUNT, 
		padding_length=PADDING_LENGTH,
		psi_change_n=PSI_CHANGE_N,
		psi_change_limit=PSI_CHANGE_LIMIT
	)
	
	# For return value, maybe return the average or the list? 
	# The original code returned a single BestPsi. Let's return the mean for compatibility, or the list.
	BestPsi = np.mean(psi_list)
	psi = psi_map # psi is now a map (H, W)

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