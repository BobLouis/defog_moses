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



def get_section_atmo_map(image, section_count=20, padding_length=50, a_change_n=10, a_change_limit=2.0):
    """
    Generate an Atmospheric Light (A) map with section-based estimation and smooth transitions.
    
    Parameters:
    - image: Input image (RGB).
    - section_count: Number of vertical sections.
    - padding_length: Length of the transition padding between sections (pixels).
    - a_change_n: The 'n' in 'change per n pixels'.
    - a_change_limit: Max allowed change in A per n pixels (per channel).
    """
    h, w = image.shape[:2]
    section_height = h // section_count
    
    # 1. Calculate A for each section
    a_values = []
    
    # Helper to calculate A for a patch
    def estimate_a(img_patch):
        if img_patch.size == 0:
            return np.array([255, 255, 255], dtype=np.float32)
            
        patch_float = img_patch.astype(np.float32)
        # Downsample for speed and robustness
        patch_ds = patch_float[::2, ::2, :]
        if patch_ds.size == 0:
             patch_ds = patch_float
             
        # Dark channel
        dark_channel_ds = minimum_filter(np.min(patch_ds, axis=2), size=8) # window_size=8 from default
        
        idx = np.argmax(dark_channel_ds)
        y, x = np.unravel_index(idx, dark_channel_ds.shape)
        return patch_ds[y, x, :]

    for i in range(section_count):
        start_y = i * section_height
        end_y = (i + 1) * section_height if i < section_count - 1 else h
        
        if start_y >= h:
            break
            
        section_img = image[start_y:end_y, :]
        
        if section_img.shape[0] == 0:
            a_values.append(a_values[-1] if a_values else np.array([255, 255, 255], dtype=np.float32))
            continue
            
        a_val = estimate_a(section_img)
        a_values.append(a_val)
    
    # 2. Initialize map with block values
    # A_map is (H, W, 3)
    a_map = np.zeros((h, w, 3), dtype=np.float32)
    
    for i in range(len(a_values)):
        start_y = i * section_height
        end_y = (i + 1) * section_height if i < section_count - 1 else h
        # Broadcast A value (1, 1, 3) to (H_section, W, 3)
        a_map[start_y:end_y, :, :] = a_values[i][np.newaxis, np.newaxis, :]
        
    # 3. Smooth the boundaries (Padding)
    half_padding = padding_length // 2
    
    # Check slope constraint
    limit_slope = a_change_limit / a_change_n
    
    for i in range(len(a_values) - 1):
        b_y = (i + 1) * section_height
        
        t_start = max(0, b_y - half_padding)
        t_end = min(h, b_y + half_padding)
        t_len = t_end - t_start
        
        if t_len > 0:
            a_prev = a_values[i]
            a_next = a_values[i+1]
            
            # Check slope for each channel
            diff = np.abs(a_next - a_prev)
            slope = diff / t_len
            
            if np.any(slope > limit_slope):
                print(f"Warning: A change at section {i}-{i+1} exceeds limit! Max Slope: {np.max(slope):.5f} > {limit_slope:.5f}")
            
            # Linear interpolation for each channel
            # shape (t_len, 3)
            gradient = np.linspace(a_prev, a_next, t_len).astype(np.float32)
            
            # Broadcast to width: (t_len, 1, 3) -> (t_len, W, 3)
            a_map[t_start:t_end, :, :] = gradient[:, np.newaxis, :]
            
    return a_map, a_values

def defog_img(hazy_image, psi=1, t0=0.2, window_size=8, epsilon=1e-6):

	H = hazy_image.astype(np.float32)
	
	# 根據論文描述，先對 hazy 圖像進行下採樣（因子為 2），用於大氣光 A 的估計
	# H_ds = H[::2, ::2, :]
	
	# 計算下採樣圖像的暗通道：對每個像素在窗口內取三個通道的最小值，然後再做最小濾波
	# dark_channel_ds = minimum_filter(np.min(H_ds, axis=2), size=window_size)
	
	# 選擇暗通道中最大值對應的像素作為大氣光 A（從下採樣圖像中取得）
	# idx = np.argmax(dark_channel_ds)
	# y, x = np.unravel_index(idx, dark_channel_ds.shape)
	# A = H_ds[y, x, :]  # 大氣光向量
	
	# Use Section-based Atmospheric Light Map
	# Configurable variables
	SECTION_COUNT = 20
	PADDING_LENGTH = 50
	A_CHANGE_N = 10
	A_CHANGE_LIMIT = 2.0 # Allow 2.0 pixel value change per 10 pixels (approx 0.2 per pixel)
	
	A_map, A_list = get_section_atmo_map(
		hazy_image,
		section_count=SECTION_COUNT,
		padding_length=PADDING_LENGTH,
		a_change_n=A_CHANGE_N,
		a_change_limit=A_CHANGE_LIMIT
	)
	
	# Use the average A for return (or the first one, or the global one if needed for logging)
	# Calculating global A just for compatibility/logging
	A = np.mean(A_list, axis=0) 
	
	psi = 1
	BestPsi = psi
	
	# 使用 A_map 進行歸一化
	# H_norm = np.empty_like(H, dtype=np.float32)
	# for c in range(3):
	# 	H_norm[:, :, c] = H[:, :, c] / (A[c] + epsilon)
	
	# Vectorized normalization with A_map
	H_norm = H / (A_map + epsilon)

	# 計算歸一化圖像的平均強度 K（每個像素的均值）
	K = np.mean(H_norm, axis=2)

	# 計算飽和度 S，公式：S = 1 - (min(R_norm, G_norm, B_norm) / (K + epsilon))
	min_norm = np.min(H_norm, axis=2)

	temp = 3*K + 3*min_norm
	t = (temp - psi*3*K*min_norm) / (temp + epsilon)
	# 限制傳輸圖的下界
	t = np.clip(t, t0, 1)

	# 利用傳輸圖恢復無霧圖像： D(x) = (H(x) - A) / t(x) + A
	# Note: A is now A_map
	t_expanded = t[:, :, np.newaxis]
	D = (H - A_map) / t_expanded + A_map
	D = np.clip(D, 0, 255).astype(np.uint8)

	return D, A, BestPsi