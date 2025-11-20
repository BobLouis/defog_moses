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

def predict_psi(image, tile=8, use_y=True, percentile=80):
    # 1) 轉灰階（改用 Y），或維持你的 R 但我建議 Y
    img = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
    if use_y:
        y = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]).astype(np.uint8)
    else:
        y = img[:,:,0]

    H, W = y.shape
    th, tw = H//tile, W//tile
    scores = []

    for i in range(tile):
        for j in range(tile):
            patch = y[i*th:(i+1)*th, j*tw:(j+1)*tw]
            if patch.size == 0: 
                continue

            max_val = patch.max()
            min_val = patch.min()
            avg_intensity = patch.mean()
            dynamic_range = max_val - min_val
            avg_deviation = np.mean(np.abs(patch - avg_intensity))

            # 邊緣：可視邊比例
            diff_h = np.abs(patch[:,1:].astype(np.int16) - patch[:,:-1].astype(np.int16))
            diff_v = np.abs(patch[1:,:].astype(np.int16) - patch[:-1,:].astype(np.int16))
            grad = np.zeros_like(patch, dtype=np.float32)
            grad[:,1:] += diff_h
            grad[1:,:] += diff_v
            # 自適應邊緣閾值：patch 內分位數
            thr = np.percentile(grad, 75)
            visible_edge_ratio = (grad > max(thr, 1)).mean()  # 0~1

            # 重新映射成霧分數（越霧 → 越高）
            # 動態範圍：用相對尺度避免固定 240/100
            dr_score = 100 * (1 - np.clip(dynamic_range/255.0, 0, 1))  # 大範圍→低分，小範圍→高分
            dev_score = 100 * (1 - np.clip(avg_deviation/64.0, 0, 1))  # 粗略縮放，可再校正
            edge_score = 100 * (1 - np.clip(visible_edge_ratio/0.12, 0, 1))  # 12% 可視邊當作「足夠清晰」

            fog_score_tile = (dr_score*2 + dev_score + edge_score) / 4
            scores.append(fog_score_tile)

    if not scores:
        fog_score = 50.0
    else:
        # 2) 用高分位數捕捉局部濃霧
        fog_score = float(np.percentile(scores, percentile))

    fog_score = np.clip(fog_score, 0, 100)
    BestPsi = 0.011099 * fog_score + 0.746386
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
	psi = predict_psi(hazy_image)
	
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

	return D, A