# defog_2023.py

import numpy as np
from scipy.ndimage import minimum_filter
from fog_estimate_antigravity import predict_psi



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