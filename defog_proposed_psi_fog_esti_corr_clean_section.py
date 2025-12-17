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

# ========== 全域參數設定 ==========
# 分區段處理參數
SECTION_COUNT = 4        # 將圖片從上到下切成的區段數量
PADDING_LENGTH = 100     # 區段交界處的padding長度（像素）
PSI_CHANGE_LIMIT = 1     # padding區域內Psi值變化的步進（每n個pixel）

# 是否使用分區段處理（True: 使用分區段, False: 使用傳統單一Psi值）
USE_SECTIONS = True
# ==================================

def predict_psi(image):
	# AVERAGE,20.3747,0.8434,7.4070
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


def create_psi_map(hazy_image):
	"""
	創建Psi值的空間變化圖

	使用全域變數:
	- SECTION_COUNT: 將圖片從上到下切成的區段數量
	- PADDING_LENGTH: 區段交界處的padding長度（像素）
	- PSI_CHANGE_LIMIT: padding區域內Psi值變化的步進（每n個pixel）

	參數:
	- hazy_image: 輸入圖像（RGB，np.uint8 or np.float32）

	返回:
	- psi_map: shape為(height, width)的Psi值圖
	- psi_values: 每個區段的Psi值列表
	"""
	# Ensure image is in uint8 format
	if hazy_image.dtype == np.float32:
		img = np.clip(hazy_image, 0, 255).astype(np.uint8)
	else:
		img = hazy_image

	height, width, channels = img.shape

	# 初始化Psi值圖
	psi_map = np.zeros((height, width), dtype=np.float32)

	# 計算每個區段的高度
	section_height = height // SECTION_COUNT

	# 存儲每個區段的Psi值
	psi_values = []

	# 為每個區段計算Psi值
	for i in range(SECTION_COUNT):
		start_row = i * section_height
		end_row = (i + 1) * section_height if i < SECTION_COUNT - 1 else height

		# 提取當前區段
		section = img[start_row:end_row, :, :]

		# 估計此區段的Psi值
		psi = predict_psi(section)
		psi_values.append(psi)
		print(f"Section {i}: BestPsi = {psi:.4f}")

	# 填充Psi值圖，包含padding的線性過渡
	for i in range(SECTION_COUNT):
		start_row = i * section_height
		end_row = (i + 1) * section_height if i < SECTION_COUNT - 1 else height

		# 當前區段的Psi值
		psi_current = psi_values[i]

		# 計算padding區域
		if i < SECTION_COUNT - 1:
			# 不是最後一個區段，需要考慮與下一個區段的過渡
			psi_next = psi_values[i + 1]

			# 區段主體部分（不包含padding）
			main_end = min(end_row - PADDING_LENGTH, end_row)
			psi_map[start_row:main_end, :] = psi_current

			# Padding過渡區域 - 使用PSI_CHANGE_LIMIT控制步進
			if main_end < end_row:
				for j in range(main_end, end_row, PSI_CHANGE_LIMIT):
					# 計算線性插值權重
					alpha = (j - main_end) / PADDING_LENGTH
					# 線性插值Psi值
					psi_interpolated = (1 - alpha) * psi_current + alpha * psi_next
					# 如果PSI_CHANGE_LIMIT > 1，填充連續的像素
					for k in range(min(PSI_CHANGE_LIMIT, end_row - j)):
						psi_map[j + k, :] = psi_interpolated
		else:
			# 最後一個區段，直接填充
			psi_map[start_row:end_row, :] = psi_current

	return psi_map, psi_values


def defog_img(hazy_image, psi=1, t0=0.2, window_size=8, epsilon=1e-6):
	"""
	基於論文方法對輸入的 hazy 圖像進行去霧處理，支援分區段Psi值處理

	使用全域變數:
	- USE_SECTIONS: 是否使用分區段處理
	- SECTION_COUNT: 區段數量
	- PADDING_LENGTH: padding長度
	- PSI_CHANGE_LIMIT: padding區域內的變化步進

	參數:
	hazy_image: 輸入圖像（RGB，np.uint8）
	psi: 擬合係數（當USE_SECTIONS=False時使用）
	t0: 傳輸圖的下界（例如 0.2）
	window_size: 最小濾波器窗口大小（15x15）
	epsilon: 防止除零的小常數

	返回:
	D: 去霧後的圖像（np.uint8）
	A: 大氣光向量（3,）
	BestPsi: Psi值（如果USE_SECTIONS=True，返回第一個區段的Psi值以保持兼容性）
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

	if USE_SECTIONS:
		# === 分區段處理模式 ===
		# 創建Psi值的空間變化圖
		psi_map, psi_values = create_psi_map(hazy_image)

		# 使用原始全解析度圖像進行後續處理：對每個通道進行歸一化(除以 A)
		H_norm = np.empty_like(H, dtype=np.float32)
		for c in range(3):
			H_norm[:, :, c] = H[:, :, c] / (A[c] + epsilon)

		# 計算歸一化圖像的平均強度 K（每個像素的均值）
		K = np.mean(H_norm, axis=2)

		# 計算最小歸一化值
		min_norm = np.min(H_norm, axis=2)

		# 計算傳輸圖 - 使用psi_map（每個像素有不同的psi值）
		temp = 3*K + 3*min_norm
		t = (temp - psi_map*3*K*min_norm) / (temp + epsilon)
		# 限制傳輸圖的下界
		t = np.clip(t, t0, 1)

		# 利用傳輸圖恢復無霧圖像： D(x) = (H(x) - A) / t(x) + A
		t_expanded = t[:, :, np.newaxis]
		D = (H - A) / t_expanded + A
		D = np.clip(D, 0, 255).astype(np.uint8)

		# 為了保持向後兼容性，返回第一個區段的Psi值
		BestPsi = psi_values[0]

	else:
		# === 傳統單一Psi值處理模式 ===
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