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
SECTION_COUNT = 20          # 將圖片從上到下切成的區段數量
PADDING_LENGTH = 50         # 區段交界處的padding長度（像素）
A_CHANGE_N = 1              # 每n個pixel的A變化（區段內）
A_CHANGE_LIMIT = 1          # padding區域內A值變化的步進（每n個pixel）

# 是否使用分區段處理（True: 使用分區段, False: 使用傳統單一A值）
USE_SECTIONS = True
# ==================================


def estimate_atmospheric_light(hazy_image_section, window_size=8):
	"""
	估計圖像區段的大氣光值 A
	"""
	H_ds = hazy_image_section[::2, ::2, :]
	dark_channel_ds = minimum_filter(np.min(H_ds, axis=2), size=window_size)
	idx = np.argmax(dark_channel_ds)
	y, x = np.unravel_index(idx, dark_channel_ds.shape)
	A = H_ds[y, x, :]
	return A


def create_atmospheric_light_map(hazy_image, window_size=8, epsilon=1e-6):
	"""
	創建大氣光值的空間變化圖

	使用全域變數:
	- SECTION_COUNT: 將圖片從上到下切成的區段數量
	- PADDING_LENGTH: 區段交界處的padding長度（像素）
	- A_CHANGE_N: 區段內每n個pixel的變化量
	- A_CHANGE_LIMIT: padding區域每n個pixel的變化量

	參數:
	- window_size: 暗通道計算的窗口大小
	- epsilon: 防止除以零的小常數

	返回:
	- A_map: shape為(height, width, 3)的大氣光值圖
	- A_values: 每個區段的A值列表
	"""
	H = hazy_image.astype(np.float32)
	height, width, channels = H.shape

	# 初始化大氣光值圖
	A_map = np.zeros((height, width, 3), dtype=np.float32)

	# 計算每個區段的高度
	section_height = height // SECTION_COUNT

	# 存儲每個區段的大氣光值
	A_values = []

	# 為每個區段計算大氣光值
	for i in range(SECTION_COUNT):
		start_row = i * section_height
		end_row = (i + 1) * section_height if i < SECTION_COUNT - 1 else height

		# 提取當前區段
		section = H[start_row:end_row, :, :]

		# 估計此區段的大氣光值
		A = estimate_atmospheric_light(section, window_size)
		A_values.append(A)

	# 填充大氣光值圖，包含padding的線性過渡
	for i in range(SECTION_COUNT):
		start_row = i * section_height
		end_row = (i + 1) * section_height if i < SECTION_COUNT - 1 else height

		# 當前區段的A值
		A_current = A_values[i]

		# 計算padding區域
		if i < SECTION_COUNT - 1:
			# 不是最後一個區段，需要考慮與下一個區段的過渡
			A_next = A_values[i + 1]

			# 區段主體部分（不包含padding）
			main_end = min(end_row - PADDING_LENGTH, end_row)
			A_map[start_row:main_end, :, :] = A_current

			# Padding過渡區域 - 使用A_CHANGE_LIMIT控制步進
			if main_end < end_row:
				for j in range(main_end, end_row, A_CHANGE_LIMIT):
					# 計算線性插值權重
					alpha = (j - main_end) / PADDING_LENGTH
					# 線性插值A值
					A_interpolated = (1 - alpha) * A_current + alpha * A_next
					# 如果A_CHANGE_LIMIT > 1，填充連續的像素
					for k in range(min(A_CHANGE_LIMIT, end_row - j)):
						A_map[j + k, :, :] = A_interpolated
		else:
			# 最後一個區段，直接填充
			A_map[start_row:end_row, :, :] = A_current

	return A_map, A_values


def defog_img_with_sections(hazy_image, psi=1, t0=0.2, window_size=8, epsilon=1e-6):
	"""
	使用分區段大氣光估計的除霧函數（使用全域變數）

	使用全域變數:
	- SECTION_COUNT: 區段數量
	- PADDING_LENGTH: padding長度
	- A_CHANGE_N: 區段內每n個pixel的變化量
	- A_CHANGE_LIMIT: padding區域每n個pixel的變化量

	參數:
	- hazy_image: 輸入的霧霾圖像
	- psi: 傳輸圖計算的參數
	- t0: 傳輸圖的下界
	- window_size: 暗通道窗口大小
	- epsilon: 防止除以零的小常數
	"""
	H = hazy_image.astype(np.float32)

	# 創建大氣光值圖（使用全域變數）
	A_map, A_values = create_atmospheric_light_map(
		hazy_image, window_size, epsilon
	)

	# 對每個通道進行歸一化
	H_norm = np.empty_like(H, dtype=np.float32)
	for c in range(3):
		H_norm[:, :, c] = H[:, :, c] / (A_map[:, :, c] + epsilon)

	# 計算歸一化圖像的平均強度 K
	K = np.mean(H_norm, axis=2)

	# 計算最小歸一化值
	min_norm = np.min(H_norm, axis=2)

	# 計算傳輸圖
	temp = 3 * K + 3 * min_norm
	t = (temp - psi * 3 * K * min_norm) / (temp + epsilon)
	t = np.clip(t, t0, 1)

	# 恢復無霧圖像
	t_expanded = t[:, :, np.newaxis]
	D = (H - A_map) / t_expanded + A_map
	D = np.clip(D, 0, 255).astype(np.uint8)

	return D, A_map, A_values


def defog_img(hazy_image, psi=1, t0=0.2, window_size=8, epsilon=1e-6):
	"""
	除霧函數 - 支援分區段處理或傳統單一A值處理

	使用全域變數控制行為:
	- USE_SECTIONS: 是否使用分區段處理
	- SECTION_COUNT: 區段數量
	- PADDING_LENGTH: padding長度
	- A_CHANGE_N: 區段內每n個pixel的變化
	- A_CHANGE_LIMIT: padding區域內的變化步進

	參數:
	- hazy_image: 輸入的霧霾圖像
	- psi: 傳輸圖計算的參數
	- t0: 傳輸圖的下界
	- window_size: 暗通道窗口大小
	- epsilon: 防止除以零的小常數

	返回:
	- D: 除霧後的圖像
	- A: 大氣光值 (如果USE_SECTIONS=True，返回第一個區段的A值以保持兼容性)
	- BestPsi: psi值
	"""
	H = hazy_image.astype(np.float32)
	BestPsi = psi

	if USE_SECTIONS:
		# === 分區段處理模式 ===
		height, width, channels = H.shape

		# 計算每個區段的高度
		section_height = height // SECTION_COUNT

		# 存儲每個區段的大氣光值
		A_values = []

		# 為每個區段計算大氣光值
		for i in range(SECTION_COUNT):
			start_row = i * section_height
			end_row = (i + 1) * section_height if i < SECTION_COUNT - 1 else height

			# 提取當前區段
			section = H[start_row:end_row, :, :]

			# 估計此區段的大氣光值
			A_section = estimate_atmospheric_light(section, window_size)
			A_values.append(A_section)

		# 創建大氣光值的空間變化圖
		A_map = np.zeros((height, width, 3), dtype=np.float32)

		# 填充大氣光值圖，包含padding的線性過渡
		for i in range(SECTION_COUNT):
			start_row = i * section_height
			end_row = (i + 1) * section_height if i < SECTION_COUNT - 1 else height

			# 當前區段的A值
			A_current = A_values[i]

			# 計算padding區域
			if i < SECTION_COUNT - 1:
				# 不是最後一個區段，需要考慮與下一個區段的過渡
				A_next = A_values[i + 1]

				# 區段主體部分（不包含padding）
				main_end = min(end_row - PADDING_LENGTH, end_row)
				A_map[start_row:main_end, :, :] = A_current

				# Padding過渡區域 - 使用A_CHANGE_LIMIT控制步進
				if main_end < end_row:
					for j in range(main_end, end_row, A_CHANGE_LIMIT):
						# 計算線性插值權重
						alpha = (j - main_end) / PADDING_LENGTH
						# 線性插值A值
						A_interpolated = (1 - alpha) * A_current + alpha * A_next
						# 如果A_CHANGE_LIMIT > 1，填充連續的像素
						for k in range(min(A_CHANGE_LIMIT, end_row - j)):
							A_map[j + k, :, :] = A_interpolated
			else:
				# 最後一個區段，直接填充
				A_map[start_row:end_row, :, :] = A_current

		# 使用A_map進行除霧
		H_norm = np.empty_like(H, dtype=np.float32)
		for c in range(3):
			H_norm[:, :, c] = H[:, :, c] / (A_map[:, :, c] + epsilon)

		# 計算歸一化圖像的平均強度 K
		K = np.mean(H_norm, axis=2)

		# 計算最小歸一化值
		min_norm = np.min(H_norm, axis=2)

		# 計算傳輸圖
		temp = 3 * K + 3 * min_norm
		t = (temp - psi * 3 * K * min_norm) / (temp + epsilon)
		t = np.clip(t, t0, 1)

		# 恢復無霧圖像
		t_expanded = t[:, :, np.newaxis]
		D = (H - A_map) / t_expanded + A_map
		D = np.clip(D, 0, 255).astype(np.uint8)

		# 為了保持向後兼容性，返回第一個區段的A值
		A = A_values[0]

	else:
		# === 傳統單一A值處理模式 ===
		# 根據論文描述，先對 hazy 圖像進行下採樣（因子為 2），用於大氣光 A 的估計
		H_ds = H[::2, ::2, :]

		# 計算下採樣圖像的暗通道：對每個像素在窗口內取三個通道的最小值，然後再做最小濾波
		dark_channel_ds = minimum_filter(np.min(H_ds, axis=2), size=window_size)

		# 選擇暗通道中最大值對應的像素作為大氣光 A（從下採樣圖像中取得）
		idx = np.argmax(dark_channel_ds)
		y, x = np.unravel_index(idx, dark_channel_ds.shape)
		A = H_ds[y, x, :]  # 大氣光向量

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