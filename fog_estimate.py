import numpy as np


dataset = "SOTS_inout"

def predict_psi(image):
	# AVERAGE,20.3747,0.8434,7.4070 inout 
	# AVERAGE,16.4091,0.6092,15.6578  Ohaze
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

	return fog_score


