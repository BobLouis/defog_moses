import numpy as np
import os
from glob import glob
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


dataset = "SOTS_inout"
# dataset = "OHaze"

def predict_psi(image):
	# AVERAGE,20.3747,0.8434,7.4070 inout 
	# AVERAGE,16.4091,0.6092,15.6578  Ohaze
	"""
	Calculate best PSI value based on fog score estimation.
	Parameters:
	image: Input image (RGB, np.uint8 or np.float32)
	Returns:
	BestPsi: Optimal PSI value predicted using learned weights from component scores.
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
	# fog_score = (fog_score_range * 2 + fog_score_deviation + fog_score_edge) / 4
	# fog_score = np.clip(fog_score, 0, 100)

	# 使用優化後的線性回歸權重 (BestPsi = w1*Range + w2*Dev + w3*Edge + b)
	# Coefficients: [0.00335279 0.00282501 0.00530094]
	# Intercept: 0.6459711197436037
	best_psi_predicted = (
		0.00335279 * fog_score_range +
		0.00282501 * fog_score_deviation +
		0.00530094 * fog_score_edge +
		0.64597112
	)

	return best_psi_predicted


def main():
	"""
	Process all images in the dataset and output fog scores to CSV
	"""
	hazy_dir = f"./dataset/{dataset}/hazy"

	# Get all hazy images
	hazy_files = sorted(glob(os.path.join(hazy_dir, "*.png")))

	if not hazy_files:
		print(f"⚠️ 找不到圖片在：{hazy_dir}")
		return

	results = []
	fog_scores = []

	print(f"處理 {len(hazy_files)} 張圖片...")

	for hazy_path in tqdm(hazy_files, desc="計算霧氣評分"):
		# Get base name (e.g., "01" from "01_hazy.png")
		full_name = os.path.splitext(os.path.basename(hazy_path))[0]
		base_name = full_name.split('_')[0]

		try:
			# Load image
			img = Image.open(hazy_path).convert('RGB')
			img_array = np.array(img)

			# Calculate fog score
			fog_score = predict_psi(img_array)

			results.append({
				"Image": base_name,
				"FogScore": fog_score
			})
			fog_scores.append(fog_score)

		except Exception as e:
			print(f"\n處理 {hazy_path} 時發生錯誤: {e}")

	# Calculate average and save to CSV
	if results:
		df = pd.DataFrame(results)

		# Add average row
		avg_fog_score = np.mean(fog_scores)
		avg_row = pd.DataFrame([{
			"Image": "AVERAGE",
			"FogScore": avg_fog_score
		}])
		df = pd.concat([df, avg_row], ignore_index=True)

		# Save to CSV
		os.makedirs(f"./dataset/{dataset}/report", exist_ok=True)
		csv_path = f"./dataset/{dataset}/report/fog_score.csv"
		df.to_csv(csv_path, index=False, float_format="%.4f")

		print(f"\n✅ 霧氣評分結果已儲存到：{csv_path}")
		print(f"平均霧氣評分: {avg_fog_score:.4f}")
		print(f"總共處理 {len(fog_scores)} 張圖片")
	else:
		print("⚠️ 沒有成功處理的圖片。")


def merge_and_regression():
	"""
	Merge fog_score and BestPsi data, calculate linear regression
	FogScore (x) vs BestPsi (y)
	"""
	# Read fog_score.csv
	fog_score_path = f"./dataset/{dataset}/report/fog_score.csv"
	if not os.path.exists(fog_score_path):
		print(f"⚠️ 找不到 fog_score.csv：{fog_score_path}")
		print("請先執行 main() 生成 fog_score.csv")
		return

	# Read BestPsi data
	bestpsi_path = f"./dataset/{dataset}/report/score_optimize_psi_grid.csv"
	if not os.path.exists(bestpsi_path):
		print(f"⚠️ 找不到 BestPsi 數據：{bestpsi_path}")
		return

	# Load data
	df_fog = pd.read_csv(fog_score_path)
	df_psi = pd.read_csv(bestpsi_path)

	# Remove AVERAGE rows
	df_fog = df_fog[df_fog['Image'] != 'AVERAGE'].copy()
	df_psi = df_psi[df_psi['Image'] != 'AVERAGE'].copy()

	# Merge on Image column
	df_merged = pd.merge(df_fog, df_psi[['Image', 'BestPsi']], on='Image', how='inner')

	if len(df_merged) == 0:
		print("⚠️ 沒有匹配的數據")
		return

	print(f"成功匹配 {len(df_merged)} 筆數據")

	# Prepare data for regression
	X = df_merged['FogScore'].values.reshape(-1, 1)  # fog_score as x
	y = df_merged['BestPsi'].values  # BestPsi as y

	# Calculate linear regression
	model = LinearRegression()
	model.fit(X, y)

	# Get regression parameters
	slope = model.coef_[0]
	intercept = model.intercept_

	# Calculate R² score
	y_pred = model.predict(X)
	r2 = r2_score(y, y_pred)

	# Add predicted values to dataframe
	df_merged['BestPsi_Predicted'] = y_pred
	df_merged['Residual'] = y - y_pred

	# Save merged data to CSV
	output_path = f"./dataset/{dataset}/report/fog_grid.csv"
	df_merged.to_csv(output_path, index=False, float_format="%.6f")

	# Print regression results
	print(f"\n{'='*60}")
	print(f"線性回歸結果 (FogScore -> BestPsi)")
	print(f"{'='*60}")
	print(f"回歸方程式: BestPsi = {slope:.6f} × FogScore + {intercept:.6f}")
	print(f"R² 值: {r2:.6f}")
	print(f"斜率 (slope): {slope:.6f}")
	print(f"截距 (intercept): {intercept:.6f}")
	print(f"{'='*60}")
	print(f"\n✅ 合併數據已儲存到：{output_path}")

	# Save regression summary
	summary_path = f"./dataset/{dataset}/report/fog_regression_summary.txt"
	with open(summary_path, 'w', encoding='utf-8') as f:
		f.write(f"線性回歸分析 (FogScore -> BestPsi)\n")
		f.write(f"{'='*60}\n\n")
		f.write(f"資料集: {dataset}\n")
		f.write(f"樣本數: {len(df_merged)}\n\n")
		f.write(f"回歸方程式:\n")
		f.write(f"  BestPsi = {slope:.6f} × FogScore + {intercept:.6f}\n\n")
		f.write(f"回歸係數:\n")
		f.write(f"  斜率 (slope): {slope:.6f}\n")
		f.write(f"  截距 (intercept): {intercept:.6f}\n\n")
		f.write(f"模型評估:\n")
		f.write(f"  R² 值: {r2:.6f}\n\n")
		f.write(f"統計量:\n")
		f.write(f"  FogScore 平均值: {X.mean():.4f}\n")
		f.write(f"  FogScore 標準差: {X.std():.4f}\n")
		f.write(f"  BestPsi 平均值: {y.mean():.4f}\n")
		f.write(f"  BestPsi 標準差: {y.std():.4f}\n")

	print(f"✅ 回歸分析摘要已儲存到：{summary_path}\n")

	return df_merged, slope, intercept, r2


if __name__ == "__main__":
	main()
	merge_and_regression()
