import numpy as np
import os
from glob import glob
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# dataset = "SOTS_inout"
dataset = "OHaze"

def predict_psi(image):
	# AVERAGE,20.3747,0.8434,7.4070 inout 
	# AVERAGE,16.4091,0.6092,15.6578  Ohaze
	"""
	Calculate best PSI value based on fog score estimation.
	Parameters:
	image: Input image (RGB, np.uint8 or np.float32)
	Returns:
	BestPsi: Optimal PSI value predicted using learned weights from advanced features 
             (Dark Channel, Saturation, Contrast, Entropy, etc.).
	"""
	# Ensure image is in uint8 format
	if image.dtype == np.float32:
		img = np.clip(image, 0, 255).astype(np.uint8)
	else:
		img = image

	# Convert to PIL/HSV for features
	# Note: Image processing here is done on-the-fly. 
	# To keep it matching exactly what was optimized, we replicate the feature extraction logic.
	
	if image.dtype == np.float32:
		img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
	else:
		img_uint8 = image
		
	img_pil = Image.fromarray(img_uint8)
	img_hsv = np.array(img_pil.convert('HSV'))
	img_gray = np.array(img_pil.convert('L')).astype(np.float32)

	# --- Feature Extraction ---
	
	# 1. Dark Channel Proxy (Mean) - Positive correlation with haze
	min_rgb = np.min(img_uint8, axis=2)
	dc_mean = np.mean(min_rgb)
	dc_std = np.std(min_rgb)

	# 2. Saturation (Std) - Haze reduces saturation variance
	s_chan = img_hsv[:, :, 1]
	sat_std = np.std(s_chan)

	# 3. Contrast (Global RMS) - Haze reduces contrast
	contrast = np.std(img_gray)

	# 4. Entropy
	hist, _ = np.histogram(img_gray, bins=256, range=(0, 256), density=True)
	hist = hist[hist > 0]
	entropy = -np.sum(hist * np.log2(hist))

	# 5. Sharpness (Laplacian)
	# Slicing approach for speed
	c = img_gray[1:-1, 1:-1]
	n = img_gray[:-2, 1:-1] + img_gray[2:, 1:-1] + img_gray[1:-1, :-2] + img_gray[1:-1, 2:]
	lap = np.abs(n - 4*c)
	sharpness_mean = np.mean(lap)
	sharpness_std = np.std(lap)

	# 6. Basic RGB Means
	r_mean = np.mean(img_uint8[:,:,0])
	g_mean = np.mean(img_uint8[:,:,1])
	# b_mean was not selected by Lasso as significant enough compared to others

	# 7. Old Statistical Features (re-calc for model)
	max_val = np.max(img_gray)
	min_val = np.min(img_gray)
	avg_intensity = np.mean(img_gray)
	dyn_range = max_val - min_val
	avg_dev = np.mean(np.abs(img_gray - avg_intensity))
	
	# --- Linear Regression Model (R2 ~ 0.48) ---
	# Features selected: G_Mean, DC_Mean, Contrast, AvgDev, Sharpness_Std, R_Mean, DynRange, DC_Std, Sat_Std, Entropy, Sharpness_Mean
	
	best_psi_predicted = (
		-0.015183 * g_mean +
		0.012467 * dc_mean +
		0.027052 * contrast +
		-0.018002 * avg_dev +
		-0.015769 * sharpness_std +
		0.003979 * r_mean +
		-0.003765 * dyn_range +
		-0.012721 * dc_std +
		0.006207 * sat_std +
		-0.102047 * entropy +
		0.007422 * sharpness_mean +
		2.656480 # Intercept
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
	output_path = f"./dataset/{dataset}/report/fog_grid_antigravity.csv"
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
