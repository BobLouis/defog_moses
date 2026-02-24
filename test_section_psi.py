"""
測試 section-based Psi fog estimation
"""
import numpy as np
import cv2
from defog_proposed_psi_fog_esti_corr_clean_section import defog_img, SECTION_COUNT, USE_SECTIONS

def test_section_psi():
	"""測試分區段Psi估計功能"""

	# 創建一個測試圖像 (從上到下霧濃度遞增)
	height, width = 600, 800
	test_img = np.zeros((height, width, 3), dtype=np.uint8)

	# 上半部分：輕霧 (較高對比度)
	test_img[0:height//2, :, :] = 150
	test_img[0:height//2, ::4, :] = 50  # 添加一些對比

	# 下半部分：濃霧 (低對比度，高亮度)
	test_img[height//2:, :, :] = 200
	test_img[height//2:, ::8, :] = 180  # 較低的對比

	print(f"測試配置:")
	print(f"  USE_SECTIONS = {USE_SECTIONS}")
	print(f"  SECTION_COUNT = {SECTION_COUNT}")
	print(f"  圖像尺寸 = {height} x {width}")
	print()

	# 執行除霧
	print("執行除霧處理...")
	defogged_img, A, BestPsi = defog_img(test_img)

	print()
	print(f"結果:")
	print(f"  大氣光 A = {A}")
	print(f"  返回的 BestPsi = {BestPsi:.4f}")
	print(f"  除霧圖像形狀 = {defogged_img.shape}")
	print()
	print("測試完成!")

	return defogged_img, test_img

if __name__ == "__main__":
	test_section_psi()
