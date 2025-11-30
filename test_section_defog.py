import cv2
import numpy as np
import matplotlib.pyplot as plt
from defog_proposed_atmo_section_claude import (
	defog_img_with_sections,
	create_atmospheric_light_map,
	SECTION_COUNT,
	PADDING_LENGTH,
	A_CHANGE_N,
	A_CHANGE_LIMIT
)


def test_section_defog(image_path):
	"""
	測試分區段除霧函數（使用全域變數）
	"""
	# 讀取圖像
	hazy_image = cv2.imread(image_path)
	if hazy_image is None:
		print(f"無法讀取圖像: {image_path}")
		return

	hazy_image = cv2.cvtColor(hazy_image, cv2.COLOR_BGR2RGB)

	print(f"圖像尺寸: {hazy_image.shape}")
	print(f"使用全域參數:")
	print(f"  SECTION_COUNT = {SECTION_COUNT}")
	print(f"  PADDING_LENGTH = {PADDING_LENGTH}")
	print(f"  A_CHANGE_N = {A_CHANGE_N}")
	print(f"  A_CHANGE_LIMIT = {A_CHANGE_LIMIT}")

	# 執行除霧（使用全域變數）
	defog_result, A_map, A_values = defog_img_with_sections(
		hazy_image,
		psi=1,
		t0=0.2,
		window_size=8
	)

	# 打印每個區段的A值
	print("\n每個區段的大氣光值A:")
	for i, A in enumerate(A_values):
		print(f"區段 {i+1}: R={A[0]:.2f}, G={A[1]:.2f}, B={A[2]:.2f}")

	# 視覺化結果
	fig, axes = plt.subplots(2, 3, figsize=(15, 10))

	# 原始霧霾圖像
	axes[0, 0].imshow(hazy_image.astype(np.uint8))
	axes[0, 0].set_title('Original Hazy Image')
	axes[0, 0].axis('off')

	# 除霧結果
	axes[0, 1].imshow(defog_result)
	axes[0, 1].set_title('Defogged Image')
	axes[0, 1].axis('off')

	# A_map的R通道
	axes[0, 2].imshow(A_map[:, :, 0], cmap='jet')
	axes[0, 2].set_title('A Map (R Channel)')
	axes[0, 2].axis('off')
	plt.colorbar(axes[0, 2].imshow(A_map[:, :, 0], cmap='jet'), ax=axes[0, 2])

	# A_map的G通道
	axes[1, 0].imshow(A_map[:, :, 1], cmap='jet')
	axes[1, 0].set_title('A Map (G Channel)')
	axes[1, 0].axis('off')
	plt.colorbar(axes[1, 0].imshow(A_map[:, :, 1], cmap='jet'), ax=axes[1, 0])

	# A_map的B通道
	axes[1, 1].imshow(A_map[:, :, 2], cmap='jet')
	axes[1, 1].set_title('A Map (B Channel)')
	axes[1, 1].axis('off')
	plt.colorbar(axes[1, 1].imshow(A_map[:, :, 2], cmap='jet'), ax=axes[1, 1])

	# A值變化圖
	section_indices = list(range(1, len(A_values) + 1))
	A_r = [A[0] for A in A_values]
	A_g = [A[1] for A in A_values]
	A_b = [A[2] for A in A_values]

	axes[1, 2].plot(section_indices, A_r, 'r-o', label='R')
	axes[1, 2].plot(section_indices, A_g, 'g-o', label='G')
	axes[1, 2].plot(section_indices, A_b, 'b-o', label='B')
	axes[1, 2].set_xlabel('Section Number')
	axes[1, 2].set_ylabel('Atmospheric Light Value')
	axes[1, 2].set_title('A Values per Section')
	axes[1, 2].legend()
	axes[1, 2].grid(True)

	plt.tight_layout()
	plt.savefig('section_defog_result.png', dpi=150, bbox_inches='tight')
	print("\n結果已保存到 section_defog_result.png")
	plt.show()

	return defog_result, A_map, A_values


def visualize_a_map_profile(A_map):
	"""
	視覺化A_map中心列的值變化，觀察padding的平滑過渡效果
	"""
	height, width, _ = A_map.shape
	center_col = width // 2

	# 提取中心列的A值
	A_profile_r = A_map[:, center_col, 0]
	A_profile_g = A_map[:, center_col, 1]
	A_profile_b = A_map[:, center_col, 2]

	plt.figure(figsize=(12, 6))
	plt.plot(A_profile_r, 'r-', label='R Channel', linewidth=2)
	plt.plot(A_profile_g, 'g-', label='G Channel', linewidth=2)
	plt.plot(A_profile_b, 'b-', label='B Channel', linewidth=2)
	plt.xlabel('Row (Pixel Height)', fontsize=12)
	plt.ylabel('Atmospheric Light Value', fontsize=12)
	plt.title('A Value Profile (Center Column) - Showing Smooth Transitions', fontsize=14)
	plt.legend(fontsize=10)
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig('a_map_profile.png', dpi=150, bbox_inches='tight')
	print("A值變化曲線已保存到 a_map_profile.png")
	plt.show()


if __name__ == "__main__":
	# 請替換成你的圖像路徑
	image_path = "your_hazy_image.jpg"

	print("請確保已設定正確的圖像路徑")
	print("使用範例:")
	print("  result, A_map, A_values = test_section_defog('path/to/your/image.jpg')")
	print("  visualize_a_map_profile(A_map)")
