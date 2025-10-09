import cv2
import numpy as np

def fog_density(image_path):
    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # 轉為浮點與灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 計算平均亮度與標準差（對比度）
    mean_brightness = np.mean(gray)
    std_contrast = np.std(gray)

    # 正規化
    brightness_norm = mean_brightness / 255.0
    contrast_norm = min(std_contrast / 128.0, 1.0)

    # 計算霧濃度指標
    fog_score = brightness_norm * (1 - contrast_norm)
    fog_percent = fog_score * 100

    return round(fog_percent, 2), mean_brightness, std_contrast

# 測試兩張圖片
fog1, b1, c1 = fog_density("../dataset/SOTS_in/hazy/001_hazy.png")
fog2, b2, c2 = fog_density("../dataset/SOTS_in/clear/001_clear.png")

print("霧圖：", fog1, "% (亮度:", b1, "對比:", c1, ")")
print("清晰圖：", fog2, "% (亮度:", b2, "對比:", c2, ")")
