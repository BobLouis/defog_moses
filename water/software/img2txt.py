import os
import numpy as np
import cv2

def float_to_fixed(value, int_bits=1, frac_bits=32):
    """
    將浮點數轉為 Q1.32 格式的 33-bit 二進位字串。
    """
    scaled_value = int(round(value * (1 << frac_bits)))
    max_val = (1 << (int_bits + frac_bits)) - 1
    scaled_value = min(scaled_value, max_val)
    return format(scaled_value, '033b')  # 固定位寬 33 bits（實際硬體你可能用 36~37 位）

def save_rgb_fixed_point_folder(input_dir, output_prefix="LSUI32_image_data_fixed"):
    """
    處理資料夾內所有圖片並輸出為定點數格式
    """
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(input_dir, filename)
        output_txt = f"{output_prefix}{idx}.txt"

        # 讀取並轉換影像為 RGB
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))  # 強制 256x256
        img_array = img.astype(np.float32) / 255.0

        # 拆 R/G/B 並寫入
        R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

        with open(output_txt, "w") as f:
            for channel, name in zip([R, G, B], ["R", "G", "B"]):
                f.write(f"//==== {name} Channel ====\n")
                for row in channel:
                    for val in row:
                        fixed_val = float_to_fixed(val, int_bits=1, frac_bits=32)
                        f.write(fixed_val + "\n")

        print(f"✅ 已儲存 {output_txt}")

# 使用方式（請改為你的資料夾路徑）
save_rgb_fixed_point_folder(r"C:\Users\user\Desktop\underwater\final2\hardware\LSUI\LSUI3_raw")
