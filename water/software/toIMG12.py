import os
import numpy as np
from PIL import Image

def q4_12_hex_to_float(hex_str):
    """
    將 Q4.12 (16-bit signed) HEX 文字轉為浮點數
    """
    val = int(hex_str, 16)
    if val >= 2**15:
        val -= 2**16  # 還原二補數 signed
    return val / (2**12)

def convert_txt_folder_to_png(txt_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    txt_files = sorted([f for f in os.listdir(txt_folder) if f.endswith(".txt")])

    for txt_file in txt_files:
        txt_path = os.path.join(txt_folder, txt_file)

        with open(txt_path, "r") as f:
            hex_vals = [line.strip() for line in f if line.strip() != ""]

        # 還原浮點並限制到 [0, 1]
        float_vals = [q4_12_hex_to_float(h) for h in hex_vals]
        float_vals = np.clip(float_vals, 0.0, 1.0)

        # reshape → [3, 256, 256]
        tensor_chw = np.array(float_vals, dtype=np.float32).reshape((3, 256, 256))
        tensor_hwc = np.transpose(tensor_chw, (1, 2, 0))  # [H, W, C]

        # 轉為 uint8 RGB 圖
        img_u8 = (tensor_hwc * 255).astype(np.uint8)

        # 輸出 PNG
        basename = os.path.splitext(txt_file)[0]
        output_path = os.path.join(output_folder, f"{basename}.png")
        Image.fromarray(img_u8, mode="RGB").save(output_path)
        print(f"✅ 已儲存: {output_path}")

# 使用方法
convert_txt_folder_to_png("./LSUI_out_txt12", "./LSUI_out_txt12")
