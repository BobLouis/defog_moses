# main.py
import numpy as np
import os
import time
from PIL import Image
from glob import glob

# 根據不同的版本、跑不同的dataset 調整!!!
from defog_proposed import defog_img
defog_version = "test917"
dataset = "SOTS_out"

def main():
    hazy_dir = f"./dataset/{dataset}/hazy"
    output_defog_dir = f"./dataset/{dataset}/result_{defog_version}"
    # output_dark_dir = f"./dataset/{dataset}/dark{defog_version}"

    os.makedirs(output_defog_dir, exist_ok=True)
    # os.makedirs(output_dark_dir, exist_ok=True)

    hazy_files = sorted(glob(os.path.join(hazy_dir, "*.png")))

    for hazy_path in hazy_files:
        full_name = os.path.splitext(os.path.basename(hazy_path))[0]
        base_name = full_name.split('_')[0]
        output_defog_path = os.path.join(output_defog_dir, f"{base_name}_{defog_version}.png")

        print(f"\n處理中: {hazy_path}")
        print(f"輸出結果: {output_defog_path}")

        try:
            img = Image.open(hazy_path).convert('RGB')
            H = np.array(img)

            start_time = time.time()
            defog_output, A = defog_img(H)
            end_time = time.time()
            diff_time = end_time - start_time

            Image.fromarray(defog_output).save(output_defog_path)

            print(f"大氣光 A: {A}")
            print(f"執行時間 = {diff_time:.3f} 秒 \t {int(diff_time*1000)} 毫秒")

        except Exception as e:
            print(f"處理 {hazy_path} 時發生錯誤: {e}")

if __name__ == "__main__":
    main()
