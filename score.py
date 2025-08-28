import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from skimage.color import rgb2lab, deltaE_ciede2000
import numpy as np
import os
from PIL import Image
from glob import glob
import pandas as pd
from tqdm import tqdm

defog_version = "proposed_psi_opti_new" 
dataset = "SOTS_out"  # 可根據需要修改

def compute_psnr(defogged_image, clear_image_path, Xsize, Ysize):
    """計算無霧圖像與清晰參考圖像之間的 PSNR 值"""
    clear_img = Image.open(clear_image_path).convert('RGB')
    if clear_img.width != Xsize or clear_img.height != Ysize:
        clear_img = clear_img.resize((Xsize, Ysize))
    clear_array = np.array(clear_img)
    if defogged_image.shape != clear_array.shape:
        min_height = min(defogged_image.shape[0], clear_array.shape[0])
        min_width = min(defogged_image.shape[1], clear_array.shape[1])
        defogged_image = defogged_image[:min_height, :min_width]
        clear_array = clear_array[:min_height, :min_width]
    try:
        psnr = calculate_psnr(clear_array, defogged_image)
        return psnr
    except Exception as e:
        print(f"Error calculating PSNR: {e}")
        return 0

def compute_ssim(defogged_image, clear_image_path, Xsize, Ysize):
    """計算無霧圖像與清晰參考圖像之間的 SSIM 值"""
    clear_img = Image.open(clear_image_path).convert('RGB')
    if clear_img.width != Xsize or clear_img.height != Ysize:
        clear_img = clear_img.resize((Xsize, Ysize))
    clear_array = np.array(clear_img)

    if defogged_image.shape != clear_array.shape:
        min_height = min(defogged_image.shape[0], clear_array.shape[0])
        min_width = min(defogged_image.shape[1], clear_array.shape[1])
        defogged_image = defogged_image[:min_height, :min_width]
        clear_array = clear_array[:min_height, :min_width]

    try:
        ssim = calculate_ssim(clear_array, defogged_image, channel_axis=-1)
        return ssim
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        return 0

def compute_ciede2000(defogged_image, clear_image_path, Xsize, Ysize, sample_step=4):
    """計算無霧圖像與清晰參考圖像之間的 CIEDE 2000 顏色差異"""
    clear_img = Image.open(clear_image_path).convert('RGB')
    if clear_img.width != Xsize or clear_img.height != Ysize:
        clear_img = clear_img.resize((Xsize, Ysize))
    clear_array = np.array(clear_img)

    if defogged_image.shape != clear_array.shape:
        min_height = min(defogged_image.shape[0], clear_array.shape[0])
        min_width = min(defogged_image.shape[1], clear_array.shape[1])
        defogged_image = defogged_image[:min_height, :min_width]
        clear_array = clear_array[:min_height, :min_width]
    
    try:
        # 將RGB轉換為LAB色彩空間
        lab_defogged = rgb2lab(defogged_image)
        lab_clear = rgb2lab(clear_array)
        
        # 計算每個像素的CIEDE 2000差異
        delta_e = deltaE_ciede2000(lab_clear, lab_defogged)
        
        # 計算平均顏色差異
        mean_delta_e = np.mean(delta_e)
        return mean_delta_e
    except Exception as e:
        print(f"Error calculating CIEDE 2000: {e}")
        return 0

def main():
    clear_dir = f"./dataset/{dataset}/clear"
    defog_dir = f"./dataset/{dataset}/result_{defog_version}"

    defog_files = sorted(glob(os.path.join(defog_dir, "*.png")))

    results = []
    avg_scores = {"PSNR": 0, "SSIM": 0, "CIEDE2000": 0}
    total = 0

    for defog_path in tqdm(defog_files, desc="Scoring"):
        base_name = os.path.splitext(os.path.basename(defog_path))[0].split('_')[0]
        clear_path = os.path.join(clear_dir, f"{base_name}_clear.png")

        if not os.path.exists(clear_path):
            print(f"⚠️ 找不到 ground truth：{clear_path}，跳過")
            continue

        defog_img = np.array(Image.open(defog_path).convert('RGB'))
        clear_img = np.array(Image.open(clear_path).convert('RGB'))
        Xsize, Ysize = defog_img.shape[1], defog_img.shape[0]

        psnr = compute_psnr(defog_img, clear_path, Xsize, Ysize)
        ssim = compute_ssim(defog_img, clear_path, Xsize, Ysize)
        ciede = compute_ciede2000(defog_img, clear_path, Xsize, Ysize, sample_step=4)

        results.append({
            "Image": base_name,
            "PSNR": psnr,
            "SSIM": ssim,
            "CIEDE2000": ciede
        })

        total += 1
        for key, val in zip(["PSNR", "SSIM", "CIEDE2000"], [psnr, ssim, ciede]):
            avg_scores[key] = (avg_scores[key] * (total - 1) + val) / total

    if total > 0:
        df = pd.DataFrame(results)
        avg_row = pd.DataFrame([{
            "Image": "AVERAGE",
            "PSNR": avg_scores["PSNR"],
            "SSIM": avg_scores["SSIM"],
            "CIEDE2000": avg_scores["CIEDE2000"]
        }])
        df = pd.concat([df, avg_row], ignore_index=True)

        os.makedirs(f"./dataset/{dataset}/report", exist_ok=True)
        csv_path = f"./dataset/{dataset}/report/score_{defog_version}.csv"
        df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"\n✅ 評分結果已儲存到：{csv_path}")
    else:
        print("⚠️ 沒有成功評分的圖片。")

if __name__ == "__main__":
    main()
