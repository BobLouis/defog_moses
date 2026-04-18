# main.py
import numpy as np
import os
import time
from PIL import Image
from glob import glob
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from skimage.color import rgb2lab, deltaE_ciede2000
from glob import glob
import pandas as pd
from tqdm import tqdm

# 根據不同的版本、跑不同的dataset 調整!!!
# from defog_proposed_atmo_section_claude import defog_img
from defog_2023 import defog_img
defog_version = "defog_2023"
# 定義要處理的所有 datasets
# datasets = ["SOTS_in", "SOTS_out", "Ohaze"]
datasets = ["SOTS_in", "SOTS_out", "Ohaze", "Ihaze"]

targets = {
    "SOTS_in":  {"PSNR": 18.8006, "SSIM": 0.7856, "CIEDE2000": 10.4843},
    "SOTS_out": {"PSNR": 22.1355, "SSIM": 0.8840, "CIEDE2000": 6.0956},
    "Ohaze":    {"PSNR": 16.7290, "SSIM": 0.5942, "CIEDE2000": 15.3479},
    "Ihaze":    {"PSNR": 0.0,     "SSIM": 0.0,    "CIEDE2000": 99.0},
}

# Per-dataset folder / filename conventions
dataset_config = {
    "SOTS_in":  {"hazy_dir": "hazy", "clear_dir": "clear", "hazy_ext": "png", "clear_ext": "png", "clear_suffix": "clear"},
    "SOTS_out": {"hazy_dir": "hazy", "clear_dir": "clear", "hazy_ext": "png", "clear_ext": "png", "clear_suffix": "clear"},
    "Ohaze":    {"hazy_dir": "hazy", "clear_dir": "clear", "hazy_ext": "png", "clear_ext": "png", "clear_suffix": "clear"},
    "Ihaze":    {"hazy_dir": "hazy", "clear_dir": "GT",    "hazy_ext": "jpg", "clear_ext": "jpg", "clear_suffix": "GT"},
}

def main(dataset):
    cfg = dataset_config[dataset]
    hazy_dir = f"./dataset/{dataset}/{cfg['hazy_dir']}"
    output_defog_dir = f"./dataset/{dataset}/result_{defog_version}"

    os.makedirs(output_defog_dir, exist_ok=True)

    hazy_files = sorted(glob(os.path.join(hazy_dir, f"*.{cfg['hazy_ext']}")))

    # 用於記錄所有的 BestPsi 值
    bestpsi_list = []
    bestpsi_records = []

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
            result = defog_img(H)
            end_time = time.time()
            diff_time = end_time - start_time

            if len(result) == 3:
                defog_output, A, BestPsi = result
            else:
                defog_output, A = result
                BestPsi = 0.0

            Image.fromarray(defog_output).save(output_defog_path)

            # 記錄 BestPsi 值
            bestpsi_list.append(BestPsi)
            bestpsi_records.append({"Image": base_name, "BestPsi": BestPsi})

            print(f"大氣光 A: {A}")
            print(f"BestPsi: {BestPsi:.6f}")
            print(f"執行時間 = {diff_time:.3f} 秒 \t {int(diff_time*1000)} 毫秒")

        except Exception as e:
            print(f"處理 {hazy_path} 時發生錯誤: {e}")

    # 計算並顯示 BestPsi 的平均值
    if bestpsi_list:
        avg_bestpsi = np.mean(bestpsi_list)
        print(f"\n{'='*60}")
        print(f"BestPsi 平均值: {avg_bestpsi:.6f}")
        print(f"總共處理 {len(bestpsi_list)} 張圖片")
        print(f"{'='*60}")

        # 儲存 BestPsi 記錄到 TXT
        os.makedirs(f"./dataset/{dataset}/report", exist_ok=True)
        txt_path = f"./dataset/{dataset}/report/bestpsi_{defog_version}.txt"

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"BestPsi 記錄\n")
            f.write(f"{'='*60}\n\n")
            for record in bestpsi_records:
                f.write(f"{record['Image']}: {record['BestPsi']:.6f}\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"平均值: {avg_bestpsi:.6f}\n")
            f.write(f"總共處理 {len(bestpsi_list)} 張圖片\n")

        print(f"BestPsi 記錄已儲存到：{txt_path}\n")


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

def score(dataset):
    cfg = dataset_config[dataset]
    clear_dir = f"./dataset/{dataset}/{cfg['clear_dir']}"
    defog_dir = f"./dataset/{dataset}/result_{defog_version}"

    defog_files = sorted(glob(os.path.join(defog_dir, "*.png")))

    results = []
    avg_scores = {"PSNR": 0, "SSIM": 0, "CIEDE2000": 0}
    total = 0

    for defog_path in tqdm(defog_files, desc=f"Scoring {dataset}"):
        base_name = os.path.splitext(os.path.basename(defog_path))[0].split('_')[0]
        # Try exact match first, then fall back to glob for datasets with
        # extra tokens in the clear filename (e.g. Ihaze: 01_indoor_GT.jpg)
        clear_path = os.path.join(clear_dir, f"{base_name}_{cfg['clear_suffix']}.{cfg['clear_ext']}")
        if not os.path.exists(clear_path):
            matches = sorted(glob(os.path.join(clear_dir, f"{base_name}_*{cfg['clear_suffix']}.{cfg['clear_ext']}")))
            if matches:
                clear_path = matches[0]

        if not os.path.exists(clear_path):
            print(f"找不到 ground truth：{clear_path}，跳過")
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
        
        return avg_scores
    else:
        print(f"⚠️ {dataset} 沒有成功評分的圖片。")
        return None

if __name__ == "__main__":
    # 儲存所有 datasets 的平均值
    all_dataset_scores = {}
    
    for dataset in datasets:
        print(f"\n{'#'*80}")
        print(f"{'#'*80}")
        print(f"############# 開始處理 Dataset: {dataset} #############")
        print(f"{'#'*80}")
        print(f"{'#'*80}\n")
        
        # 執行 main 處理霧化圖片
        main(dataset)
        
        # 執行 score 計算評分
        avg_scores = score(dataset)
        
        if avg_scores:
            all_dataset_scores[dataset] = avg_scores
    
    # ========== Summary ==========
    print(f"\n\n{'='*70}")
    print(f"{defog_version} Summary vs Targets")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} | {'PSNR':>8} ({'Tgt':>8}) | {'SSIM':>8} ({'Tgt':>8}) | {'CIEDE':>8} ({'Tgt':>8})")
    print(f"{'-'*15}-+-{'-'*19}-+-{'-'*19}-+-{'-'*19}")
    for ds in datasets:
        if ds in all_dataset_scores:
            s = all_dataset_scores[ds]
            t = targets[ds]
            p_ok = "+" if s["PSNR"] > t["PSNR"] else "-"
            s_ok = "+" if s["SSIM"] > t["SSIM"] else "-"
            c_ok = "+" if s["CIEDE2000"] < t["CIEDE2000"] else "-"
            print(f"{ds:<15} | {s['PSNR']:>7.4f}{p_ok} ({t['PSNR']:>7.4f}) | {s['SSIM']:>7.4f}{s_ok} ({t['SSIM']:>7.4f}) | {s['CIEDE2000']:>7.4f}{c_ok} ({t['CIEDE2000']:>7.4f})")
    print(f"{'='*70}")

