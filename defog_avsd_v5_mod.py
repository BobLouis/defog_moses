# defog_v5_template.py (基於 proposed_v5 邏輯的模板格式)
import numpy as np
from scipy.ndimage import minimum_filter

""" 公式整理
H(x) = D(x)*t(x) + A*(1-t(x))
D(x) = ((H(x) - A) / t(x)) + A
=================================================================
A 從暗通道中選擇最亮的像素作為 A = (Ar, Ag, Ab) (patch = 8x8, 無降採樣)
t(x) = 1 - w * (K_H(x) / A) * (1 - S_H(x)/S_D(x))
=================================================================
which
S(x) 為飽和度, K(x) 為像素強度值
w = psi (動態計算，範圍 0.5~1.2)
K_H(x) = Hr(x) + Hg(x) + Hb(x) / 3
    => (K_H(x) / A) = (Hr(x)/Ar + Hg(x)/Ag + Hb(x)/Ab) / 3
    => let H_norm[] = [Hr(x)/Ar, Hg(x)/Ag, Hb(x)/Ab]
    => (K_H(x) / A) = avg(H_norm)
S_D(x) = S_H(x) * (2 - S_H(x))
S_H(x) = 1 - (min_c(H_c(x)) / K_H(x)), which c is rgb
=================================================================
    => t(x) = 1 - w * (K_H(x) / A) * (1 - 1/(2 - S_H(x)) )
簡化後：
    t(x) = (temp - psi * 3 * K * min_norm) / temp
    where temp = 3*K + 3*min_norm
"""

#======================================================================
# AVSD V5 Mod Summary vs Targets
#======================================================================
'''
======================================================================
AVSD V5 Mod Summary vs Targets
======================================================================
Dataset         |     PSNR (     Tgt) |     SSIM (     Tgt) |    CIEDE (     Tgt)
----------------+---------------------+---------------------+--------------------
OHaze           | 16.6058- (16.7290) |  0.6072+ ( 0.5942) | 15.8305- (15.3479)
SOTS_out        | 22.3576+ (22.1355) |  0.8873+ ( 0.8840) |  5.8381+ ( 6.0956)
SOTS_in         | 18.8638+ (18.8006) |  0.8028+ ( 0.7856) |  8.2583+ (10.4843)
Ihaze           | 15.9231+ ( 0.0000) |  0.7454+ ( 0.0000) | 13.7096+ (99.0000)
======================================================================
'''
#======================================================================


def predict_psi(image):
    """
    基於硬體霧氣評分預測最佳 PSI 值（V5 版本：無降採樣）
    回歸公式：BestPsi = 0.009308 × HW_FogScore + 0.927009
    限制範圍：0.5 ~ 1.2
    
    參數:
    image: 輸入圖像（RGB，np.uint8 或 float32）
    
    返回:
    BestPsi: 最佳 PSI 值（float）
    """
    # V5 硬體：直接用 full resolution 取 R channel（不降採樣）
    gray = image[:, :, 0]
    height, width = gray.shape
    
    # 單次掃描找最大最小值
    max_val = 0
    min_val = 255
    
    for i in range(height):
        for j in range(width):
            pixel = int(gray[i, j])
            if pixel > max_val:
                max_val = pixel
            if pixel < min_val:
                min_val = pixel
    
    # 計算動態範圍
    dynamic_range = max_val - min_val
    
    # 霧氣評分（用位移代替除法）
    if dynamic_range >= 240:
        fog_score = 0
    elif dynamic_range <= 100:
        fog_score = 100
    else:
        fog_score = (240 - dynamic_range) >> 1
    
    # 限制範圍
    fog_score = max(0, min(100, fog_score))
    
    # 套用回歸公式
    BestPsi = 0.009308 * fog_score + 0.927009
    
    # 限制範圍 0.5 ~ 1.2
    BestPsi = max(0.5, min(1.2, BestPsi))
    
    return BestPsi


def defog_img(hazy_image, psi=1, t0=0.2, window_size=8, epsilon=1e-6,
              intensity_boost=0.35, intensity_threshold=0.45, intensity_divisor=0.35,
              a_gate_threshold=0.98, a_gate_divisor=0.02, psi_cap=1.35):
    """
    基於 proposed_v5 方法對輸入的 hazy 圖像進行去霧處理，返回無霧圖像、大氣光和最佳 PSI。

    參數:
    hazy_image: 輸入圖像（RGB，np.uint8）
    psi: 擬合係數（會被自動計算的 BestPsi 覆蓋）
    t0: 傳輸圖的下界（預設 0.2）
    window_size: 最小濾波器窗口大小（8x8，V5 版本）
    epsilon: 防止除零的小常數
    intensity_boost: 亮度自適應 PSI 增幅係數
    intensity_threshold: 亮度門檻
    intensity_divisor: 亮度歸一化因子
    a_gate_threshold: 大氣光亮度門檻（歸一化到0-1，用於區分室內/室外）
    a_gate_divisor: 大氣光門檻歸一化因子
    psi_cap: PSI 上限

    返回:
    D: 去霧後的圖像（np.uint8）
    A: 大氣光向量（3,）
    BestPsi: 自動計算的最佳 PSI 值
    """
    # 將輸入轉換為 float 型態以便計算
    H = hazy_image.astype(np.float32)

    # ========== 使用動態 PSI（V5 版本：無降採樣）==========
    BestPsi = predict_psi(H)
    psi = BestPsi

    # ========== 計算大氣光 A（V5 版本：直接對 full resolution 做處理）==========
    # 直接對 full resolution 計算暗通道
    dark_channel = np.min(H, axis=2)  # 取 RGB 最小值
    dark_min = minimum_filter(dark_channel, size=window_size)  # 8x8 window min

    # 找出最大的 dark channel 值對應的位置
    idx = np.argmax(dark_min)
    y, x = np.unravel_index(idx, dark_min.shape)
    A = H[y, x, :].copy()  # 大氣光向量

    # ========== 亮度自適應 PSI 增幅（僅對極亮大氣光生效，區分室內場景）==========
    mean_intensity = np.mean(H) / 255.0
    a_brightness = np.mean(A) / 255.0

    # 亮度因子：圖像越亮（霧越濃）→ 增幅越大
    intensity_factor = max(0.0, (mean_intensity - intensity_threshold) / intensity_divisor)
    # 大氣光門檻：僅在 A 極亮時啟用（室內合成霧 A≈254，室外 A≈235）
    a_gate = max(0.0, min(1.0, (a_brightness - a_gate_threshold) / a_gate_divisor))

    adaptive_boost = intensity_boost * intensity_factor * a_gate
    psi = min(psi_cap, psi + adaptive_boost)

    # ========== 使用原始全解析度圖像進行去霧處理 ==========
    # 對每個通道進行歸一化（除以 A）
    H_norm = H / (A + epsilon)

    # 計算歸一化圖像的平均強度 K（每個像素的均值）
    K = np.mean(H_norm, axis=2)

    # 計算最小歸一化值
    min_norm = np.min(H_norm, axis=2)

    # 計算傳輸圖 t
    temp = 3 * K + 3 * min_norm
    t = (temp - psi * 3 * K * min_norm) / (temp + epsilon)

    # 限制傳輸圖的下界
    t = np.clip(t, t0, 1)

    # 利用傳輸圖恢復無霧圖像： D(x) = (H(x) - A) / t(x) + A
    t_expanded = t[:, :, np.newaxis]
    D = (H - A) / t_expanded + A
    D = np.clip(D, 0, 255).astype(np.uint8)

    return D, A, psi


if __name__ == "__main__":
    import os
    import time
    from PIL import Image
    from glob import glob
    from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
    from skimage.metrics import structural_similarity as calculate_ssim
    from skimage.color import rgb2lab, deltaE_ciede2000
    import pandas as pd
    from tqdm import tqdm

    defog_version = "defog_avsd_v5_mod"
    datasets = ["OHaze", "SOTS_out", "SOTS_in", "Ihaze"]

    targets = {
        "OHaze":    {"PSNR": 16.7290, "SSIM": 0.5942, "CIEDE2000": 15.3479},
        "SOTS_out": {"PSNR": 22.1355, "SSIM": 0.8840, "CIEDE2000": 6.0956},
        "SOTS_in":  {"PSNR": 18.8006, "SSIM": 0.7856, "CIEDE2000": 10.4843},
        "Ihaze":    {"PSNR": 0.0,     "SSIM": 0.0,    "CIEDE2000": 99.0},
    }

    # Per-dataset folder / filename conventions
    dataset_config = {
        "OHaze":    {"hazy_dir": "hazy", "clear_dir": "clear", "hazy_ext": "png", "clear_ext": "png", "clear_suffix": "clear"},
        "SOTS_out": {"hazy_dir": "hazy", "clear_dir": "clear", "hazy_ext": "png", "clear_ext": "png", "clear_suffix": "clear"},
        "SOTS_in":  {"hazy_dir": "hazy", "clear_dir": "clear", "hazy_ext": "png", "clear_ext": "png", "clear_suffix": "clear"},
        "Ihaze":    {"hazy_dir": "hazy", "clear_dir": "GT",    "hazy_ext": "jpg", "clear_ext": "jpg", "clear_suffix": "GT"},
    }

    def compute_psnr(defogged_image, clear_image_path, Xsize, Ysize):
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
            return calculate_psnr(clear_array, defogged_image)
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
            return 0

    def compute_ssim(defogged_image, clear_image_path, Xsize, Ysize):
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
            return calculate_ssim(clear_array, defogged_image, channel_axis=-1)
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            return 0

    def compute_ciede2000(defogged_image, clear_image_path, Xsize, Ysize, sample_step=4):
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
            lab_defogged = rgb2lab(defogged_image)
            lab_clear = rgb2lab(clear_array)
            delta_e = deltaE_ciede2000(lab_clear, lab_defogged)
            return np.mean(delta_e)
        except Exception as e:
            print(f"Error calculating CIEDE 2000: {e}")
            return 0

    def main(dataset):
        cfg = dataset_config[dataset]
        hazy_dir = f"./dataset/{dataset}/{cfg['hazy_dir']}"
        output_defog_dir = f"./dataset/{dataset}/result_{defog_version}"
        os.makedirs(output_defog_dir, exist_ok=True)

        hazy_files = sorted(glob(os.path.join(hazy_dir, f"*.{cfg['hazy_ext']}")))

        for hazy_path in hazy_files:
            full_name = os.path.splitext(os.path.basename(hazy_path))[0]
            base_name = full_name.split('_')[0]
            output_defog_path = os.path.join(output_defog_dir, f"{base_name}_{defog_version}.png")

            try:
                img = Image.open(hazy_path).convert('RGB')
                H = np.array(img)

                start_time = time.time()
                defog_output, A, BestPsi = defog_img(H)
                elapsed = time.time() - start_time

                Image.fromarray(defog_output).save(output_defog_path)

            except Exception as e:
                print(f"Error processing {hazy_path}: {e}")
                import traceback
                traceback.print_exc()

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
                continue

            defog_img_arr = np.array(Image.open(defog_path).convert('RGB'))
            Xsize, Ysize = defog_img_arr.shape[1], defog_img_arr.shape[0]

            psnr = compute_psnr(defog_img_arr, clear_path, Xsize, Ysize)
            ssim = compute_ssim(defog_img_arr, clear_path, Xsize, Ysize)
            ciede = compute_ciede2000(defog_img_arr, clear_path, Xsize, Ysize, sample_step=4)

            results.append({"Image": base_name, "PSNR": psnr, "SSIM": ssim, "CIEDE2000": ciede})
            total += 1
            for key, val in zip(["PSNR", "SSIM", "CIEDE2000"], [psnr, ssim, ciede]):
                avg_scores[key] = (avg_scores[key] * (total - 1) + val) / total

        if total > 0:
            df = pd.DataFrame(results)
            avg_row = pd.DataFrame([{"Image": "AVERAGE", **avg_scores}])
            df = pd.concat([df, avg_row], ignore_index=True)

            os.makedirs(f"./dataset/{dataset}/report", exist_ok=True)
            csv_path = f"./dataset/{dataset}/report/score_{defog_version}.csv"
            df.to_csv(csv_path, index=False, float_format="%.4f")

        return avg_scores if total > 0 else None

    # ========== Run all datasets ==========
    all_scores = {}
    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"### Processing: {dataset}")
        print(f"{'#'*60}")
        main(dataset)
        avg = score(dataset)
        if avg:
            all_scores[dataset] = avg

    # ========== Summary ==========
    print(f"\n\n{'='*70}")
    print(f"AVSD V5 Mod Summary vs Targets")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} | {'PSNR':>8} ({'Tgt':>8}) | {'SSIM':>8} ({'Tgt':>8}) | {'CIEDE':>8} ({'Tgt':>8})")
    print(f"{'-'*15}-+-{'-'*19}-+-{'-'*19}-+-{'-'*19}")
    for ds in datasets:
        if ds in all_scores:
            s = all_scores[ds]
            t = targets[ds]
            p_ok = "+" if s["PSNR"] > t["PSNR"] else "-"
            s_ok = "+" if s["SSIM"] > t["SSIM"] else "-"
            c_ok = "+" if s["CIEDE2000"] < t["CIEDE2000"] else "-"
            print(f"{ds:<15} | {s['PSNR']:>7.4f}{p_ok} ({t['PSNR']:>7.4f}) | {s['SSIM']:>7.4f}{s_ok} ({t['SSIM']:>7.4f}) | {s['CIEDE2000']:>7.4f}{c_ok} ({t['CIEDE2000']:>7.4f})")
    print(f"{'='*70}")