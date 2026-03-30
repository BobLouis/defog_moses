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
# AVSD V8: Per-channel psi weights (3-type) for better color restoration
# Based on V5 Mod + dehaze_v7_3type channel scaling concept
# Blue channel gets more defogging to suppress blue haze cast
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
              a_gate_threshold=0.98, a_gate_divisor=0.02, psi_cap=1.35,
              blue_extra_max=0.40, blue_sensitivity=20.0,
              blue_R_ratio=0.4, blue_G_ratio=0.1,
              blue_bias_threshold=0.01,
              base_scale_r=0.99, base_scale_g=1.00, base_scale_b=1.005,
              wb_sensitivity=5.0, wb_max_strength=0.5,
              gamma=0.95, gamma_psi_threshold=1.10, gamma_blue_threshold=0.02,
              ):
    """
    基於 proposed_v5 + 自適應藍通道 PSI 進行去霧處理。
    藍色偏差自適應：檢測 A_blue 偏高時增加藍通道去霧力道
    blue_bias_threshold: 最低藍色偏差門檻，低於此值不啟用通道差異
    """
    H = hazy_image.astype(np.float32)

    # ========== 使用動態 PSI ==========
    BestPsi = predict_psi(H)
    psi = BestPsi

    # ========== 計算大氣光 A ==========
    dark_channel = np.min(H, axis=2)
    dark_min = minimum_filter(dark_channel, size=window_size)
    idx = np.argmax(dark_min)
    y, x = np.unravel_index(idx, dark_min.shape)
    A = H[y, x, :].copy()

    # ========== 亮度自適應 PSI 增幅 ==========
    mean_intensity = np.mean(H) / 255.0
    a_brightness = np.mean(A) / 255.0
    intensity_factor = max(0.0, (mean_intensity - intensity_threshold) / intensity_divisor)
    a_gate = max(0.0, min(1.0, (a_brightness - a_gate_threshold) / a_gate_divisor))
    adaptive_boost = intensity_boost * intensity_factor * a_gate
    psi = min(psi_cap, psi + adaptive_boost)

    # ========== 自適應藍色偏差檢測 ==========
    a_mean = np.mean(A)
    blue_bias = max(0.0, (A[2] - a_mean) / (a_mean + epsilon))
    # 只有在藍色偏差超過門檻時才啟用通道差異
    effective_bias = max(0.0, blue_bias - blue_bias_threshold)
    blue_extra = min(blue_extra_max, effective_bias * blue_sensitivity)
    channel_scale = (base_scale_r - blue_extra * blue_R_ratio,
                     base_scale_g + blue_extra * blue_G_ratio,
                     base_scale_b + blue_extra)

    # ========== 三通道不同權重去霧 ==========
    H_norm = H / (A + epsilon)
    K = np.mean(H_norm, axis=2)
    min_norm = np.min(H_norm, axis=2)
    temp = 3 * K + 3 * min_norm

    D = np.zeros_like(H)
    for c in range(3):
        psi_c = psi * channel_scale[c]
        t_c = (temp - psi_c * 3 * K * min_norm) / (temp + epsilon)
        t_c = np.clip(t_c, t0, 1)
        D[:, :, c] = (H[:, :, c] - A[c]) / t_c + A[c]

    D = np.clip(D, 0, 255)
    # ========== 灰世界白平衡校正（僅藍偏圖像） ==========
    if blue_bias > blue_bias_threshold:
        wb_strength = min(wb_max_strength, effective_bias * wb_sensitivity)
        mean_rgb = np.mean(D, axis=(0, 1))
        gray_avg = np.mean(mean_rgb)
        for c in range(3):
            if mean_rgb[c] > epsilon:
                scale_c = gray_avg / mean_rgb[c]
                D[:, :, c] = D[:, :, c] * (1.0 + wb_strength * (scale_c - 1.0))
    D = np.clip(D, 0, 255)
    # ========== 藍偏+高霧雙閘 Gamma 校正 ==========
    if gamma != 1.0 and psi > gamma_psi_threshold and blue_bias > gamma_blue_threshold:
        gamma_strength = min(1.0, (psi - gamma_psi_threshold) / 0.15)
        blue_strength = min(1.0, (blue_bias - gamma_blue_threshold) / 0.05)
        effective_gamma = 1.0 + (gamma - 1.0) * gamma_strength * blue_strength
        D = 255.0 * np.power(D / 255.0, effective_gamma)
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

    defog_version = "defog_avsd_v8"
    datasets = ["OHaze", "SOTS_out", "SOTS_in"]

    targets = {
        "OHaze":    {"PSNR": 16.7290, "SSIM": 0.5942, "CIEDE2000": 15.3479},
        "SOTS_out": {"PSNR": 22.1355, "SSIM": 0.8840, "CIEDE2000": 6.0956},
        "SOTS_in":  {"PSNR": 18.8006, "SSIM": 0.7856, "CIEDE2000": 10.4843},
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
        hazy_dir = f"./dataset/{dataset}/hazy"
        output_defog_dir = f"./dataset/{dataset}/result_{defog_version}"
        os.makedirs(output_defog_dir, exist_ok=True)

        hazy_files = sorted(glob(os.path.join(hazy_dir, "*.png")))

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
        clear_dir = f"./dataset/{dataset}/clear"
        defog_dir = f"./dataset/{dataset}/result_{defog_version}"
        defog_files = sorted(glob(os.path.join(defog_dir, "*.png")))

        results = []
        avg_scores = {"PSNR": 0, "SSIM": 0, "CIEDE2000": 0}
        total = 0

        for defog_path in tqdm(defog_files, desc=f"Scoring {dataset}"):
            base_name = os.path.splitext(os.path.basename(defog_path))[0].split('_')[0]
            clear_path = os.path.join(clear_dir, f"{base_name}_clear.png")
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
    print(f"AVSD V8 (3-Type Channel) Summary vs Targets")
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