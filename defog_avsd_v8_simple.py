# defog_avsd_v8_simple.py
# V8 Simple: V5 + 三通道 t_c（藍偏自適應）— 硬體友善版
# 僅增加 +4 乘法器（channel_scale × psi, psi_c × 3 × K × min_norm per channel）
# 移除：WB 白平衡（需 frame buffer）、Gamma 校正、base_scale
import numpy as np
from scipy.ndimage import minimum_filter

""" 公式整理
H(x) = D(x)*t(x) + A*(1-t(x))
D(x) = ((H(x) - A) / t(x)) + A
=================================================================
A 從暗通道中選擇最亮的像素作為 A = (Ar, Ag, Ab) (patch = 8x8, 無降採樣)
t(x) = 1 - w * (K_H(x) / A) * (1 - S_H(x)/S_D(x))
=================================================================
V8 Simple 改進：
  t_c(x) = (temp - psi_c * 3 * K * min_norm) / temp
  where psi_c = psi * channel_scale[c]
  channel_scale 由藍色偏差自適應調整
  硬體成本：僅 +4 乘法器（vs V5）
"""

#======================================================================
# AVSD V8 Simple: Per-channel psi weights only (+4 multipliers vs V5)
# - 保留: predict_psi, A estimation, adaptive PSI boost, blue bias, per-channel t_c
# - 移除: base_scale, WB correction, gamma correction
#======================================================================


def predict_psi(image):
    """
    基於硬體霧氣評分預測最佳 PSI 值（V5 版本：無降採樣）
    回歸公式：BestPsi = 0.009308 × HW_FogScore + 0.927009
    限制範圍：0.5 ~ 1.2
    """
    gray = image[:, :, 0]
    height, width = gray.shape

    max_val = 0
    min_val = 255

    for i in range(height):
        for j in range(width):
            pixel = int(gray[i, j])
            if pixel > max_val:
                max_val = pixel
            if pixel < min_val:
                min_val = pixel

    dynamic_range = max_val - min_val

    if dynamic_range >= 240:
        fog_score = 0
    elif dynamic_range <= 100:
        fog_score = 100
    else:
        fog_score = (240 - dynamic_range) >> 1

    fog_score = max(0, min(100, fog_score))
    BestPsi = 0.009308 * fog_score + 0.927009
    BestPsi = max(0.5, min(1.2, BestPsi))

    return BestPsi


def defog_img(hazy_image, psi=1, t0=0.2, window_size=8, epsilon=1e-6,
              intensity_boost=0.35, intensity_threshold=0.45, intensity_divisor=0.35,
              a_gate_threshold=0.98, a_gate_divisor=0.02, psi_cap=1.35,
              blue_extra_max=0.40, blue_sensitivity=20.0,
              blue_R_ratio=0.4, blue_G_ratio=0.1,
              blue_bias_threshold=0.01,
              ):
    """
    V8 Simple: V5 + 自適應藍通道 per-channel t_c
    硬體成本：僅比 V5 多 4 個乘法器
    - blue_bias 檢測 A_blue 偏高
    - channel_scale = (1 - blue_extra*0.4, 1 + blue_extra*0.1, 1 + blue_extra)
    - 每通道獨立 t_c = (temp - psi_c * 3 * K * min_norm) / temp
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
    effective_bias = max(0.0, blue_bias - blue_bias_threshold)
    blue_extra = min(blue_extra_max, effective_bias * blue_sensitivity)
    channel_scale = (1.0 - blue_extra * blue_R_ratio,
                     1.0 + blue_extra * blue_G_ratio,
                     1.0 + blue_extra)

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

    defog_version = "defog_avsd_v8_simple"
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
    print(f"AVSD V8 Simple (Per-channel t_c only, +4 multipliers) Summary vs Targets")
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
