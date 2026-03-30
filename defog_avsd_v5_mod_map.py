# defog_v5_template.py (基於 proposed_v5 邏輯的模板格式 + PSI Map)
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter, uniform_filter1d

""" 公式整理
H(x) = D(x)*t(x) + A*(1-t(x))
D(x) = ((H(x) - A) / t(x)) + A
=================================================================
A 從暗通道中選擇最亮的像素作為 A = (Ar, Ag, Ab) (patch = 8x8, 無降採樣)
t(x) = (temp - psi_map * 3 * K * min_norm) / temp
where temp = 3*K + 3*min_norm
=================================================================
PSI Map: 混合法 (Hybrid Method) - 結合大氣光比較 + 局部對比度
+ 全域霧濃度校正
"""


#  ┌──────────┬──────────┬─────────┬─────────────────┐
#  │ Dataset  │ Baseline │  Final  │      Delta      │
#  ├──────────┼──────────┼─────────┼─────────────────┤
#  │ OHaze    │ 16.6067  │ 16.6142 │ +0.008 (stable) │
#  ├──────────┼──────────┼─────────┼─────────────────┤
#  │ SOTS_out │ 22.3646  │ 21.7779 │ -0.587          │
#  ├──────────┼──────────┼─────────┼─────────────────┤
#  │ SOTS_in  │ 17.9952  │ 18.8080 │ +0.813          │
#  └──────────┴──────────┴─────────┴─────────────────┘


def estimate_fog_score(image):
    """對比度下降法估計霧濃度 (0~100)"""
    if image.dtype == np.float32:
        img = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img = image

    gray = img[:, :, 0]
    height, width = gray.shape
    total_pixels = height * width

    max_val = np.max(gray)
    min_val = np.min(gray)
    avg_intensity = np.mean(gray)
    dynamic_range = max_val - min_val
    avg_deviation = np.mean(np.abs(gray.astype(np.float32) - avg_intensity))

    diff_h = np.abs(gray[:, :-1].astype(np.int16) - gray[:, 1:].astype(np.int16))
    diff_v = np.abs(gray[:-1, :].astype(np.int16) - gray[1:, :].astype(np.int16))
    avg_local_diff = (np.sum(diff_h) + np.sum(diff_v)) / (2 * total_pixels)

    if dynamic_range >= 240:
        fog_score_range = 0
    elif dynamic_range <= 100:
        fog_score_range = 100
    else:
        fog_score_range = 100 - ((dynamic_range - 100) / 140.0) * 100

    if avg_deviation >= 60:
        fog_score_deviation = 0
    elif avg_deviation <= 20:
        fog_score_deviation = 100
    else:
        fog_score_deviation = 100 - ((avg_deviation - 20) / 40.0) * 100

    if avg_local_diff >= 10:
        fog_score_edge = 0
    elif avg_local_diff <= 1:
        fog_score_edge = 100
    else:
        fog_score_edge = 100 - ((avg_local_diff - 1) / 9.0) * 100

    fog_score = (fog_score_range * 2 + fog_score_deviation + fog_score_edge) / 4
    fog_score = np.clip(fog_score, 0, 100)
    return fog_score


def predict_global_psi_v5(image):
    """V5 原始全域 PSI 預測 (使用 R channel dynamic range)"""
    gray = image[:, :, 0]
    if gray.dtype == np.float32:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    max_val = int(np.max(gray))
    min_val = int(np.min(gray))
    dynamic_range = max_val - min_val
    if dynamic_range >= 240:
        fog_score = 0
    elif dynamic_range <= 100:
        fog_score = 100
    else:
        fog_score = (240 - dynamic_range) >> 1
    fog_score = max(0, min(100, fog_score))
    best_psi = 0.009308 * fog_score + 0.927009
    best_psi = max(0.5, min(1.2, best_psi))
    return best_psi, fog_score


def compute_psi_map(H, A, psi_min=0.52, psi_max=1.25, buffer_size=16, epsilon=1e-6,
                    alpha=0.55, contrast_threshold=55, brightness_threshold=190):
    """
    混合法 PSI Map: 大氣光比較 + 局部對比度
    """
    height, width = H.shape[:2]
    psi_range = psi_max - psi_min

    # 大氣光霧分數
    diff = np.abs(H - A)
    relative_diff = diff / (A + epsilon)
    fog_density_atm = np.mean(relative_diff, axis=2) * 100
    fog_density_atm = np.clip(fog_density_atm, 0, 100)
    fog_score_atm = 100 - fog_density_atm

    # 平滑
    fog_score_atm_smooth = uniform_filter1d(fog_score_atm, size=buffer_size, axis=1, mode='nearest')

    # 灰階與局部對比度
    gray = H[:, :, 0]
    local_max = maximum_filter(gray, size=(1, buffer_size))
    local_min = minimum_filter(gray, size=(1, buffer_size))
    dynamic_range = local_max - local_min

    fog_score_contrast = np.where(
        dynamic_range >= 200, 0,
        np.where(dynamic_range <= 20, 100,
                 (200 - dynamic_range) / 1.8)
    )
    fog_score_contrast = np.clip(fog_score_contrast, 0, 100)

    pixel_brightness = np.mean(H, axis=2)

    # 條件混合
    cond_fog = (pixel_brightness > brightness_threshold) & (dynamic_range < contrast_threshold)
    cond_clear = (pixel_brightness < brightness_threshold * 0.6) & (dynamic_range > contrast_threshold)

    fog_score_final = np.where(
        cond_fog,
        np.maximum(fog_score_atm_smooth, fog_score_contrast) * 1.1,
        np.where(
            cond_clear,
            np.minimum(fog_score_atm_smooth, fog_score_contrast) * 0.8,
            alpha * fog_score_atm_smooth + (1 - alpha) * fog_score_contrast
        )
    )
    fog_score_final = np.clip(fog_score_final, 0, 100)

    psi_map = psi_min + (fog_score_final / 100.0) * psi_range
    psi_map = np.clip(psi_map, psi_min, psi_max).astype(np.float32)

    return psi_map


def defog_img(hazy_image, t0=0.2, window_size=8, buffer_size=16, epsilon=1e-6,
              intensity_boost=0.26, intensity_threshold=0.45, intensity_divisor=0.35,
              psi_cap=1.35, spatial_weight=0.05):
    """
    V5 + PSI Map: V5 global PSI + intensity-adaptive boost + spatial refinement
    - V5 global PSI as base (0.93~1.2 based on R channel dynamic range)
    - Intensity-based adaptive boost: brighter images (more fog) get higher PSI
    - Spatial PSI map (method3 hybrid) for per-pixel fine-tuning
    """
    H = hazy_image.astype(np.float32)
    height, width, channels = H.shape

    # ========== 計算大氣光 A（V5：full resolution）==========
    dark_channel = np.min(H, axis=2)
    dark_min = minimum_filter(dark_channel, size=window_size)
    idx = np.argmax(dark_min)
    y, x = np.unravel_index(idx, dark_min.shape)
    A = H[y, x, :].copy()

    # ========== V5 global PSI ==========
    base_psi, hw_fog_score = predict_global_psi_v5(hazy_image)

    # ========== Intensity-adaptive boost ==========
    mean_intensity = np.mean(H) / 255.0
    boost_factor = max(0, (mean_intensity - intensity_threshold) / intensity_divisor)
    adaptive_boost = intensity_boost * boost_factor
    global_psi = min(psi_cap, base_psi + adaptive_boost)

    # ========== 空間 PSI Map (method3 hybrid) ==========
    psi_map_spatial = compute_psi_map(H, A, psi_min=0.5, psi_max=psi_cap,
                                       buffer_size=buffer_size, epsilon=epsilon)

    # ========== Combine: global PSI + spatial offset ==========
    spatial_mean = np.mean(psi_map_spatial)
    spatial_offset = psi_map_spatial - spatial_mean
    psi_map = global_psi + spatial_weight * spatial_offset
    psi_map = np.clip(psi_map, 0.5, psi_cap)

    # ========== 計算歸一化圖像 ==========
    H_norm = H / (A + epsilon)
    K = np.mean(H_norm, axis=2)
    min_norm = np.min(H_norm, axis=2)

    # ========== 計算傳輸圖 t ==========
    temp = 3 * K + 3 * min_norm
    t = (temp - psi_map * 3 * K * min_norm) / (temp + epsilon)
    t = np.clip(t, t0, 1)

    # ========== 恢復無霧圖像 ==========
    t_expanded = t[:, :, np.newaxis]
    D = (H - A) / t_expanded + A
    D = np.clip(D, 0, 255).astype(np.uint8)

    return D, A, psi_map


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

    defog_version = "defog_avsd_v5_mod_map"
    datasets = ["OHaze", "SOTS_out", "SOTS_in"]

    targets = {
        "OHaze":    {"PSNR": 16.7290, "SSIM": 0.5942, "CIEDE2000": 15.3479},
        "SOTS_out": {"PSNR": 22.3655, "SSIM": 0.8840, "CIEDE2000": 6.0956},
        "SOTS_in":  {"PSNR": 18.7906, "SSIM": 0.7856, "CIEDE2000": 10.4843},
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
    print(f"AVSD V5 Mod Map Summary vs Targets")
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