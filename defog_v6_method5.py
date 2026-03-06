# defog_v6_method5.py (方法五：全域/空間 PSI 自適應混合)
import numpy as np
from scipy.ndimage import minimum_filter

""" 公式整理
H(x) = D(x)*t(x) + A*(1-t(x))
D(x) = ((H(x) - A) / t(x)) + A
=================================================================
V6 Method 5:
- 2x 下採樣估計大氣光 A
- predict_psi 計算全域 PSI (如 moses 方法)
- 大氣光比較法生成空間 PSI map (如 method1)
- 根據霧均勻度混合全域/空間 PSI
  - 均勻霧 (SOTS 合成) → 偏向全域 PSI
  - 不均勻霧 (OHaze 真實) → 偏向空間 PSI
- 動態 t0 根據霧濃度調整
"""


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


def predict_psi(fog_score):
    """根據 fog_score 計算全域最佳 PSI"""
    best_psi = 0.011099 * fog_score + 0.746386
    best_psi = np.clip(best_psi, 0.7, 1.2)
    return best_psi


def compute_psi_map(H, A, psi_min=0.52, psi_max=1.38, buffer_size=8, epsilon=1e-6):
    """方法一的空間 PSI map"""
    height, width = H.shape[:2]
    psi_range = psi_max - psi_min

    diff = np.abs(H - A)
    relative_diff = diff / (A + epsilon)
    fog_density = np.min(relative_diff, axis=2) * 100
    fog_density = np.power(fog_density / 100.0, 1.05) * 100
    fog_density = np.clip(fog_density, 0, 100)

    raw_psi = psi_max - (fog_density / 100.0) * psi_range

    psi_map = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        line_buffer = np.zeros(buffer_size, dtype=np.float32)
        buffer_sum = 0.0
        buffer_count = 0
        head = 0

        for j in range(width):
            current_psi = raw_psi[i, j]

            if buffer_count >= buffer_size:
                buffer_sum -= line_buffer[head]
            else:
                buffer_count += 1

            line_buffer[head] = current_psi
            buffer_sum += current_psi
            head = (head + 1) % buffer_size

            psi_map[i, j] = np.clip(buffer_sum / buffer_count, psi_min, psi_max)

    return psi_map


def defog_img(hazy_image, psi_min=0.52, psi_max=1.38, t0=0.20, window_size=8,
              buffer_size=8, epsilon=1e-6):
    """
    方法五：全域中心 + 空間偏移 PSI
    - 用 predict_psi 決定全域最佳 PSI 中心
    - 用空間 PSI map 的偏移量加入局部變化
    - 保留空間適應性同時使用最佳全域校正
    """
    H = hazy_image.astype(np.float32)
    height, width, channels = H.shape

    # ========== 計算大氣光 A (2x 下採樣) ==========
    H_ds = H[::2, ::2, :]
    dark_channel_ds = minimum_filter(np.min(H_ds, axis=2), size=window_size)
    idx = np.argmax(dark_channel_ds)
    y, x = np.unravel_index(idx, dark_channel_ds.shape)
    A = H_ds[y, x, :].copy()

    # ========== 計算絕對霧濃度 ==========
    fog_score = estimate_fog_score(hazy_image)
    fog_ratio = fog_score / 100.0

    # ========== 全域 PSI (predict_psi 方法) ==========
    global_psi = predict_psi(fog_score)

    # ========== 空間 PSI Map (方法一) ==========
    psi_map_spatial = compute_psi_map(H, A, psi_min, psi_max, buffer_size, epsilon)

    # ========== 以全域 PSI 為中心 + 空間偏移 ==========
    # 計算空間 PSI 相對於其均值的偏移
    spatial_mean = np.mean(psi_map_spatial)
    spatial_offset = psi_map_spatial - spatial_mean

    # 空間偏移的強度：霧濃(OHaze) 需要更多空間變化, 均勻霧需要較少
    psi_cv = np.std(psi_map_spatial) / (spatial_mean + epsilon)
    # 空間權重：cv 高 → 保留更多空間變化
    spatial_strength = np.clip(psi_cv * 4.0, 0.1, 0.6)

    psi_map = global_psi + spatial_strength * spatial_offset

    # ========== 計算歸一化圖像 ==========
    H_norm = H / (A + epsilon)
    K = np.mean(H_norm, axis=2)
    min_norm = np.min(H_norm, axis=2)

    # ========== 計算傳輸圖 t ==========
    temp = 3 * K + 3 * min_norm
    t = (temp - psi_map * 3 * K * min_norm) / (temp + epsilon)

    # 動態 t0
    dynamic_t0 = t0 + (1 - fog_ratio) * 0.08
    t = np.clip(t, dynamic_t0, 1)

    # ========== 恢復無霧圖像 ==========
    t_expanded = t[:, :, np.newaxis]
    D = (H - A) / t_expanded + A
    D = np.clip(D, 0, 255).astype(np.uint8)

    return D, A, psi_map, fog_score, global_psi, spatial_strength


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

    defog_version = "defog_v6_method5"
    datasets = ["OHaze", "SOTS_out", "SOTS_in"]

    targets = {
        "OHaze": {"PSNR": 16.7290, "SSIM": 0.5942, "CIEDE2000": 15.3479},
        "SOTS_out":   {"PSNR": 22.1355, "SSIM": 0.8840, "CIEDE2000": 6.0956},
        "SOTS_in":    {"PSNR": 17.1906, "SSIM": 0.7856, "CIEDE2000": 10.4843},
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
                defog_output, A, psi_map, fog_score, global_psi, alpha = defog_img(H, buffer_size=16)
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
    print(f"Method 5 Summary vs Targets")
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
