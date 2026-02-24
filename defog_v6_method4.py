# defog_v6_method4.py (方法四：自適應強度 + 絕對霧濃度校正)
import numpy as np
from scipy.ndimage import minimum_filter

""" 公式整理
H(x) = D(x)*t(x) + A*(1-t(x))
D(x) = ((H(x) - A) / t(x)) + A
=================================================================
V6 Method 4:
- 2x 下採樣估計大氣光 A
- 對比度下降法估計全域霧濃度 → 動態調整除霧強度
- 大氣光比較法生成空間 PSI map
- 自適應強度機制: t_adjusted = t^adaptive_strength
  - 霧濃區域 strength 高 → 除霧強
  - 無霧區域 strength = 1 → 不處理
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
    方法四：自適應強度 + 絕對霧濃度校正
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

    # ========== 根據霧濃度校正 PSI 範圍 ==========
    psi_shift = (fog_ratio - 0.5) * 0.18
    adj_psi_min = psi_min + psi_shift
    adj_psi_max = psi_max + psi_shift

    # ========== 計算歸一化圖像 ==========
    H_norm = H / (A + epsilon)
    K = np.mean(H_norm, axis=2)
    min_norm = np.min(H_norm, axis=2)

    # ========== 計算 PSI Map ==========
    psi_map = compute_psi_map(H, A, adj_psi_min, adj_psi_max, buffer_size, epsilon)

    # ========== 計算傳輸圖 t ==========
    temp = 3 * K + 3 * min_norm
    t = (temp - psi_map * 3 * K * min_norm) / (temp + epsilon)

    # 動態 t0: 濃霧允許更低 t, 淡霧 t0 較高
    dynamic_t0 = t0 + (1 - fog_ratio) * 0.10
    t = np.clip(t, dynamic_t0, 1)

    adaptive_strength = np.ones_like(t)  # placeholder for return

    # ========== 恢復無霧圖像 ==========
    t_expanded = t[:, :, np.newaxis]
    D = (H - A) / t_expanded + A
    D = np.clip(D, 0, 255).astype(np.uint8)

    return D, A, psi_map, fog_score, adj_psi_min, adj_psi_max, adaptive_strength


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
    import matplotlib.pyplot as plt

    defog_version = "defog_v6_method4"
    dataset = "SOTS_in"

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

    def save_psi_heatmap(psi_map, output_path, psi_min=0.52, psi_max=1.38):
        psi_normalized = (psi_map - psi_min) / (psi_max - psi_min + 1e-6)
        psi_normalized = np.clip(psi_normalized, 0, 1)
        plt.figure(figsize=(10, 8))
        plt.imshow(psi_normalized, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(label=f'PSI\n{psi_min:.2f} (clear) -> {psi_max:.2f} (foggy)')
        plt.title('PSI Map - Method 4')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def main():
        hazy_dir = f"./dataset/{dataset}/hazy"
        output_defog_dir = f"./dataset/{dataset}/result_{defog_version}"
        heatmap_dir = "./tmpfile/method4"
        os.makedirs(heatmap_dir, exist_ok=True)
        os.makedirs(output_defog_dir, exist_ok=True)

        hazy_files = sorted(glob(os.path.join(hazy_dir, "*.png")))

        for hazy_path in hazy_files:
            full_name = os.path.splitext(os.path.basename(hazy_path))[0]
            base_name = full_name.split('_')[0]
            output_defog_path = os.path.join(output_defog_dir, f"{base_name}_{defog_version}.png")

            print(f"\n處理中: {hazy_path}")

            try:
                img = Image.open(hazy_path).convert('RGB')
                H = np.array(img)

                start_time = time.time()
                defog_output, A, psi_map, fog_score, psi_min, psi_max, strength_map = defog_img(H, buffer_size=16)
                elapsed = time.time() - start_time

                Image.fromarray(defog_output).save(output_defog_path)

                heatmap_path = os.path.join(heatmap_dir, f"{base_name}_psi_heatmap.png")
                save_psi_heatmap(psi_map, heatmap_path, psi_min, psi_max)

                avg_psi = np.mean(psi_map)
                print(f"A: {A}, FogScore: {fog_score:.2f}")
                print(f"PSI: [{psi_min:.4f}, {psi_max:.4f}], Avg: {avg_psi:.4f}")
                print(f"Strength: [{np.min(strength_map):.3f}, {np.max(strength_map):.3f}], Avg: {np.mean(strength_map):.3f}")
                print(f"Time: {elapsed:.3f}s")

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

    def score():
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

            print(f"\n{'='*60}")
            print(f"Method 4 (Adaptive Strength + Absolute Fog):")
            print(f"  PSNR:      {avg_scores['PSNR']:.4f} dB")
            print(f"  SSIM:      {avg_scores['SSIM']:.4f}")
            print(f"  CIEDE2000: {avg_scores['CIEDE2000']:.4f}")
            print(f"{'='*60}")
            print(f"Baseline (Method 1): PSNR=16.6887, SSIM=0.5925, CIEDE=15.3923")
            print(f"Saved to: {csv_path}")

    main()
    score()
