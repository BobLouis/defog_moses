# defog_v6_method1.py (使用方法一：大氣光比較法)
import numpy as np
from scipy.ndimage import minimum_filter

""" 公式整理
H(x) = D(x)*t(x) + A*(1-t(x))
D(x) = ((H(x) - A) / t(x)) + A
=================================================================
A 從暗通道中選擇最亮的像素作為 A = (Ar, Ag, Ab) (patch = 8x8, 無降採樣)
t(x) = (temp - psi * 3 * K * min_norm) / temp
where temp = 3*K + 3*min_norm
=================================================================
V6 Method 1: 大氣光比較法 (Atmospheric Light Comparison)
- 計算每個像素與大氣光 A 的相對距離
- 距離小 = 霧濃 = PSI 高
- 距離大 = 無霧 = PSI 低
- 使用 line buffer 平滑
"""


def compute_psi_map_method1(H, A, psi_min=0.52, psi_max=1.38, buffer_size=8, epsilon=1e-6):
    """
    方法一：大氣光比較法 (Atmospheric Light Comparison)

    原理：
    - 霧氣會使像素值趨近大氣光 A
    - 計算每個像素與 A 的相對距離: |pixel - A| / A
    - 距離小 = 霧濃 = PSI 高
    - 距離大 = 無霧 = PSI 低

    參數:
        H: 輸入圖像 (float32, shape: height x width x 3)
        A: 大氣光向量 (shape: 3,)
        psi_min: PSI 最小值
        psi_max: PSI 最大值
        buffer_size: line buffer 大小
        epsilon: 防除零

    返回:
        psi_map: PSI 分布圖 (shape: height x width)
    """
    height, width = H.shape[:2]
    psi_range = psi_max - psi_min

    # 計算霧濃度: 0=接近A(霧濃), 100=遠離A(無霧)
    # 使用 min 通道計算霧濃度 (暗通道原理)
    diff = np.abs(H - A)
    relative_diff = diff / (A + epsilon)
    fog_density = np.min(relative_diff, axis=2) * 100
    # 輕微的 power 變換
    fog_density = np.power(fog_density / 100.0, 1.05) * 100
    fog_density = np.clip(fog_density, 0, 100)

    # 計算 raw PSI: 霧濃 → PSI 高
    raw_psi = psi_max - (fog_density / 100.0) * psi_range

    # Line buffer 平滑
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


def defog_img(hazy_image, psi_min=0.52, psi_max=1.38, t0=0.25, window_size=8, buffer_size=8, epsilon=1e-6):
    """
    基於 proposed_v5 方法 + 動態 PSI 調整 (方法一：大氣光比較法)

    參數:
    hazy_image: 輸入圖像（RGB，np.uint8）
    psi_min: PSI 最小值 (無霧時)
    psi_max: PSI 最大值 (濃霧時)
    t0: 傳輸圖的下界（預設 0.25）
    window_size: 最小濾波器窗口大小（8x8）
    buffer_size: sliding window 大小 (8 or 16)
    epsilon: 防止除零的小常數

    返回:
    D: 去霧後的圖像（np.uint8）
    A: 大氣光向量（3,）
    psi_map: PSI 分布圖
    """
    # 將輸入轉換為 float 型態以便計算
    H = hazy_image.astype(np.float32)
    height, width, channels = H.shape

    # ========== 計算大氣光 A ==========
    dark_channel = np.min(H, axis=2)
    dark_min = minimum_filter(dark_channel, size=window_size)
    idx = np.argmax(dark_min)
    y, x = np.unravel_index(idx, dark_min.shape)
    A = H[y, x, :].copy()

    # ========== 計算歸一化圖像 ==========
    H_norm = H / (A + epsilon)
    K = np.mean(H_norm, axis=2)
    min_norm = np.min(H_norm, axis=2)

    # ========== 計算 PSI Map (方法一：大氣光比較法) ==========
    psi_map = compute_psi_map_method1(H, A, psi_min, psi_max, buffer_size, epsilon)

    # ========== 計算傳輸圖 t (使用動態 PSI) ==========
    temp = 3 * K + 3 * min_norm
    t = (temp - psi_map * 3 * K * min_norm) / (temp + epsilon)

    # 限制傳輸圖的下界
    t = np.clip(t, t0, 1)

    # ========== 利用傳輸圖恢復無霧圖像 ==========
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
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # ========== 設定版本與 dataset ==========
    defog_version = "defog_v6_method1"
    # dataset = "OHaze"
    dataset = "SOTS_out"

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

    def save_psi_heatmap(psi_map, output_path, psi_min=0.52, psi_max=1.38):
        """
        將 PSI map 保存為 heat map 圖像
        PSI 高 (霧濃) -> 紅色/暖色
        PSI 低 (無霧) -> 藍色/冷色
        """
        # 正規化 PSI 到 0-1 範圍
        psi_normalized = (psi_map - psi_min) / (psi_max - psi_min)
        psi_normalized = np.clip(psi_normalized, 0, 1)

        # 使用 jet colormap (藍色=低, 紅色=高)
        # PSI 高 = 霧濃 = 紅色
        plt.figure(figsize=(10, 8))
        plt.imshow(psi_normalized, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(label=f'PSI (Fog Density)\n{psi_min:.2f} (clear) → {psi_max:.2f} (foggy)')
        plt.title('PSI Map - Fog Density Heatmap (Method 1)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def main():
        hazy_dir = f"./dataset/{dataset}/hazy"
        output_defog_dir = f"./dataset/{dataset}/result_{defog_version}"

        # PSI heatmap 輸出資料夾
        heatmap_dir = "./tmpfile/method1"
        os.makedirs(heatmap_dir, exist_ok=True)

        os.makedirs(output_defog_dir, exist_ok=True)

        hazy_files = sorted(glob(os.path.join(hazy_dir, "*.png")))

        # 用於記錄所有的 PSI 值
        psi_list = []
        psi_records = []

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
                defog_output, A, psi_map = defog_img(H, buffer_size=16)
                end_time = time.time()
                diff_time = end_time - start_time

                Image.fromarray(defog_output).save(output_defog_path)

                # 保存 PSI heatmap
                heatmap_path = os.path.join(heatmap_dir, f"{base_name}_psi_heatmap.png")
                save_psi_heatmap(psi_map, heatmap_path)
                print(f"PSI Heatmap 已保存: {heatmap_path}")

                # 記錄 PSI 平均值
                avg_psi = np.mean(psi_map)
                psi_list.append(avg_psi)
                psi_records.append({"Image": base_name, "AvgPsi": avg_psi})

                print(f"大氣光 A: {A}")
                print(f"平均 PSI: {avg_psi:.6f}")
                print(f"PSI 範圍: {psi_map.min():.4f} ~ {psi_map.max():.4f}")
                print(f"執行時間 = {diff_time:.3f} 秒 \t {int(diff_time*1000)} 毫秒")

            except Exception as e:
                print(f"處理 {hazy_path} 時發生錯誤: {e}")

        # 計算並顯示 PSI 的平均值
        if psi_list:
            avg_psi_all = np.mean(psi_list)
            print(f"\n{'='*60}")
            print(f"PSI 平均值: {avg_psi_all:.6f}")
            print(f"總共處理 {len(psi_list)} 張圖片")
            print(f"{'='*60}")

            # 儲存 PSI 記錄到 TXT
            os.makedirs(f"./dataset/{dataset}/report", exist_ok=True)
            txt_path = f"./dataset/{dataset}/report/psi_{defog_version}.txt"

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"PSI 記錄 (方法一：大氣光比較法)\n")
                f.write(f"{'='*60}\n\n")
                for record in psi_records:
                    f.write(f"{record['Image']}: {record['AvgPsi']:.6f}\n")
                f.write(f"\n{'='*60}\n")
                f.write(f"平均值: {avg_psi_all:.6f}\n")
                f.write(f"總共處理 {len(psi_list)} 張圖片\n")

            print(f"PSI 記錄已儲存到：{txt_path}\n")

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
                print(f"⚠️ 找不到 ground truth：{clear_path}，跳過")
                continue

            defog_img_arr = np.array(Image.open(defog_path).convert('RGB'))
            clear_img = np.array(Image.open(clear_path).convert('RGB'))
            Xsize, Ysize = defog_img_arr.shape[1], defog_img_arr.shape[0]

            psnr = compute_psnr(defog_img_arr, clear_path, Xsize, Ysize)
            ssim = compute_ssim(defog_img_arr, clear_path, Xsize, Ysize)
            ciede = compute_ciede2000(defog_img_arr, clear_path, Xsize, Ysize, sample_step=4)

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

            print(f"\n{'='*60}")
            print(f"方法一：大氣光比較法")
            print(f"平均 PSNR:      {avg_scores['PSNR']:.4f} dB")
            print(f"平均 SSIM:      {avg_scores['SSIM']:.4f}")
            print(f"平均 CIEDE2000: {avg_scores['CIEDE2000']:.4f}")
            print(f"{'='*60}")
            print(f"\n✅ 評分結果已儲存到：{csv_path}")
        else:
            print("⚠️ 沒有成功評分的圖片。")

    # ========== 執行主程序 ==========
    main()
    score()
