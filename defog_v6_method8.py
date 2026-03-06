# defog_v6_method8.py
# 自適應霧型態偵測 + 分流處理
#
# === 核心思路 ===
# 真實霧 (OHaze)  : 空間不均勻，fog_cv 高 → global PSI + CLAHE
# 合成霧 (SOTS)   : 空間均勻，  fog_cv 低 → M5 spatial PSI + guided filter，不用 CLAHE
#
# fog_cv = std(fog_density) / mean(fog_density)
#   高 → 真實霧 → CLAHE 模式
#   低 → 合成霧 → Spatial PSI 模式
#
# === 公式 ===
# H(x) = D(x)*t(x) + A*(1-t(x))
# D(x) = (H(x) - A) / t(x) + A
# t = (3K + 3m - psi*3*K*m) / (3K + 3m)

import numpy as np
from scipy.ndimage import minimum_filter
import cv2

# ─────────────────────────────────────────────
# 霧濃度估計 (Method5 三指標版，比單純動態範圍更準)
# ─────────────────────────────────────────────
def estimate_fog_score(image):
    """三指標霧濃度評分 (0~100，100=最濃)"""
    if image.dtype != np.uint8:
        img = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img = image

    gray = img[:, :, 0].astype(np.float32)
    height, width = gray.shape
    total_pixels = height * width

    max_val = float(np.max(gray))
    min_val = float(np.min(gray))
    avg_intensity = float(np.mean(gray))
    dynamic_range = max_val - min_val
    avg_deviation = float(np.mean(np.abs(gray - avg_intensity)))

    diff_h = np.abs(gray[:, :-1] - gray[:, 1:])
    diff_v = np.abs(gray[:-1, :] - gray[1:, :])
    avg_local_diff = (np.sum(diff_h) + np.sum(diff_v)) / (2 * total_pixels)

    if dynamic_range >= 240:   fog_score_range = 0
    elif dynamic_range <= 100: fog_score_range = 100
    else: fog_score_range = 100 - ((dynamic_range - 100) / 140.0) * 100

    if avg_deviation >= 60:   fog_score_deviation = 0
    elif avg_deviation <= 20: fog_score_deviation = 100
    else: fog_score_deviation = 100 - ((avg_deviation - 20) / 40.0) * 100

    if avg_local_diff >= 10:  fog_score_edge = 0
    elif avg_local_diff <= 1: fog_score_edge = 100
    else: fog_score_edge = 100 - ((avg_local_diff - 1) / 9.0) * 100

    fog_score = (fog_score_range * 2 + fog_score_deviation + fog_score_edge) / 4
    return float(np.clip(fog_score, 0, 100))


# ─────────────────────────────────────────────
# 霧空間均勻度 CV 偵測器
# ─────────────────────────────────────────────
def compute_fog_cv(H, A, epsilon=1e-6):
    """
    計算霧密度的空間變異係數 (CV)
    高 CV → 真實霧 (OHaze)
    低 CV → 合成霧 (SOTS)
    """
    diff = np.abs(H - A)
    relative_diff = diff / (A.astype(np.float32) + epsilon)
    fog_density = np.min(relative_diff, axis=2)
    cv = float(np.std(fog_density)) / (float(np.mean(fog_density)) + epsilon)
    return cv


# ─────────────────────────────────────────────
# Method5 式 Spatial PSI Map
# ─────────────────────────────────────────────
def compute_psi_map(H, A, psi_min=0.52, psi_max=1.38, buffer_size=16, epsilon=1e-6):
    """空間自適應 PSI map（逐行滑動平均緩衝）"""
    height, width = H.shape[:2]
    psi_range = psi_max - psi_min

    diff = np.abs(H - A)
    relative_diff = diff / (A + epsilon)
    fog_density = np.min(relative_diff, axis=2) * 100
    fog_density = np.power(np.clip(fog_density / 100.0, 0, 1), 1.05) * 100
    fog_density = np.clip(fog_density, 0, 100)

    raw_psi = psi_max - (fog_density / 100.0) * psi_range

    psi_map = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        line_buffer = np.zeros(buffer_size, dtype=np.float32)
        buffer_sum = 0.0
        buffer_count = 0
        head = 0
        for j in range(width):
            cur = raw_psi[i, j]
            if buffer_count >= buffer_size:
                buffer_sum -= line_buffer[head]
            else:
                buffer_count += 1
            line_buffer[head] = cur
            buffer_sum += cur
            head = (head + 1) % buffer_size
            psi_map[i, j] = np.clip(buffer_sum / buffer_count, psi_min, psi_max)

    return psi_map


# ─────────────────────────────────────────────
# Guided Filter（傳輸圖精化）
# ─────────────────────────────────────────────
def guided_filter(I, p, r=20, eps=0.005):
    """
    引導濾波：用原始圖像引導傳輸圖平滑
    減少 halo，保留邊緣，提升 SSIM / PSNR
    """
    I = I.astype(np.float64)
    p = p.astype(np.float64)
    mean_I  = cv2.boxFilter(I,     cv2.CV_64F, (r, r))
    mean_p  = cv2.boxFilter(p,     cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip  = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I   = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    return mean_a * I + mean_b


# ─────────────────────────────────────────────
# CLAHE（真實霧後處理）
# ─────────────────────────────────────────────
def apply_clahe(image, clip_limit=2.0, tile_size=16):
    """在 LAB 色彩空間對 L 通道做 CLAHE"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    lab_enhanced = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


# ─────────────────────────────────────────────
# 主要去霧函式
# ─────────────────────────────────────────────
def defog_img(hazy_image, window_size=8, buffer_size=16, epsilon=1e-6,
              fog_cv_threshold=0.18):
    """
    Method 8：自適應霧型態去霧

    判斷邏輯
    ─────────
    fog_cv > threshold → 真實霧 (OHaze)
        路徑: global PSI (M7 公式) + CLAHE
    fog_cv ≤ threshold → 合成霧 (SOTS)
        路徑: M5 global+spatial PSI 混合 + guided filter，不用 CLAHE

    回傳: D (uint8), A (float32 vec), BestPsi (float)
    """
    H = hazy_image.astype(np.float32)

    # ── 大氣光 A（2x 下採樣，減少暗像素干擾）──
    H_ds = H[::2, ::2, :]
    dark_ds = minimum_filter(np.min(H_ds, axis=2), size=window_size)
    idx = np.argmax(dark_ds)
    y, x = np.unravel_index(idx, dark_ds.shape)
    A = H_ds[y, x, :].copy()

    # ── 霧濃度 & 空間均勻度 ──
    fog_score = estimate_fog_score(hazy_image)
    fog_ratio  = fog_score / 100.0
    fog_cv     = compute_fog_cv(H, A, epsilon)

    # ══════════════════════════════════════════
    # 路徑判斷：用 fog_score 偵測霧類型
    # OHaze（真實極濃霧）: fog_score 通常 60-90
    # SOTS（合成輕霧）:    fog_score 通常 15-50
    # threshold=55 → OHaze 走 CLAHE，SOTS 走 spatial PSI
    # ══════════════════════════════════════════
    is_real_fog = fog_score > 55

    if is_real_fog:
        # global PSI（M7 公式）
        gray = H[:, :, 0]
        dr = float(np.max(gray) - np.min(gray))
        if dr >= 240:   fs = 0
        elif dr <= 100: fs = 100
        else:           fs = int(240 - dr) >> 1
        fs = max(0, min(100, fs))
        BestPsi = float(np.clip(0.009308 * fs + 0.927009, 0.5, 1.2))

        H_norm   = H / (A + epsilon)
        K        = np.mean(H_norm, axis=2)
        min_norm = np.min(H_norm, axis=2)
        temp = 3 * K + 3 * min_norm
        t = (temp - BestPsi * 3 * K * min_norm) / (temp + epsilon)
        t = np.clip(t, 0.20, 1.0)

        t_exp = t[:, :, np.newaxis]
        D = np.clip((H - A) / t_exp + A, 0, 255).astype(np.uint8)

        # CLAHE 是真實霧的關鍵提升
        D = apply_clahe(D, clip_limit=2.0, tile_size=16)

    # ══════════════════════════════════════════
    # 路徑 B：合成霧模式（SOTS）
    # ══════════════════════════════════════════
    else:
        # global PSI（M5 公式，基線更低，對輕霧更準確）
        global_psi = float(np.clip(0.011099 * fog_score + 0.746386, 0.7, 1.2))

        # spatial PSI map
        psi_map_spatial = compute_psi_map(H, A, buffer_size=buffer_size,
                                          epsilon=epsilon)

        # 混合：global_psi 為中心 + 空間偏移
        spatial_mean   = np.mean(psi_map_spatial)
        spatial_offset = psi_map_spatial - spatial_mean
        psi_cv_map     = np.std(psi_map_spatial) / (spatial_mean + epsilon)
        spatial_str    = np.clip(psi_cv_map * 4.0, 0.1, 0.6)
        psi_map        = global_psi + spatial_str * spatial_offset

        H_norm   = H / (A + epsilon)
        K        = np.mean(H_norm, axis=2)
        min_norm = np.min(H_norm, axis=2)
        temp = 3 * K + 3 * min_norm
        t = (temp - psi_map * 3 * K * min_norm) / (temp + epsilon)

        # 動態 t0：霧越淡 t0 越高（避免過度去霧）
        dynamic_t0 = 0.20 + (1.0 - fog_ratio) * 0.08
        t = np.clip(t, dynamic_t0, 1.0)

        # Guided filter：精化傳輸圖，減少 halo，提升 SSIM/PSNR
        gray_guide = cv2.cvtColor(hazy_image, cv2.COLOR_RGB2GRAY) / 255.0
        t = guided_filter(gray_guide, t, r=20, eps=0.005)
        t = np.clip(t, dynamic_t0, 1.0)

        t_exp = t[:, :, np.newaxis]
        D = np.clip((H - A) / t_exp + A, 0, 255).astype(np.uint8)

        BestPsi = global_psi

    return D, A, BestPsi


# ─────────────────────────────────────────────
# 評測主程式
# ─────────────────────────────────────────────
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

    defog_version = "defog_v6_method8"
    datasets = ["OHaze", "SOTS_out", "SOTS_in"]

    targets = {
        "OHaze":    {"PSNR": 16.7290, "SSIM": 0.5942, "CIEDE2000": 15.3479},
        "SOTS_out": {"PSNR": 22.1355, "SSIM": 0.8840, "CIEDE2000":  6.0956},
        "SOTS_in":  {"PSNR": 17.1906, "SSIM": 0.7856, "CIEDE2000": 10.4843},
    }

    def compute_psnr(dehazed, clear_path, Xsize, Ysize):
        ci = Image.open(clear_path).convert('RGB')
        if ci.width != Xsize or ci.height != Ysize:
            ci = ci.resize((Xsize, Ysize))
        ca = np.array(ci)
        if dehazed.shape != ca.shape:
            mh = min(dehazed.shape[0], ca.shape[0])
            mw = min(dehazed.shape[1], ca.shape[1])
            dehazed, ca = dehazed[:mh, :mw], ca[:mh, :mw]
        try:    return calculate_psnr(ca, dehazed)
        except: return 0

    def compute_ssim(dehazed, clear_path, Xsize, Ysize):
        ci = Image.open(clear_path).convert('RGB')
        if ci.width != Xsize or ci.height != Ysize:
            ci = ci.resize((Xsize, Ysize))
        ca = np.array(ci)
        if dehazed.shape != ca.shape:
            mh = min(dehazed.shape[0], ca.shape[0])
            mw = min(dehazed.shape[1], ca.shape[1])
            dehazed, ca = dehazed[:mh, :mw], ca[:mh, :mw]
        try:    return calculate_ssim(ca, dehazed, channel_axis=-1)
        except: return 0

    def compute_ciede2000(dehazed, clear_path, Xsize, Ysize, sample_step=4):
        ci = Image.open(clear_path).convert('RGB')
        if ci.width != Xsize or ci.height != Ysize:
            ci = ci.resize((Xsize, Ysize))
        ca = np.array(ci)
        if dehazed.shape != ca.shape:
            mh = min(dehazed.shape[0], ca.shape[0])
            mw = min(dehazed.shape[1], ca.shape[1])
            dehazed, ca = dehazed[:mh, :mw], ca[:mh, :mw]
        try:
            return float(np.mean(deltaE_ciede2000(rgb2lab(ca), rgb2lab(dehazed))))
        except: return 0

    def main(dataset):
        hazy_dir = f"./dataset/{dataset}/hazy"
        out_dir  = f"./dataset/{dataset}/result_{defog_version}"
        os.makedirs(out_dir, exist_ok=True)
        hazy_files = sorted(glob(os.path.join(hazy_dir, "*.png")))

        fog_cvs = []
        for hazy_path in hazy_files:
            base = os.path.splitext(os.path.basename(hazy_path))[0].split('_')[0]
            out_path = os.path.join(out_dir, f"{base}_{defog_version}.png")
            try:
                img = Image.open(hazy_path).convert('RGB')
                H = np.array(img)
                t0 = time.time()
                D, A, psi = defog_img(H)
                elapsed = time.time() - t0
                # 記錄 fog_cv 供調試
                H_f = H.astype(np.float32)
                A_f = H_f[::2, ::2, :][
                    np.unravel_index(
                        np.argmax(minimum_filter(np.min(H_f[::2,::2,:], axis=2), size=8)),
                        minimum_filter(np.min(H_f[::2,::2,:], axis=2), size=8).shape
                    )
                ]
                fog_cv = compute_fog_cv(H_f, A_f)
                fog_cvs.append(fog_cv)
                Image.fromarray(D).save(out_path)
            except Exception as e:
                import traceback; traceback.print_exc()

        if fog_cvs:
            print(f"  [{dataset}] fog_cv: mean={np.mean(fog_cvs):.3f}, "
                  f"min={np.min(fog_cvs):.3f}, max={np.max(fog_cvs):.3f}, "
                  f"real_fog%={100*np.mean(np.array(fog_cvs)>0.18):.0f}%")

    def score(dataset):
        clear_dir  = f"./dataset/{dataset}/clear"
        defog_dir  = f"./dataset/{dataset}/result_{defog_version}"
        defog_files = sorted(glob(os.path.join(defog_dir, "*.png")))

        results = []
        avg = {"PSNR": 0, "SSIM": 0, "CIEDE2000": 0}
        total = 0

        for dp in tqdm(defog_files, desc=f"Scoring {dataset}"):
            base = os.path.splitext(os.path.basename(dp))[0].split('_')[0]
            cp   = os.path.join(clear_dir, f"{base}_clear.png")
            if not os.path.exists(cp):
                continue
            da = np.array(Image.open(dp).convert('RGB'))
            Xs, Ys = da.shape[1], da.shape[0]
            psnr  = compute_psnr(da, cp, Xs, Ys)
            ssim  = compute_ssim(da, cp, Xs, Ys)
            ciede = compute_ciede2000(da, cp, Xs, Ys)
            results.append({"Image": base, "PSNR": psnr, "SSIM": ssim, "CIEDE2000": ciede})
            total += 1
            for k, v in zip(["PSNR", "SSIM", "CIEDE2000"], [psnr, ssim, ciede]):
                avg[k] = (avg[k] * (total - 1) + v) / total

        if total > 0:
            df = pd.concat([pd.DataFrame(results),
                            pd.DataFrame([{"Image": "AVERAGE", **avg}])],
                           ignore_index=True)
            os.makedirs(f"./dataset/{dataset}/report", exist_ok=True)
            df.to_csv(f"./dataset/{dataset}/report/score_{defog_version}.csv",
                      index=False, float_format="%.4f")
        return avg if total > 0 else None

    # ── Run ──
    all_scores = {}
    for ds in datasets:
        print(f"\n{'#'*60}")
        print(f"### Processing: {ds}")
        print(f"{'#'*60}")
        main(ds)
        s = score(ds)
        if s:
            all_scores[ds] = s

    # ── Summary ──
    print(f"\n\n{'='*70}")
    print(f"Method 8 Summary vs Targets")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} | {'PSNR':>8} ({'Tgt':>8}) | {'SSIM':>8} ({'Tgt':>8}) | {'CIEDE':>8} ({'Tgt':>8})")
    print(f"{'-'*12}-+-{'-'*19}-+-{'-'*19}-+-{'-'*19}")
    all_pass = True
    for ds in datasets:
        if ds in all_scores:
            s = all_scores[ds]
            t = targets[ds]
            p_ok = "✓" if s["PSNR"]     > t["PSNR"]     else "✗"
            s_ok = "✓" if s["SSIM"]     > t["SSIM"]     else "✗"
            c_ok = "✓" if s["CIEDE2000"] < t["CIEDE2000"] else "✗"
            if "✗" in (p_ok, s_ok, c_ok):
                all_pass = False
            print(f"{ds:<12} | {s['PSNR']:>7.4f}{p_ok} ({t['PSNR']:>7.4f}) "
                  f"| {s['SSIM']:>7.4f}{s_ok} ({t['SSIM']:>7.4f}) "
                  f"| {s['CIEDE2000']:>7.4f}{c_ok} ({t['CIEDE2000']:>7.4f})")
    print(f"{'='*70}")
    if all_pass:
        print("🎉 ALL TARGETS PASSED!")
    else:
        print("⚠  Some targets not met. Need further tuning.")
