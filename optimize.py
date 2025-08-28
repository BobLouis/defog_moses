# gridsearch_best_psi_to_version.py
import os
import time
import math
from glob import glob

import numpy as np
from PIL import Image

# 嘗試兩個來源，對應你現有 defog_img 的位置
try:
    from defog_proposed import defog_img
except ImportError:
    from defog_2023 import defog_img

from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from skimage.color import rgb2lab, deltaE_ciede2000

# -------- 基本設定（依需求修改） --------
dataset = "SOTS_inout"
defog_version = "optimize_psi"  # 依你的輸出資料夾命名慣例
psi_start = 0.5
psi_end = 2.00
psi_step = 0.02
SAVE_FULL_CURVE = False  # 若要每張圖輸出 PSNR-psi 曲線 CSV，改成 True

# -------- 路徑 --------
base_dir = f"./dataset/{dataset}"
hazy_dir = os.path.join(base_dir, "hazy")
clear_dir = os.path.join(base_dir, "clear")

# 圖片輸出：按照你的舊規格 -> result_{defog_version}/{basename}_{defog_version}.png
result_dir = os.path.join(base_dir, f"result_{defog_version}")
os.makedirs(result_dir, exist_ok=True)

# 報表輸出：沿用 report 目錄，檔名避免覆蓋原 score，加上 _grid 後綴
report_dir = os.path.join(base_dir, "report")
os.makedirs(report_dir, exist_ok=True)
summary_csv = os.path.join(report_dir, f"score_{defog_version}_grid.csv")

# 可選：每張圖的 PSNR-psi 曲線
curve_dir = os.path.join(report_dir, "curves") if SAVE_FULL_CURVE else None
if SAVE_FULL_CURVE:
    os.makedirs(curve_dir, exist_ok=True)

# -------- 小工具 --------
def _load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def _ensure_same_size(img_a, img_b):
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    h = min(ha, hb)
    w = min(wa, wb)
    return img_a[:h, :w], img_b[:h, :w]

def _psnr(clear_arr, test_arr):
    clear_arr, test_arr = _ensure_same_size(clear_arr, test_arr)
    return sk_psnr(clear_arr, test_arr, data_range=255)

def _ssim(clear_arr, test_arr):
    clear_arr, test_arr = _ensure_same_size(clear_arr, test_arr)
    return sk_ssim(clear_arr, test_arr, channel_axis=-1)

def _ciede2000(clear_arr, test_arr):
    clear_arr, test_arr = _ensure_same_size(clear_arr, test_arr)
    lab_clear = rgb2lab(clear_arr)
    lab_test = rgb2lab(test_arr)
    delta_e = deltaE_ciede2000(lab_clear, lab_test)
    return float(np.mean(delta_e))

def _psi_range(start, end, step):
    # 產生含端點的等差序列，避免浮點累積誤差
    n = int(round((end - start) / step)) + 1
    return [round(start + i * step, 10) for i in range(n)]

# -------- 主流程 --------
def main():
    hazy_files = sorted(glob(os.path.join(hazy_dir, "*.png")))
    if not hazy_files:
        print(f"找不到霧化影像：{hazy_dir}")
        return

    psi_values = _psi_range(psi_start, psi_end, psi_step)

    # CSV 標頭（比你的 score 多 BestPsi 與 A）
    rows = []
    header = ["Image", "BestPsi", "Ar", "Ag", "Ab", "PSNR", "SSIM", "CIEDE2000", "Tried", "TotalTimeSec", "OutputPath"]
    rows.append(",".join(header))

    # for average row
    sum_psnr = 0.0
    sum_ssim = 0.0
    sum_ciede = 0.0
    count = 0

    for hazy_path in hazy_files:
        full_name = os.path.splitext(os.path.basename(hazy_path))[0]
        base_name = full_name.split("_")[0]
        clear_path = os.path.join(clear_dir, f"{base_name}_clear.png")

        if not os.path.exists(clear_path):
            print(f"⚠️ 找不到 ground truth：{clear_path}；跳過 {hazy_path}")
            continue

        H = _load_rgb(hazy_path)
        clear = _load_rgb(clear_path)

        best = {"psnr": -math.inf, "psi": None, "A": None, "D": None}
        curve = []
        t0 = time.time()

        for psi in psi_values:
            try:
                D, A = defog_img(H, psi=psi)
            except TypeError:
                D, A = defog_img(H, psi)

            cur_psnr = _psnr(clear, D)
            curve.append((psi, cur_psnr))

            # 以 PSNR 為主；同分時取較小 psi（保守）
            if (cur_psnr > best["psnr"]) or (
                abs(cur_psnr - best["psnr"]) < 1e-9 and (best["psi"] is None or psi < best["psi"])
            ):
                best.update({"psnr": cur_psnr, "psi": psi, "A": A, "D": D})

        elapsed = time.time() - t0

        if best["psi"] is None:
            print(f"⚠️ {hazy_path} 未找到有效 psi")
            continue

        # 最佳結果：算 SSIM/CIEDE2000
        best_ssim = _ssim(clear, best["D"])
        best_ciede = _ciede2000(clear, best["D"])

        # 圖片輸出到既有 defog_version 目錄（檔名沿用你原本慣例）
        out_name = f"{base_name}_{defog_version}.png"
        out_path = os.path.join(result_dir, out_name)
        Image.fromarray(best["D"]).save(out_path)

        Ar, Ag, Ab = [float(best["A"][i]) for i in range(3)]
        row = [
            base_name,
            f"{best['psi']:.2f}",
            f"{Ar:.6f}", f"{Ag:.6f}", f"{Ab:.6f}",
            f"{best['psnr']:.6f}",
            f"{best_ssim:.6f}",
            f"{best_ciede:.6f}",
            f"{len(psi_values)}",
            f"{elapsed:.3f}",
            out_path.replace(",", ";"),
        ]
        rows.append(",".join(row))

        # 加總做平均
        sum_psnr += best["psnr"]
        sum_ssim += best_ssim
        sum_ciede += best_ciede
        count += 1

        # 可選：輸出每張圖的 PSNR–psi 曲線
        if SAVE_FULL_CURVE:
            curve_csv = os.path.join(curve_dir, f"{base_name}_curve.csv")
            with open(curve_csv, "w", encoding="utf-8") as f:
                f.write("Psi,PSNR\n")
                for p, v in curve:
                    f.write(f"{p:.4f},{v:.6f}\n")

        print(
            f"[{base_name}] best psi={best['psi']:.2f}, "
            f"PSNR={best['psnr']:.3f}, SSIM={best_ssim:.4f}, CIEDE2000={best_ciede:.3f}, "
            f"A=({Ar:.1f},{Ag:.1f},{Ab:.1f}) -> {out_path}"
        )

    # 平均列
    if count > 0:
        avg_row = [
            "AVERAGE", "", "", "", "",
            f"{(sum_psnr / count):.6f}",
            f"{(sum_ssim / count):.6f}",
            f"{(sum_ciede / count):.6f}",
            "", "", ""
        ]
        rows.append(",".join(avg_row))

    # 寫出 CSV
    with open(summary_csv, "w", encoding="utf-8") as f:
        for line in rows:
            f.write(line + "\n")

    print("\n✅ 全部完成")
    print(f"✅ 最佳去霧影像輸出：{result_dir}")
    print(f"✅ 彙總評分輸出：{summary_csv}")
    if SAVE_FULL_CURVE:
        print(f"ℹ️ PSNR–psi 曲線輸出：{curve_dir}")

if __name__ == "__main__":
    main()
