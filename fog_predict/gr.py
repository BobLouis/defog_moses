import os
import time
from glob import glob
import numpy as np
from PIL import Image

# -------- 基本設定 --------
dataset = "SOTS_inout"
# dataset = "OHaze"

# -------- 路徑 --------
base_dir = f"./dataset/{dataset}"
hazy_dir = os.path.join(base_dir, "hazy")

# 報表輸出
report_dir = os.path.join(base_dir, "report")
os.makedirs(report_dir, exist_ok=True)
summary_csv = os.path.join(report_dir, "fog_estimation_gr_nb.csv")

# -------- 小工具：3x3 最小/最大濾波（純 numpy）--------
def min_filter_3x3(arr):
    h, w = arr.shape
    padded = np.pad(arr, ((1,1),(1,1)), mode='edge')
    neighs = [
        padded[0:h,   0:w  ], padded[0:h,   1:w+1], padded[0:h,   2:w+2],
        padded[1:h+1, 0:w  ], padded[1:h+1, 1:w+1], padded[1:h+1, 2:w+2],
        padded[2:h+2, 0:w  ], padded[2:h+2, 1:w+1], padded[2:h+2, 2:w+2],
    ]
    out = neighs[0]
    for k in range(1, 9):
        out = np.minimum(out, neighs[k])
    return out

def max_filter_3x3(arr):
    h, w = arr.shape
    padded = np.pad(arr, ((1,1),(1,1)), mode='edge')
    neighs = [
        padded[0:h,   0:w  ], padded[0:h,   1:w+1], padded[0:h,   2:w+2],
        padded[1:h+1, 0:w  ], padded[1:h+1, 1:w+1], padded[1:h+1, 2:w+2],
        padded[2:h+2, 0:w  ], padded[2:h+2, 1:w+1], padded[2:h+2, 2:w+2],
    ]
    out = neighs[0]
    for k in range(1, 9):
        out = np.maximum(out, neighs[k])
    return out

# -------- GR/NB 霧氣估算（硬體友善）--------
def estimate_fog_gr_nb(img_array):
    """
    GR/NB（梯度比率／可視邊）霧濃度估算
    指標：
      - NBRatio: 可視邊比例（在強對比 + 強梯度條件下的邊緣佔比）
      - GRMean : 可視邊上的梯度比率，G / G_max(3x3)
    FogScore: (1-GR) 與 (1-NB) 的加權映射到 0~100
    """
    # ---- 1) 準備資料 ----
    if img_array.dtype != np.uint8:
        img = np.clip(img_array, 0, 255).astype(np.uint8)
    else:
        img = img_array

    # 亮度 Y（硬體友善：RGB 平均）
    y = img.mean(axis=2).astype(np.float32)

    # ---- 2) 梯度與局部對比 ----
    # 簡單前向差分（硬體省乘法）：L1 近似梯度
    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    gx[:, 1:] = np.abs(y[:, 1:] - y[:, :-1])
    gy[1:, :] = np.abs(y[1:, :] - y[:-1, :])
    g = gx + gy  # 梯度幅值近似

    # 局部對比（3x3）：max - min
    y_max = max_filter_3x3(y)
    y_min = min_filter_3x3(y.astype(np.uint8)).astype(np.float32)
    local_contrast = y_max - y_min

    # ---- 3) 可視邊集合 E 的自適應門檻 ----
    # 以分位數當穩健閾值（可再依資料校準）
    tau_g = float(np.percentile(g, 75.0))        # 強梯度
    tau_c = float(np.percentile(local_contrast, 50.0))  # 強對比
    visible_edges = (g > tau_g) & (local_contrast > tau_c)

    # NBRatio：可視邊比例（對全部像素正規化，硬體只需計數器）
    total_pixels = y.size
    nb_ratio = float(np.count_nonzero(visible_edges)) / float(total_pixels)

    # ---- 4) GR（梯度比率）----
    # 以 3x3 區域最大梯度當作本地上限
    g_max_local = max_filter_3x3(g)
    eps = 1e-6
    # 只在可視邊集合上計算比率，避免平坦區域的 0/0
    if np.any(visible_edges):
        gr_vals = g[visible_edges] / (g_max_local[visible_edges] + eps)
        # 理論上 gr ∈ (0,1]，但做個穩健裁切防數值誤差
        gr_vals = np.clip(gr_vals, 0.0, 1.0)
        gr_mean = float(np.mean(gr_vals))
    else:
        gr_mean = 0.0  # 沒有可視邊（極濃霧或極平坦），視為最差情況

    # ---- 5) 映射到 0~100 的 FogScore ----
    # 定義正向/反向線性映射（便於閱讀與後續調參）
    def linear_score(x, lo, hi):
        # x in [lo,hi] -> [0,100]
        if x <= lo: return 0.0
        if x >= hi: return 100.0
        return 100.0 * (x - lo) / (hi - lo)

    def inverse_linear_score(x, lo, hi):
        # x in [lo,hi]，值越小越「霧」，映射到越高分
        if x <= lo: return 100.0
        if x >= hi: return 0.0
        return 100.0 * (hi - x) / (hi - lo)

    # A) GRScore：GR 越小 → 霧越濃（反向）
    # 經驗區間：gr_mean ∈ [0.2, 0.9] 之間映射到 [100,0]
    gr_score = inverse_linear_score(gr_mean, lo=0.20, hi=0.90)

    # B) NBScore：NBRatio 越小 → 霧越濃（反向）
    # 經驗區間：nb_ratio ∈ [0.02, 0.25] 映射到 [100,0]
    nb_score = inverse_linear_score(nb_ratio, lo=0.02, hi=0.25)

    # C) 合成（初始建議 0.6:0.4，可再以資料回歸微調）
    fog_score = 0.6 * gr_score + 0.4 * nb_score
    fog_score = float(np.clip(fog_score, 0.0, 100.0))

    # 分級（與你既有一致）
    if fog_score < 25:
        fog_level = "Clear"
    elif fog_score < 45:
        fog_level = "Light"
    elif fog_score < 65:
        fog_level = "Moderate"
    elif fog_score < 85:
        fog_level = "Heavy"
    else:
        fog_level = "Dense"

    metrics = {
        'nb_ratio': round(nb_ratio, 6),
        'gr_mean': round(gr_mean, 6),
        'nb_score': round(nb_score, 2),
        'gr_score': round(gr_score, 2),
        'fog_score': round(fog_score, 2),
        'fog_level': fog_level
    }
    return metrics

# -------- 主流程 --------
def main():
    # 尋找霧化圖片
    hazy_files = sorted(glob(os.path.join(hazy_dir, "*.png")))
    if not hazy_files:
        print(f"找不到霧化影像：{hazy_dir}")
        return

    print(f"找到 {len(hazy_files)} 張圖片")

    # CSV 標頭
    rows = []
    header = [
        "Image",
        "NBRatio",
        "GRMean",
        "NBScore",
        "GRScore",
        "FogScore",
        "FogLevel",
        "ProcessTimeSec"
    ]
    rows.append(",".join(header))

    # 用於計算平均值
    sum_fog_score = 0.0
    count = 0

    # 批量處理
    for hazy_path in hazy_files:
        full_name = os.path.splitext(os.path.basename(hazy_path))[0]
        base_name = full_name.split("_")[0]

        try:
            # 讀取圖片
            img = Image.open(hazy_path).convert("RGB")
            img_array = np.array(img)

            # 計時開始
            t0 = time.time()

            # 估算霧氣濃度（GR/NB）
            metrics = estimate_fog_gr_nb(img_array)

            # 計時結束
            elapsed = time.time() - t0

            # 組成 CSV row
            row = [
                base_name,
                f"{metrics['nb_ratio']}",
                f"{metrics['gr_mean']}",
                f"{metrics['nb_score']}",
                f"{metrics['gr_score']}",
                f"{metrics['fog_score']}",
                metrics['fog_level'],
                f"{elapsed:.4f}"
            ]
            rows.append(",".join(row))

            sum_fog_score += metrics['fog_score']
            count += 1

            # 輸出進度
            print(f"[{base_name}] FogScore={metrics['fog_score']} ({metrics['fog_level']}), "
                  f"NBRatio={metrics['nb_ratio']:.4f}, GRMean={metrics['gr_mean']:.4f}")

        except Exception as e:
            print(f"⚠️ 處理 {hazy_path} 時發生錯誤：{e}")
            continue

    # 計算平均值
    if count > 0:
        avg_row = [
            "AVERAGE",
            "", "", "", "",
            f"{round(sum_fog_score / count, 2)}",
            "", ""
        ]
        rows.append(",".join(avg_row))

    # 寫入 CSV
    with open(summary_csv, "w", encoding="utf-8") as f:
        for line in rows:
            f.write(line + "\n")

    print("\n✅ 全部完成")
    print(f"✅ 共處理 {count} 張圖片")
    print(f"✅ 評分報表輸出：{summary_csv}")
    if count > 0:
        print(f"📊 平均霧氣評分：{round(sum_fog_score / count, 2)}")

if __name__ == "__main__":
    main()
