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
summary_csv = os.path.join(report_dir, "fog_estimation_dc_edge.csv")

# -------- 小工具：快速 3x3 最小濾波（純 numpy，無需額外套件）--------
def min_filter_3x3(gray_uint8):
    """
    3x3 最小值濾波（morphological erosion），不使用第三方套件。
    以九個位移版本取 element-wise 的最小值。
    """
    h, w = gray_uint8.shape
    padded = np.pad(gray_uint8, ((1,1),(1,1)), mode='edge')
    # 取九個位移視圖
    neighs = [
        padded[0:h,   0:w  ], padded[0:h,   1:w+1], padded[0:h,   2:w+2],
        padded[1:h+1, 0:w  ], padded[1:h+1, 1:w+1], padded[1:h+1, 2:w+2],
        padded[2:h+2, 0:w  ], padded[2:h+2, 1:w+1], padded[2:h+2, 2:w+2],
    ]
    out = neighs[0]
    for k in range(1, 9):
        out = np.minimum(out, neighs[k])
    return out

# -------- DC + 偽邊緣 霧氣估算（硬體友善）--------
def estimate_fog_dc_edge(img_array):
    """
    DC + 偽邊緣 霧濃度估算
    指標：
      - DCMean: 暗通道均值
      - DCP90 : 暗通道第90百分位
      - PseudoEdgeRatio: 低對比區中被判為邊緣的比例（偽邊緣比例）
    FogScore: 以上三者加權線性映射為 0~100
    """
    # ---- 1) 準備資料 ----
    # RGB -> 暗通道（逐像素 min）
    if img_array.dtype != np.uint8:
        img = np.clip(img_array, 0, 255).astype(np.uint8)
    else:
        img = img_array

    # 暗通道 per-pixel（不跨視窗）
    dc = np.min(img, axis=2)  # uint8
    # 再做 3x3 最小濾波以近似局部暗通道（DCP 的局部版本）
    dc_local = min_filter_3x3(dc)

    # 統計值
    dc_mean = float(np.mean(dc_local))
    dc_p90 = float(np.percentile(dc_local, 90))

    # ---- 2) 偽邊緣比例 ----
    # 亮度 Y：這裡取 RGB 平均（硬體簡單）
    y = img.mean(axis=2).astype(np.float32)

    # 梯度（使用簡單前向差分，避免捲積）
    # Gx ~ |Y[x,y] - Y[x,y-1]|, Gy ~ |Y[x,y] - Y[x-1,y]|
    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    gx[:, 1:] = np.abs(y[:, 1:] - y[:, :-1])
    gy[1:, :] = np.abs(y[1:, :] - y[:-1, :])
    g = gx + gy  # L1 近似梯度幅值（硬體省乘法）

    # 局部對比（3x3 範圍）：max-min
    # 用 3x3 的最大與最小快速近似（與 min_filter 類似作法）
    def max_filter_3x3(gray_f32):
        h, w = gray_f32.shape
        padded = np.pad(gray_f32, ((1,1),(1,1)), mode='edge')
        neighs = [
            padded[0:h,   0:w  ], padded[0:h,   1:w+1], padded[0:h,   2:w+2],
            padded[1:h+1, 0:w  ], padded[1:h+1, 1:w+1], padded[1:h+1, 2:w+2],
            padded[2:h+2, 0:w  ], padded[2:h+2, 1:w+1], padded[2:h+2, 2:w+2],
        ]
        out = neighs[0]
        for k in range(1, 9):
            out = np.maximum(out, neighs[k])
        return out

    y_max = max_filter_3x3(y)
    y_min = min_filter_3x3(y.astype(np.uint8)).astype(np.float32)  # 這裡用前面 uint8 版 min，可視需要改為 float 版
    local_contrast = y_max - y_min  # 3x3 範圍內的亮度 range

    # 適應式門檻（以分位數作穩健閾值）
    # 邊緣門檻：梯度的 P75；對比門檻：local_contrast 的 P40
    # 你可以依資料集再微調
    tau_g = float(np.percentile(g, 75.0))
    tau_c = float(np.percentile(local_contrast, 40.0))

    edges = (g > tau_g)
    low_contrast = (local_contrast < tau_c)
    pseudo_edges = edges & low_contrast

    edge_count = int(np.count_nonzero(edges))
    pseudo_count = int(np.count_nonzero(pseudo_edges))
    pseudo_edge_ratio = float(pseudo_count) / float(edge_count + 1e-6)

    # ---- 3) 映射到 0~100 的 FogScore ----
    # A) 暗通道分數（暗通道越亮→霧越濃）
    # 以經驗範圍做線性縮放：dc in [10, 180] → score in [0, 100]
    def linear_score(x, lo, hi):
        if x <= lo: return 0.0
        if x >= hi: return 100.0
        return 100.0 * (x - lo) / (hi - lo)

    dc_mean_score = linear_score(dc_mean, 10.0, 180.0)
    dc_p90_score  = linear_score(dc_p90,  30.0, 220.0)

    # B) 偽邊緣比例分數（比例越高→霧越濃）
    # 經驗縮放：ratio in [0.05, 0.6] → [0, 100]
    pseudo_edge_score = linear_score(pseudo_edge_ratio, 0.05, 0.60)

    # C) 加權合成（可再用資料集做最小平方法微調 a,b,c）
    # 初始：暗通道均值 0.5、P90 0.3、偽邊緣 0.2
    fog_score = 0.5 * dc_mean_score + 0.3 * dc_p90_score + 0.2 * pseudo_edge_score
    fog_score = float(np.clip(fog_score, 0.0, 100.0))

    # 霧氣等級（與你現有分級一致）
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
        'dc_mean': round(dc_mean, 4),
        'dc_p90': round(dc_p90, 4),
        'pseudo_edge_ratio': round(pseudo_edge_ratio, 6),
        'dc_mean_score': round(dc_mean_score, 2),
        'dc_p90_score': round(dc_p90_score, 2),
        'pseudo_edge_score': round(pseudo_edge_score, 2),
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
        "DCMean",
        "DCP90",
        "PseudoEdgeRatio",
        "DCMeanScore",
        "DCP90Score",
        "PseudoEdgeScore",
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

            # 估算霧氣濃度（DC-Edge）
            metrics = estimate_fog_dc_edge(img_array)

            # 計時結束
            elapsed = time.time() - t0

            # 組成 CSV row
            row = [
                base_name,
                f"{metrics['dc_mean']}",
                f"{metrics['dc_p90']}",
                f"{metrics['pseudo_edge_ratio']}",
                f"{metrics['dc_mean_score']}",
                f"{metrics['dc_p90_score']}",
                f"{metrics['pseudo_edge_score']}",
                f"{metrics['fog_score']}",
                metrics['fog_level'],
                f"{elapsed:.4f}"
            ]
            rows.append(",".join(row))

            sum_fog_score += metrics['fog_score']
            count += 1

            # 輸出進度
            print(f"[{base_name}] FogScore={metrics['fog_score']} ({metrics['fog_level']}), "
                  f"DCMean={metrics['dc_mean']}, DCP90={metrics['dc_p90']}, "
                  f"PseudoEdge={metrics['pseudo_edge_ratio']:.4f}")

        except Exception as e:
            print(f"⚠️ 處理 {hazy_path} 時發生錯誤：{e}")
            continue

    # 計算平均值
    if count > 0:
        avg_row = [
            "AVERAGE",
            "", "", "",
            "", "", "",
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
