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
summary_csv = os.path.join(report_dir, "fog_estimation_simple.csv")


# -------- 新版 tile-based 霧氣估算（對應核心 predict_psi）--------
def estimate_fog_tile(img_array, tile=8, use_y=True, percentile=80):
    """
    使用與核心除霧相同邏輯（分塊 + 分位數）估算霧分數與 PSI。
    回傳用來輸出 CSV 的各項指標。
    """
    # 確保 uint8
    img = img_array if img_array.dtype == np.uint8 else np.clip(img_array, 0, 255).astype(np.uint8)

    # 轉灰階：預設使用 Y (建議)，也可切回 R channel
    if use_y:
        y = (0.299 * img[:, :, 0] +
             0.587 * img[:, :, 1] +
             0.114 * img[:, :, 2]).astype(np.uint8)
    else:
        y = img[:, :, 0]

    H, W = y.shape
    th, tw = H // tile, W // tile

    tile_scores = []   # 每個 tile 的 fog_score_tile
    dr_list = []       # 每個 tile 的 dr_score
    dev_list = []      # 每個 tile 的 dev_score
    edge_list = []     # 每個 tile 的 edge_score

    for i in range(tile):
        for j in range(tile):
            patch = y[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            if patch.size == 0:
                continue

            max_val = patch.max()
            min_val = patch.min()
            avg_intensity = patch.mean()
            dynamic_range = max_val - min_val
            avg_deviation = np.mean(np.abs(patch - avg_intensity))

            # 邊緣：可視邊比例
            diff_h = np.abs(patch[:, 1:].astype(np.int16) - patch[:, :-1].astype(np.int16))
            diff_v = np.abs(patch[1:, :].astype(np.int16) - patch[:-1, :].astype(np.int16))

            grad = np.zeros_like(patch, dtype=np.float32)
            grad[:, 1:] += diff_h
            grad[1:, :] += diff_v

            # 自適應邊緣閾值：patch 內分位數
            thr = np.percentile(grad, 75)
            visible_edge_ratio = (grad > max(thr, 1)).mean()  # 0~1

            # 重新映射成霧分數（越霧 → 越高）
            dr_score = 100.0 * (1.0 - np.clip(dynamic_range / 255.0, 0.0, 1.0))
            dev_score = 100.0 * (1.0 - np.clip(avg_deviation / 64.0, 0.0, 1.0))
            edge_score = 100.0 * (1.0 - np.clip(visible_edge_ratio / 0.12, 0.0, 1.0))

            fog_score_tile = (dr_score * 2.0 + dev_score + edge_score) / 4.0

            tile_scores.append(fog_score_tile)
            dr_list.append(dr_score)
            dev_list.append(dev_score)
            edge_list.append(edge_score)

    # 全圖霧分數：用高分位數捕捉局部濃霧
    if not tile_scores:
        fog_score = 50.0
    else:
        fog_score = float(np.percentile(tile_scores, percentile))

    fog_score = float(np.clip(fog_score, 0.0, 100.0))

    # 目前的線性關係（你之後重算回歸，就改這兩個係數）
    psi = 0.011099 * fog_score + 0.746386

    # 一些統計量方便回頭分析
    metrics = {
        "fog_score": fog_score,
        "psi": psi,
        "tile": tile,
        "percentile": percentile,
        "use_y": int(bool(use_y)),
        "fog_score_mean_tile": float(np.mean(tile_scores)) if tile_scores else 50.0,
        "fog_score_std_tile": float(np.std(tile_scores)) if tile_scores else 0.0,
        "dr_mean": float(np.mean(dr_list)) if dr_list else 0.0,
        "dev_mean": float(np.mean(dev_list)) if dev_list else 0.0,
        "edge_mean": float(np.mean(edge_list)) if edge_list else 0.0,
    }

    return metrics


# -------- 主流程 --------
def main():
    hazy_files = sorted(glob(os.path.join(hazy_dir, "*.png")))
    if not hazy_files:
        print(f"找不到霧化影像：{hazy_dir}")
        return

    print(f"找到 {len(hazy_files)} 張圖片\n")

    header = [
        "Image",
        "FogScore",          # 分塊 + 分位數後的霧分數（0~100）
        "Psi",               # 目前線性公式算出的 PSI
        "Tile",              # tile 大小（tile x tile）
        "Percentile",        # 使用的分位數
        "UseY",              # 是否用 Y 通道（1=Y, 0=R）
        "FogScoreMeanTile",  # 各 tile 霧分數平均
        "FogScoreStdTile",   # 各 tile 霧分數標準差
        "DrMean",            # dr_score 平均
        "DevMean",           # dev_score 平均
        "EdgeMean",          # edge_score 平均
        "ProcessTimeSec"
    ]

    rows = [",".join(header)]
    sum_fog_score = 0.0
    sum_psi = 0.0
    count = 0

    for hazy_path in hazy_files:
        base_name = os.path.splitext(os.path.basename(hazy_path))[0]

        try:
            img = Image.open(hazy_path).convert("RGB")
            img_array = np.array(img)

            t0 = time.time()
            metrics = estimate_fog_tile(img_array, tile=8, use_y=True, percentile=80)
            elapsed = time.time() - t0

            row = [
                base_name,
                f"{metrics['fog_score']:.4f}",
                f"{metrics['psi']:.4f}",
                f"{metrics['tile']}",
                f"{metrics['percentile']}",
                f"{metrics['use_y']}",
                f"{metrics['fog_score_mean_tile']:.4f}",
                f"{metrics['fog_score_std_tile']:.4f}",
                f"{metrics['dr_mean']:.4f}",
                f"{metrics['dev_mean']:.4f}",
                f"{metrics['edge_mean']:.4f}",
                f"{elapsed:.4f}",
            ]
            rows.append(",".join(row))

            sum_fog_score += metrics['fog_score']
            sum_psi += metrics['psi']
            count += 1

            print(f"[{base_name}] FogScore={metrics['fog_score']:6.2f}  "
                  f"Psi={metrics['psi']:.3f}  mean_tile={metrics['fog_score_mean_tile']:6.2f}")

        except Exception as e:
            print(f"⚠️ 處理 {base_name} 時發生錯誤：{e}")
            continue

    # 平均值行
    if count > 0:
        avg_fog = sum_fog_score / count
        avg_psi = sum_psi / count
        avg_row = [
            "AVERAGE",
            f"{avg_fog:.4f}",
            f"{avg_psi:.4f}",
            "", "", "", "", "", "", "", "",
            ""
        ]
        rows.append(",".join(avg_row))

    with open(summary_csv, "w", encoding="utf-8") as f:
        for line in rows:
            f.write(line + "\n")

    print("\n✅ 全部完成")
    print(f"✅ 共處理 {count} 張圖片")
    print(f"✅ 報表輸出：{summary_csv}")
    if count > 0:
        print(f"📊 平均 FogScore：{avg_fog:.2f}")
        print(f"📊 平均 Psi：{avg_psi:.3f}")


if __name__ == "__main__":
    main()
