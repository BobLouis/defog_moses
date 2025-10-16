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

# -------- 簡化霧氣估算（適合硬體電路）--------
def estimate_fog_simple(img_array):
    """
    簡化版霧氣估算 - 調整後的評分公式
    """
    
    # 如果是彩色圖，轉灰階（簡化版：只取 R channel）
    if len(img_array.shape) == 3:
        gray = img_array[:, :, 0]
    else:
        gray = img_array
    
    height, width = gray.shape
    total_pixels = height * width
    
    # ========== 單次掃描計算所有指標 ==========
    sum_intensity = 0
    max_val = 0
    min_val = 255
    local_diff_sum = 0
    
    for i in range(height):
        for j in range(width):
            pixel = int(gray[i, j])
            
            sum_intensity += pixel
            
            if pixel > max_val:
                max_val = pixel
            if pixel < min_val:
                min_val = pixel
            
            if j < width - 1:
                diff = abs(pixel - int(gray[i, j+1]))
                local_diff_sum += diff
            if i < height - 1:
                diff = abs(pixel - int(gray[i+1, j]))
                local_diff_sum += diff
    
    # ========== 計算指標 ==========
    avg_intensity = sum_intensity // total_pixels
    dynamic_range = max_val - min_val
    avg_local_diff = local_diff_sum // (2 * total_pixels)
    
    # 第二次掃描：計算變異度
    variance_sum = 0
    for i in range(height):
        for j in range(width):
            pixel = int(gray[i, j])
            variance_sum += abs(pixel - avg_intensity)
    
    avg_deviation = variance_sum // total_pixels
    
    # ========== 改進的霧氣評分計算 ==========
    # 根據實際數據分布調整閾值
    
    # 方法1：基於動態範圍（更嚴格的映射）
    # 動態範圍 250+ → 0分（無霧）
    # 動態範圍 100- → 100分（濃霧）
    if dynamic_range >= 240:
        fog_score_range = 0
    elif dynamic_range <= 100:
        fog_score_range = 100
    else:
        # 線性映射 100-240 → 100-0
        fog_score_range = int(100 - ((dynamic_range - 100) / 140.0) * 100)
    
    # 方法2：基於平均偏差（調整閾值）
    # 偏差 60+ → 0分（無霧）
    # 偏差 20- → 100分（濃霧）
    if avg_deviation >= 60:
        fog_score_deviation = 0
    elif avg_deviation <= 20:
        fog_score_deviation = 100
    else:
        fog_score_deviation = int(100 - ((avg_deviation - 20) / 40.0) * 100)
    
    # 方法3：基於局部差異（調整閾值）
    # 局部差異 10+ → 0分（無霧）
    # 局部差異 1- → 100分（濃霧）
    if avg_local_diff >= 10:
        fog_score_edge = 0
    elif avg_local_diff <= 1:
        fog_score_edge = 100
    else:
        fog_score_edge = int(100 - ((avg_local_diff - 1) / 9.0) * 100)
    
    # 綜合評分：動態範圍權重最高（50%），其他各25%
    fog_score = int((fog_score_range * 2 + fog_score_deviation + fog_score_edge) / 4)
    fog_score = max(0, min(100, fog_score))
    
    # 霧氣等級（調整閾值）
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
        'avg_intensity': avg_intensity,
        'dynamic_range': dynamic_range,
        'avg_local_diff': avg_local_diff,
        'avg_deviation': avg_deviation,
        'fog_score_range': fog_score_range,
        'fog_score_deviation': fog_score_deviation,
        'fog_score_edge': fog_score_edge,
        'fog_score': fog_score,
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
        "AvgIntensity",
        "DynamicRange",
        "AvgLocalDiff",
        "AvgDeviation",
        "FogScoreRange",
        "FogScoreDeviation",
        "FogScoreEdge",
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
            
            # 估算霧氣濃度
            metrics = estimate_fog_simple(img_array)
            
            # 計時結束
            elapsed = time.time() - t0
            
            # 組成 CSV row
            row = [
                base_name,
                f"{metrics['avg_intensity']}",
                f"{metrics['dynamic_range']}",
                f"{metrics['avg_local_diff']}",
                f"{metrics['avg_deviation']}",
                f"{metrics['fog_score_range']}",
                f"{metrics['fog_score_deviation']}",
                f"{metrics['fog_score_edge']}",
                f"{metrics['fog_score']}",
                metrics['fog_level'],
                f"{elapsed:.4f}"
            ]
            rows.append(",".join(row))
            
            sum_fog_score += metrics['fog_score']
            count += 1
            
            # 輸出進度
            print(f"[{base_name}] FogScore={metrics['fog_score']} ({metrics['fog_level']}), "
                  f"Range={metrics['dynamic_range']}, Deviation={metrics['avg_deviation']}")
        
        except Exception as e:
            print(f"⚠️ 處理 {hazy_path} 時發生錯誤：{e}")
            continue
    
    # 計算平均值
    if count > 0:
        avg_row = [
            "AVERAGE", "", "", "", "", "", "", "",
            f"{int(sum_fog_score / count)}",
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
        print(f"📊 平均霧氣評分：{int(sum_fog_score / count)}")


if __name__ == "__main__":
    main()