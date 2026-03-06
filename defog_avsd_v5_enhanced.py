# defog_avsd_v5_enhanced.py (優化增強版本)
# 添加了多種後處理技術來提升PSNR和視覺質量

import numpy as np
from scipy.ndimage import minimum_filter
import cv2

"""
原始公式（AVSD方法）
H(x) = D(x)*t(x) + A*(1-t(x))
D(x) = ((H(x) - A) / t(x)) + A

優化增強：
1. ✅ 保留原始AVSD核心算法
2. ✅ 添加CLAHE後處理（關鍵提升）
3. ✅ 添加Gamma校正選項
4. ✅ 添加引導濾波選項
5. ✅ 參數自動優化
"""


def predict_psi(image):
    """
    基於硬體霧氣評分預測最佳 PSI 值（V5 版本：無降採樣）
    回歸公式：BestPsi = 0.009308 × HW_FogScore + 0.927009
    限制範圍：0.5 ~ 1.2
    """
    gray = image[:, :, 0]
    height, width = gray.shape

    # 單次掃描找最大最小值
    max_val = 0
    min_val = 255

    for i in range(height):
        for j in range(width):
            pixel = int(gray[i, j])
            if pixel > max_val:
                max_val = pixel
            if pixel < min_val:
                min_val = pixel

    # 計算動態範圍
    dynamic_range = max_val - min_val

    # 霧氣評分
    if dynamic_range >= 240:
        fog_score = 0
    elif dynamic_range <= 100:
        fog_score = 100
    else:
        fog_score = (240 - dynamic_range) >> 1

    fog_score = max(0, min(100, fog_score))

    # 套用回歸公式
    BestPsi = 0.009308 * fog_score + 0.927009
    BestPsi = max(0.5, min(1.2, BestPsi))

    return BestPsi


def apply_clahe(image, clip_limit=2.0, tile_size=16):
    """
    應用CLAHE（對比度受限自適應直方圖均衡化）

    參數:
        image: RGB圖像（np.uint8）
        clip_limit: 對比度限制（推薦 1.5-3.0）
        tile_size: 分塊大小（推薦 8 或 16）

    返回:
        增強後的圖像（np.uint8）
    """
    # 轉換到LAB色彩空間
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # 對L通道應用CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)

    # 合併回去
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    return rgb_enhanced


def apply_gamma_correction(image, gamma=1.0):
    """
    應用Gamma校正

    參數:
        image: RGB圖像（np.uint8）
        gamma: Gamma值（< 1變亮，> 1變暗，1.0不變）

    返回:
        校正後的圖像（np.uint8）
    """
    if gamma == 1.0:
        return image

    # 構建查找表
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                     for i in range(256)]).astype(np.uint8)

    # 應用查找表
    return cv2.LUT(image, table)


def guided_filter(I, p, r=15, eps=0.01):
    """
    引導濾波器（用於平滑transmission map，減少halo效應）

    參數:
        I: 引導圖像
        p: 待濾波圖像
        r: 窗口半徑
        eps: 正則化參數

    返回:
        濾波後的圖像
    """
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * I + mean_b
    return q


def defog_img_basic(hazy_image, psi=1, t0=0.2, window_size=8, epsilon=1e-6):
    """
    基礎AVSD去霧（原始算法，無後處理）

    參數:
        hazy_image: 輸入圖像（RGB，np.uint8）
        psi: 擬合係數（會被自動計算的 BestPsi 覆蓋）
        t0: 傳輸圖的下界（預設 0.2）
        window_size: 最小濾波器窗口大小（8x8）
        epsilon: 防止除零的小常數

    返回:
        D: 去霧後的圖像（np.uint8）
        A: 大氣光向量（3,）
        BestPsi: 自動計算的最佳 PSI 值
        t: 傳輸圖（用於後續優化）
    """
    H = hazy_image.astype(np.float32)

    # 動態PSI預測
    BestPsi = predict_psi(H)
    psi = BestPsi

    # 計算大氣光A
    dark_channel = np.min(H, axis=2)
    dark_min = minimum_filter(dark_channel, size=window_size)

    idx = np.argmax(dark_min)
    y, x = np.unravel_index(idx, dark_min.shape)
    A = H[y, x, :].copy()

    # 歸一化
    H_norm = H / (A + epsilon)
    K = np.mean(H_norm, axis=2)
    min_norm = np.min(H_norm, axis=2)

    # 計算傳輸圖
    temp = 3 * K + 3 * min_norm
    t = (temp - psi * 3 * K * min_norm) / (temp + epsilon)
    t = np.clip(t, t0, 1)

    # 恢復無霧圖像
    t_expanded = t[:, :, np.newaxis]
    D = (H - A) / t_expanded + A
    D = np.clip(D, 0, 255).astype(np.uint8)

    return D, A, BestPsi, t


def defog_img_enhanced(hazy_image,
                      psi=1,
                      t0=0.2,
                      window_size=8,
                      use_clahe=True,
                      clahe_clip_limit=2.0,
                      clahe_tile_size=16,
                      use_gamma=False,
                      gamma_value=1.0,
                      use_guided_filter=False,
                      guided_filter_radius=15):
    """
    增強版AVSD去霧（帶多種後處理選項）

    參數:
        hazy_image: 輸入圖像（RGB，np.uint8）
        psi: 擬合係數（會被自動計算覆蓋）
        t0: 傳輸圖下界（預設 0.2）
        window_size: 暗通道窗口大小

        後處理選項：
        use_clahe: 是否使用CLAHE（強烈推薦，提升SSIM）
        clahe_clip_limit: CLAHE對比度限制（推薦 1.5-3.0）
        clahe_tile_size: CLAHE分塊大小（推薦 8 或 16）
        use_gamma: 是否使用Gamma校正
        gamma_value: Gamma值（< 1變亮，> 1變暗）
        use_guided_filter: 是否對transmission使用引導濾波
        guided_filter_radius: 引導濾波半徑

    返回:
        D: 增強去霧後的圖像（np.uint8）
        A: 大氣光向量（3,）
        BestPsi: 自動計算的最佳 PSI 值
        info: 處理信息字典
    """
    H = hazy_image.astype(np.float32)
    epsilon = 1e-6

    # 動態PSI預測
    BestPsi = predict_psi(H)
    psi = BestPsi

    # 計算大氣光A
    dark_channel = np.min(H, axis=2)
    dark_min = minimum_filter(dark_channel, size=window_size)

    idx = np.argmax(dark_min)
    y, x = np.unravel_index(idx, dark_min.shape)
    A = H[y, x, :].copy()

    # 歸一化
    H_norm = H / (A + epsilon)
    K = np.mean(H_norm, axis=2)
    min_norm = np.min(H_norm, axis=2)

    # 計算傳輸圖
    temp = 3 * K + 3 * min_norm
    t = (temp - psi * 3 * K * min_norm) / (temp + epsilon)
    t = np.clip(t, t0, 1)

    # 可選：對transmission使用引導濾波
    if use_guided_filter:
        gray = cv2.cvtColor(hazy_image, cv2.COLOR_RGB2GRAY) / 255.0
        t = guided_filter(gray, t, r=guided_filter_radius, eps=0.01)
        t = np.clip(t, t0, 1)

    # 恢復無霧圖像
    t_expanded = t[:, :, np.newaxis]
    D = (H - A) / t_expanded + A
    D = np.clip(D, 0, 255).astype(np.uint8)

    # 後處理1: Gamma校正（可選）
    if use_gamma and gamma_value != 1.0:
        D = apply_gamma_correction(D, gamma_value)

    # 後處理2: CLAHE（推薦，關鍵提升！）
    if use_clahe:
        D = apply_clahe(D, clip_limit=clahe_clip_limit, tile_size=clahe_tile_size)

    # 收集處理信息
    info = {
        'psi': BestPsi,
        'airlight': A,
        'used_clahe': use_clahe,
        'used_gamma': use_gamma,
        'used_guided_filter': use_guided_filter,
        't_mean': np.mean(t),
        't_min': np.min(t),
        't_max': np.max(t)
    }

    return D, A, BestPsi, info


def defog_img(hazy_image, psi=1, t0=0.2, window_size=8, epsilon=1e-6):
    """
    向後兼容的接口（使用優化的默認配置）

    推薦配置：
    - CLAHE開啟（clip_limit=2.0, tile_size=16）
    - Gamma關閉
    - Guided Filter關閉（對AVSD方法影響較小）

    參數:
        hazy_image: 輸入圖像（RGB，np.uint8）
        psi: 擬合係數（會被自動覆蓋）
        t0: 傳輸圖下界
        window_size: 暗通道窗口
        epsilon: 防止除零

    返回:
        D: 去霧後的圖像（np.uint8）
        A: 大氣光向量（3,）
        BestPsi: 自動計算的PSI值
    """
    # 使用優化的默認配置
    D, A, BestPsi, info = defog_img_enhanced(
        hazy_image,
        psi=psi,
        t0=t0,
        window_size=window_size,
        use_clahe=True,        # 關鍵：開啟CLAHE
        clahe_clip_limit=2.0,   # 最優參數
        clahe_tile_size=16,     # 最優參數
        use_gamma=False,
        use_guided_filter=False
    )

    return D, A, BestPsi


# ========== 優化配置預設 ==========

# 配置1: 最佳質量（推薦用於評估）
def defog_best_quality(hazy_image):
    """最佳質量配置"""
    return defog_img_enhanced(
        hazy_image,
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_tile_size=16,
        use_gamma=False,
        use_guided_filter=False
    )


# 配置2: 高對比度（適合霧很濃的場景）
def defog_high_contrast(hazy_image):
    """高對比度配置（濃霧）"""
    return defog_img_enhanced(
        hazy_image,
        t0=0.1,  # 更低的t0，更強的去霧
        use_clahe=True,
        clahe_clip_limit=3.0,  # 更高的對比度
        clahe_tile_size=8,     # 更小的塊，更局部
        use_gamma=False,
        use_guided_filter=True,
        guided_filter_radius=15
    )


# 配置3: 保守去霧（適合霧較淡的場景）
def defog_conservative(hazy_image):
    """保守去霧配置（淡霧）"""
    return defog_img_enhanced(
        hazy_image,
        t0=0.3,  # 更高的t0，更溫和
        use_clahe=True,
        clahe_clip_limit=1.5,  # 較低的對比度增強
        clahe_tile_size=16,
        use_gamma=False,
        use_guided_filter=False
    )


# 配置4: 快速模式（無後處理，最快速度）
def defog_fast(hazy_image):
    """快速模式（無後處理）"""
    return defog_img_basic(hazy_image)[:3]  # 只返回前3個值


# ========== 批量處理和評估 ==========

def batch_process(image_list, config='best'):
    """
    批量處理圖像

    參數:
        image_list: 圖像列表
        config: 配置類型 ('best', 'high_contrast', 'conservative', 'fast')

    返回:
        結果列表
    """
    configs = {
        'best': defog_best_quality,
        'high_contrast': defog_high_contrast,
        'conservative': defog_conservative,
        'fast': defog_fast
    }

    func = configs.get(config, defog_best_quality)
    results = []

    for img in image_list:
        result = func(img)
        results.append(result)

    return results


if __name__ == "__main__":
    """
    使用示例
    """
    import sys

    if len(sys.argv) > 1:
        # 從命令行讀取圖像
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "dehazed_avsd.png"

        # 讀取圖像（注意轉換為RGB）
        img_bgr = cv2.imread(input_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 去霧（使用最佳質量配置）
        dehazed, A, psi, info = defog_best_quality(img_rgb)

        # 保存結果
        dehazed_bgr = cv2.cvtColor(dehazed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, dehazed_bgr)

        print(f"✓ Dehazing completed!")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")
        print(f"  PSI: {psi:.4f}")
        print(f"  Airlight: [{A[0]:.1f}, {A[1]:.1f}, {A[2]:.1f}]")
        print(f"  CLAHE: {info['used_clahe']}")
        print(f"  Transmission range: [{info['t_min']:.3f}, {info['t_max']:.3f}]")
    else:
        print("Usage: python defog_avsd_v5_enhanced.py <input_image> [output_image]")
        print("\nOptimizations added:")
        print("  ✓ CLAHE post-processing (key improvement)")
        print("  ✓ Gamma correction option")
        print("  ✓ Guided filter option")
        print("  ✓ Multiple preset configurations")
