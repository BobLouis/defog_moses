# defog_2023.py

import numpy as np
from scipy.ndimage import minimum_filter

""" 公式整理
H(x) = D(x)*t(x) + A*(1-t(x))
D(x) = ((H(x) - A) / t(x)) + A
=================================================================
A 從暗通道中選擇最亮的像素作為 A = (Ar, Ag, Ab) (patch = 15*15, downsample 2)
t(x) = 1 - w * (K_H(x) / A) * (1 - S_H(x)/S_D(x))
=================================================================
which
S(x) 為飽和度, K(x) 為像素強度值
w = 1.25
K_H(x) = Hr(x) + Hg(x) + Hb(x) / 3
	=> (K_H(x) / A) = (Hr(x)/Ar + Hg(x)/Ag + Hb(x)/Ab) / 3
	=> let H_norm[] = [Hr(x)/Ar, Hg(x)/Ag, Hb(x)/Ab]
	=> (K_H(x) / A) = avg(H_norm)
S_D(x) = S_H(x) * (2 - S_H(x))
S_H(x) = 1 - (min_c(H_c(x)) / K_H(x)), which c is rgb
=================================================================
	=> t(x) = 1 - w * (K_H(x) / A) * (1 - 1/(2 - S_H(x)) )
"""

import numpy as np

def predict_psi(image):
    """
    Fog estimation - 方案B（超省算力：16-bin 直方圖 + LUT）
    核心：用稀疏取樣 + 小bins直方圖取得全域對比/分散度特徵，查表(分段線性)映射到 FogScore，再線性到 psi。

    步驟（全整數/低成本友好）：
      1) 取 uint8，stride 取樣（預設 8），只用 G 通道。
      2) 建 16-bin 直方圖（pixel >> 4），累積為 CDF。
      3) 從 CDF 取 P5 / P50 / P95（以 bin 中心近似 0..255 亮度）。
      4) 特徵：
          - dyn_range = P95 - P5（對比範圍；霧重時偏低）
          - mad_bin   = sum(|bin_center - P50| * count) / N（分散度近似；霧重時偏小）
          - eta_high  = 高亮比例（bins >= 12）；用作極端白霧的保險觸發
      5) FogScore = 0~100：70% 由 dyn_range，30% 由 mad_bin（皆分段線性映射）
      6) psi = 0.011099 * FogScore + 0.746386，再夾限到 [0.80, 1.30]

    備註：閾值沿用你原本 FogScore 的標準（range: 100~240、MAD: 20~60），
          以維持和舊版行為的一致性；若資料分布差異較大，可微調。
    """
    # ---------- 1) 準備資料：uint8 + stride 取樣 + 單通道 ----------
    if image.dtype != np.uint8:
        img = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img = image

    if img.ndim == 3 and img.shape[2] >= 2:
        # 取 G 通道（更便宜）；若想更穩可改成 max(R,G,B)
        g_full = img[..., 1]
    else:
        # 灰階圖直接用
        g_full = img

    stride = 8  # 可調：4/8/16；越大越省
    g = g_full[::stride, ::stride]
    if g.size == 0:
        return float(1.0)  # 退化保守值

    # ---------- 2) 16-bin 直方圖（位移代替除法） ----------
    bins = (g >> 4).ravel()
    hist = np.bincount(bins, minlength=16).astype(np.int64)
    cdf = np.cumsum(hist)
    N = int(cdf[-1]) if cdf.size else 0
    if N == 0:
        return float(1.0)

    # bin 中心（0..15 -> 8, 24, ..., 248）
    bin_centers = (np.arange(16, dtype=np.float32) * 16.0 + 8.0)

    def percentile_from_cdf(cdf_arr, p):
        """在 16-bin CDF 上近似百分位，回傳對應 bin 的中心值"""
        thr = (p / 100.0) * N
        idx = int(np.searchsorted(cdf_arr, thr, side='left'))
        idx = 0 if idx < 0 else (15 if idx > 15 else idx)
        return float(bin_centers[idx])

    # ---------- 3) 取 P5 / P50 / P95 ----------
    p5   = percentile_from_cdf(cdf, 5.0)
    p50  = percentile_from_cdf(cdf, 50.0)
    p95  = percentile_from_cdf(cdf, 95.0)

    # ---------- 4) 特徵：對比範圍 + 分散度近似 + 高亮比例 ----------
    dyn_range = p95 - p5  # 霧重 => 小
    # MAD' 近似：以 bin 中心對 p50 的絕對差，加權平均
    abs_diff = np.abs(bin_centers - p50)  # shape (16,)
    mad_bin = float(np.dot(abs_diff, hist) / max(N, 1))  # 霧重 => 小
    # 高亮比例（極端白霧保險：若 range 很小且高亮很多，偏向增強去霧）
    eta_high = float(hist[12:].sum()) / float(N)

    # ---------- 5) FogScore（0~100）：70% range + 30% MAD'（分段線性） ----------
    # 與你原版保持一致的映射：range>=240 => 0 分；range<=100 => 100 分
    if dyn_range >= 240.0:
        score_range = 0.0
    elif dyn_range <= 100.0:
        score_range = 100.0
    else:
        score_range = 100.0 - ((dyn_range - 100.0) / 140.0) * 100.0

    # MAD'：>=60 => 0 分；<=20 => 100 分；中間線性
    if mad_bin >= 60.0:
        score_mad = 0.0
    elif mad_bin <= 20.0:
        score_mad = 100.0
    else:
        score_mad = 100.0 - ((mad_bin - 20.0) / 40.0) * 100.0

    fog_score = 0.7 * score_range + 0.3 * score_mad

    # 極端白霧保險：對比很小而且高亮比例很大時，略微增加去霧力度（加分降低 psi 前的波動）
    # 例：dyn_range < 15 & eta_high > 0.6 -> +5 分（夾限到 0..100）
    if dyn_range < 15.0 and eta_high > 0.60:
        fog_score = min(100.0, fog_score + 5.0)

    fog_score = float(np.clip(fog_score, 0.0, 100.0))

    # ---------- 6) FogScore -> psi（線性 + 夾限） ----------
    psi = 0.011099 * fog_score + 0.746386
    psi = float(np.clip(psi, 0.80, 1.60))  # 依資料可改 0.75~1.35 等

    return psi




def defog_img(hazy_image, psi=1, t0=0.2, window_size=8, epsilon=1e-6):
	"""
	基於論文方法對輸入的 hazy 圖像進行去霧處理，返回無霧圖像、暗通道圖像、大氣光和傳輸圖。
	參數:
	hazy_image: 輸入圖像（RGB，np.uint8）
	psi: 擬合係數（論文中固定為 1.25）
	t0: 傳輸圖的下界（例如 0.2）
	window_size: 最小濾波器窗口大小（15x15）
	epsilon: 防止除零的小常數
	返回:
	D: 去霧後的圖像（np.uint8）
	dark_output: 暗通道圖像（灰階複製為三通道，np.uint8）
	A: 大氣光向量（3,）
	t: 傳輸圖（float32）
	"""
	# 將輸入轉換為 float 型態以便計算
	H = hazy_image.astype(np.float32)
	
	# 根據論文描述，先對 hazy 圖像進行下採樣（因子為 2），用於大氣光 A 的估計
	H_ds = H[::2, ::2, :]
	
	# 計算下採樣圖像的暗通道：對每個像素在窗口內取三個通道的最小值，然後再做最小濾波
	dark_channel_ds = minimum_filter(np.min(H_ds, axis=2), size=window_size)
	
	# 選擇暗通道中最大值對應的像素作為大氣光 A（從下採樣圖像中取得）
	idx = np.argmax(dark_channel_ds)
	y, x = np.unravel_index(idx, dark_channel_ds.shape)
	A = H_ds[y, x, :]  # 大氣光向量

	# Calculate optimal PSI based on fog estimation
	psi = predict_psi(hazy_image)
	
	# 使用原始全解析度圖像進行後續處理：對每個通道進行歸一化(除以 A)
	H_norm = np.empty_like(H, dtype=np.float32)
	for c in range(3):
		H_norm[:, :, c] = H[:, :, c] / (A[c] + epsilon)
	
	# 計算歸一化圖像的平均強度 K（每個像素的均值）
	K = np.mean(H_norm, axis=2)
	
	# 計算飽和度 S，公式：S = 1 - (min(R_norm, G_norm, B_norm) / (K + epsilon))
	min_norm = np.min(H_norm, axis=2)

	temp = 3*K + 3*min_norm
	t = (temp - psi*3*K*min_norm) / (temp + epsilon)
	# 限制傳輸圖的下界
	t = np.clip(t, t0, 1)
	
	# 利用傳輸圖恢復無霧圖像： D(x) = (H(x) - A) / t(x) + A
	t_expanded = t[:, :, np.newaxis]
	D = (H - A) / t_expanded + A
	D = np.clip(D, 0, 255).astype(np.uint8)

	return D, A