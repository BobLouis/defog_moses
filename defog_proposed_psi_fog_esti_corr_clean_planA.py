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
    Fog estimation (方案A：下採樣 + 低成本統計) -> 自適應 psi
    - 重用下採樣觀念（stride=2）
    - 灰階採 max(R,G,B) 以增加穩定度
    - 動態範圍：以 16-bin 直方圖近似 P95-P5（抗雜訊）
    - 平均偏差：區塊均值的 MAD（8x8 block）
    - 邊緣/細節：行列抽樣差分（每4列/4行）
    - FogScore → psi 線性映射（並做適度夾限）
    """
    # --- 1) 取 uint8 並下採樣 ---
    if image.dtype != np.uint8:
        img = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img = image
    H_ds = img[::2, ::2, :]  # 下採樣 1/4 像素

    # --- 2) 灰階：max(R,G,B) ---
    g = np.maximum.reduce([H_ds[..., 0], H_ds[..., 1], H_ds[..., 2]])

    # --- 3) 以 16-bin 直方圖近似 P5/P95（更抗雜訊、低成本） ---
    # bins: 0..15 代表 [0,16), [16,32), ...
    bins = (g >> 4).ravel()
    hist = np.bincount(bins, minlength=16).astype(np.int64)
    cdf = np.cumsum(hist)
    N = cdf[-1] if cdf.size else 1

    def percentile_from_hist(cdf, hist, p):
        """在 16-bin 直方圖上近似百分位，回傳該 bin 的中心值"""
        if N <= 0:
            return 0.0
        thr = (p / 100.0) * N
        idx = int(np.searchsorted(cdf, thr, side='left'))
        idx = min(max(idx, 0), 15)
        # 取該 bin 的中心值（近似原像素亮度）
        return idx * 16 + 8.0

    p5  = percentile_from_hist(cdf, hist, 5.0)
    p95 = percentile_from_hist(cdf, hist, 95.0)
    dyn_range = float(p95 - p5)  # 對比近似範圍

    # --- 4) 區塊 MAD（以區塊均值近似），block=8x8 ---
    bh, bw = 8, 8
    H, W = g.shape
    Hc = (H // bh) * bh
    Wc = (W // bw) * bw
    if Hc == 0 or Wc == 0:
        block_mad = float(np.mean(np.abs(g.astype(np.float32) - float(np.mean(g)))))
    else:
        gc = g[:Hc, :Wc].astype(np.float32)
        # 轉成 (n_block_h, bh, n_block_w, bw)
        gh = Hc // bh
        gw = Wc // bw
        blocks = gc.reshape(gh, bh, gw, bw)
        block_means = blocks.mean(axis=(1,3), keepdims=True)  # 形狀 (gh,1,gw,1)
        mad_blocks = np.abs(blocks - block_means).mean(axis=(1,3))  # (gh,gw)
        block_mad = float(mad_blocks.mean())

    # --- 5) 抽樣邊緣（每4列/4行差分） ---
    # 水平差分
    rows = g[::4, :]
    if rows.shape[1] > 1:
        diff_h = np.abs(rows[:, 1:].astype(np.int16) - rows[:, :-1].astype(np.int16))
        avg_diff_h = float(diff_h.mean())
    else:
        avg_diff_h = 0.0
    # 垂直差分
    cols = g[:, ::4]
    if cols.shape[0] > 1:
        diff_v = np.abs(cols[1:, :].astype(np.int16) - cols[:-1, :].astype(np.int16))
        avg_diff_v = float(diff_v.mean())
    else:
        avg_diff_v = 0.0
    avg_local_diff = 0.5 * (avg_diff_h + avg_diff_v)

    # --- 6) 三項指標 → FogScore（延用你原本的分段與權重 50/25/25） ---
    # 動態範圍（越小代表霧越重）：閾值沿用原版
    if dyn_range >= 240:
        score_range = 0.0
    elif dyn_range <= 100:
        score_range = 100.0
    else:
        score_range = 100.0 - ((dyn_range - 100.0) / 140.0) * 100.0

    # 區塊 MAD（越小代表霧越重）
    if block_mad >= 60.0:
        score_mad = 0.0
    elif block_mad <= 20.0:
        score_mad = 100.0
    else:
        score_mad = 100.0 - ((block_mad - 20.0) / 40.0) * 100.0

    # 抽樣邊緣（越小代表霧越重）
    if avg_local_diff >= 10.0:
        score_edge = 0.0
    elif avg_local_diff <= 1.0:
        score_edge = 100.0
    else:
        score_edge = 100.0 - ((avg_local_diff - 1.0) / 9.0) * 100.0

    fog_score = (2.0 * score_range + score_mad + score_edge) / 4.0
    fog_score = float(np.clip(fog_score, 0.0, 100.0))

    # --- 7) FogScore -> psi（線性映射 + 合理夾限以增加穩定） ---
    psi = 0.011099 * fog_score + 0.746386
    psi = float(np.clip(psi, 0.80, 1.30))  # 可依資料調整夾限

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