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


def defog_img(hazy_image, psi=1.25, t0=0.2, window_size=8, epsilon=1e-6):
	
    def to_fx(arr, frac_bits):
        """float → fixed (int64)"""
        return np.round(arr * (1 << frac_bits)).astype(np.int64)

    def from_fx(arr, frac_bits):
        """fixed → float (np.float32)；僅用於除錯輸出"""
        return arr.astype(np.float32) / (1 << frac_bits)

    def clip_u(arr, bitwidth):
        """無號飽和 (0 … 2^bitwidth‑1)"""
        return np.clip(arr, 0, (1 << bitwidth) - 1).astype(np.int64)

    def nozero(arr):
        return np.where(arr == 0, 1, arr)

    def rshift_round(v,s):
        return (v + (1<<(s-1))) >> s  # 四捨五入右移
    
    H = hazy_image.astype(np.int64)

    H_ds = H[::2, ::2, :]

    """ ALE module """

    ''' 計算大氣光A '''
    # 計算下採樣圖像的暗通道：對每個像素在窗口內取三個通道的最小值，然後再做最小濾波
    dark_img = np.min(H_ds, axis=2)
    dark_patch = minimum_filter(dark_img, size=window_size)
    # 選擇暗通道中最大值對應的像素作為大氣光 A（從下採樣圖像中取得）
    idx = np.argmax(dark_patch)
    y, x = np.unravel_index(idx, dark_patch.shape)
    A = H_ds[y, x, :]  # 大氣光向量

    int_K_Hn3_rec   = 6         # 6 # dont change
    int_c6_lut      = 8        # 12 # dont change
    int_trec        = 4         # 4 # dont change

    frac_A_rec      = 12        # 12
    frac_H_norm     = 8         # 8 # frac_H_norm <= frac_A_rec
    frac_K_Hn3_rec  = 6         # 6
    frac_S_Hn       = 12        # 12 # frac_S_Hn <= (frac_H_norm + frac_K_Hn3_rec)
    frac_w          = 10        # 10
    frac_wK_Hn      = 9         # 9 # frac_wK_Hn <= (frac_H_norm + frac_w)
    frac_c6_lut     = 4         # 0
    frac_trec       = 8         # 8 # frac_trec <= frac_S_Hn + frac_c6_lut
    
    # # high fixed
    # frac_A_rec      = 14
    # frac_H_norm     = 10
    # frac_K_Hn3_rec  = 8
    # frac_S_Hn       = 14
    # frac_w          = 10
    # frac_wK_Hn      = 12
    # frac_c6_lut     = 4
    # frac_trec       = 12

    """ INSE module """
    #--- 計算 H_norm ---------------------------------------------------
    H_norm = np.empty_like(H, dtype=np.int64)
    A_rec = clip_u(to_fx(1.0 / A, frac_A_rec), frac_A_rec)       # Q0.12

    for c in range(3):
        H_norm[:, :, c] = (H[:, :, c] * A_rec[c])  # Q2.12

    H_norm = rshift_round(H_norm, frac_A_rec - frac_H_norm)  # Q2.8 # frac_H_norm
    #==========================  stage 2  ===================================
    c2_add = H_norm[:,:,0] + H_norm[:,:,1]          # Q3.8 # frac_H_norm
    min_Hn = np.min(H_norm, axis=2)                 # Q2.8 # frac_H_norm


    #==========================  stage 3  ===================================
    K_Hn3 = c2_add + H_norm[:,:,2]       # Q4.8 # frac_H_norm

    K_Hn3_rec = clip_u(to_fx(1.0 / from_fx(nozero(K_Hn3), frac_H_norm), frac_K_Hn3_rec), frac_K_Hn3_rec + int_K_Hn3_rec) # Q6.6
    c3_mul0 = min_Hn * 3                # Q4.8 # frac_H_norm
    c3_sub = K_Hn3 - c3_mul0            # Q4.8 # frac_H_norm

    # frac_H_norm * frac_K_Hn3_rec = 8 + 6 = 14
    S_Hn = clip_u(rshift_round(c3_sub * K_Hn3_rec, frac_H_norm + frac_K_Hn3_rec - frac_S_Hn), frac_S_Hn)    # Q10.14 => Qx.12
    S_Hn = np.clip(S_Hn, 1 << 6, a_max=None)

    #==========================  stage 4  ===================================
    CONST2 = to_fx(2, frac_S_Hn) - 1       # Q1.12 # frac_S_Hn
    c4_sub = CONST2 ^ S_Hn                 # Q1.12 # frac_S_Hn
    S_Dn = rshift_round(S_Hn * c4_sub, frac_S_Hn)    # Q0.12 # frac_S_Hn
    # wK_Hn = (K_Hn3 * 1.25 / 3)
    wK_Hn = rshift_round(K_Hn3 * to_fx(psi/3, frac_w), frac_H_norm + frac_w - frac_wK_Hn)      # Q4.8 * Q0.10 = Q4.18 => Q3.9


    """ TCAL module """
    #==========================  stage 5  ===================================

    c5_sub0 = S_Dn - S_Hn               # Q0.12 # frac_S_Hn
    c5_mul = (wK_Hn * c5_sub0)          # Q3.21 # frac_S_Hn + frac_wK_Hn

    c5_sub1 = np.clip(S_Dn - (rshift_round(c5_mul, frac_wK_Hn)), 0, (2**frac_S_Hn) - 1) # Q0.12 # frac_S_Hn

    #==========================  stage 6  ===================================

    c6_lut = clip_u(to_fx(1.0 / from_fx(nozero(c5_sub1), frac_S_Hn), frac_c6_lut), int_c6_lut + frac_c6_lut)       # Q12.0
    c6_mul = S_Dn * c6_lut                  # Q12.12 # frac_S_Hn + frac_c6_lut
    # trec = clip_u((c6_mul >> 4), 12)        # Q4.8
    trec = np.clip(rshift_round(c6_mul, frac_S_Hn + frac_c6_lut - frac_trec), 2**frac_trec, 2**(frac_trec + int_trec)) # trec range 1~16


    """ RESTORE module """

    t_expanded = trec[:, :, np.newaxis] # Q4.8 # frac_trec
    HA = H - A

    c7_mul = HA * t_expanded            # Q12.8 # frac_trec

    c7_add = rshift_round(c7_mul, frac_trec) + A

    clip_D = clip_u(c7_add, 8).astype(np.uint8)
	
    return clip_D, A