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

    def rshift(v,s):
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

    frac_A_rec      = 12        # 12
    frac_H_norm     = 8         # 8 # frac_H_norm <= frac_A_rec


    """ INSE module """
    #--- 計算 H_norm ---------------------------------------------------
    H_norm = np.empty_like(H, dtype=np.int64)
    A_rec = clip_u(to_fx(1.0 / A, frac_A_rec), frac_A_rec)       # Q0.12

    for c in range(3):
        H_norm[:, :, c] = (H[:, :, c] * A_rec[c])  # Q2.12

    # H_norm = rshift(H_norm, frac_A_rec - frac_H_norm)  # Q2.8 # frac_H_norm
    H_norm = H_norm >> (frac_A_rec - frac_H_norm)  # Q2.8 # frac_H_norm
    #==========================  stage 2  ===================================
    c2_add = H_norm[:,:,0] + H_norm[:,:,1]          # Q3.8 # frac_H_norm
    min_Hn = np.min(H_norm, axis=2)                 # Q2.8 # frac_H_norm


    #==========================  stage 3  ===================================
    K_Hn3 = c2_add + H_norm[:,:,2]       # Q4.8 # frac_H_norm
    min_Hn3 = min_Hn * 3				# Q4.8 # frac_H_norm

    #==========================  stage dayum  ===================================
    son = clip_u(K_Hn3 + min_Hn3, 12)	# Q5.8 # frac_H_norm # 再捨棄變成Q4.8
    c3_mul = K_Hn3 * min_Hn				# Q6.16 # frac_H_norm * 2
    km = c3_mul >> 8					# Q4.8
    mom = son - km						# Q4.8 # frac_H_norm

    #==========================  stage 6  ===================================

    c6_lut = clip_u(to_fx(1.0 / from_fx(nozero(mom), frac_H_norm), frac_H_norm), 12)       # Q4.8
    c6_mul = son * c6_lut                  # Q8.16
    # trec = clip_u((c6_mul >> 4), 12)        # Q4.8
    # trec = np.clip(rshift(c6_mul, frac_H_norm), 2**frac_H_norm, 2**12) # trec range 1~16 # Q4.8
    trec = np.clip((c6_mul >> frac_H_norm), 2**frac_H_norm, 2**12) # trec range 1~16 # Q4.8

    """ RESTORE module """

    t_expanded = trec[:, :, np.newaxis] # Q4.8 # frac_trec
    HA = H - A

    c7_mul = HA * t_expanded            # Q12.8 # frac_trec

    # c7_add = rshift(c7_mul, frac_H_norm) + A
    c7_add = (c7_mul >> frac_H_norm) + A

    clip_D = clip_u(c7_add, 8).astype(np.uint8)
	
    return clip_D, A