import numpy as np
from scipy.ndimage import minimum_filter

def defog_img(hazy_image, window_size=15, t0=0.1, beta=1.0, epsilon=1e-6):
    """
    論文方法：SLAEM + DMTME + SRM
    參數:
        hazy_image: (H,W,3) uint8 RGB
        window_size: 暗通道最小濾波視窗 (建議 15)
        t0: 傳輸圖下界 (對應式(5)中的 t0，常見 0.1~0.2)
        beta: 介質消光係數 β (式(13))，可依能見度調整
        epsilon: 避免除零
    回傳:
        D: 去霧後影像 (uint8)
        A: 全域空氣光向量 (float32, 長度3，0~255)
    依據論文：
      - 全域空氣光 A_global：由暗通道最大值像素取得 (式(6))
      - 局部空氣光 A_local：SLAEM (式(7), (10), (12))
      - 透射率 t：DMTME，d = θ0 + θ1*V + θ2*S；t = exp(-β d) (式(14), (13))
      - 復原：J = (I - A_local)/max(t,t0) + A_local (式(5))
    """
    # ------------------------------------------------------------
    # 0) 預處理：轉 float、正規化到 [0,1]
    # ------------------------------------------------------------
    Iu8 = hazy_image
    H, W = Iu8.shape[:2]
    I = Iu8.astype(np.float32) / 255.0  # I(z) ∈ [0,1]

    # 分離通道
    R = I[:, :, 0]
    G = I[:, :, 1]
    B = I[:, :, 2]

    # ------------------------------------------------------------
    # 1) 暗通道 & 全域空氣光 A_global (式(6))
    #    暗通道：先對每像素取 min(R,G,B)，再以 window_size 做最小濾波
    # ------------------------------------------------------------
    per_pixel_min = np.minimum(np.minimum(R, G), B)
    dark_channel = minimum_filter(per_pixel_min, size=window_size)

    # 取暗通道最大值位置，以原圖該像素的 RGB 作為 A_global
    idx = np.argmax(dark_channel)
    y, x = np.unravel_index(idx, dark_channel.shape)
    A_global = I[y, x, :].copy()  # in [0,1], shape (3,)

    # ------------------------------------------------------------
    # 2) SLAEM：飽和度式推導的局部空氣光 A_local (式(7), (10), (12))
    #    I_int(z) = (R+G+B)/3   (式(10))
    #    I_dark(z) = min(R,G,B)（像素級）
    #    Isat(z) = I_dark / (I_int + I_dark) (式(12))
    #    A_local(z) = A_global * [ Isat(z) - ((I_int - I_dark)*(I_dark + I_int))/Isat(z) ] (式(7))
    #    ※ 這裡係數為逐像素的純量，乘上向量 A_global → 得到每像素的 RGB 向量 A_local
    # ------------------------------------------------------------
    I_int = (R + G + B) / 3.0
    I_dark_pix = per_pixel_min  # 不再做視窗，依論文推導採像素級

    Isat = I_dark_pix / (I_int + I_dark_pix + epsilon)  # (12)

    # 依式(7)的純量係數（避免數值發散，做輕量裁切）
    coef = Isat - ((I_int - I_dark_pix) * (I_dark_pix + I_int)) / (Isat + epsilon)
    # 適度限制係數範圍，避免極端像素造成過度放大/反相（可視資料集微調）
    coef = np.clip(coef, 0.0, 2.0)

    # 廣播 A_local：每像素 scalar * 向量 A_global
    A_local = np.dstack([coef * A_global[0],
                         coef * A_global[1],
                         coef * A_global[2]])  # shape (H,W,3), ∈ [0, ~2]

    # ------------------------------------------------------------
    # 3) DMTME：深度與透射率 (式(14) → 式(13))
    #    SI(z) = 1 - min/max ； VI(z) = max
    #    d(z) = θ0 + θ1*VI + θ2*SI
    #    t(z) = exp(-β d(z))
    #    係數採論文引用之最佳化值：θ0=0.12, θ1=0.96, θ2=-0.78
    # ------------------------------------------------------------
    I_min = per_pixel_min
    I_max = np.maximum(np.maximum(R, G), B)

    SI = 1.0 - (I_min / (I_max + epsilon))  # (15)
    VI = I_max                               # (16)

    theta0, theta1, theta2 = 0.12, 0.96, -0.78  # (14) 引用值
    d = theta0 + theta1 * VI + theta2 * SI     # (14)

    t = np.exp(-beta * d).astype(np.float32)   # (13)
    t = np.clip(t, t0, 1.0)                    # (5) 的下界保護

    # ------------------------------------------------------------
    # 4) 場景復原 SRM (式(5))：J = (I - A_local)/t + A_local
    # ------------------------------------------------------------
    t3 = t[:, :, None]  # 擴展至三通道
    J = (I - A_local) / t3 + A_local
    J = np.clip(J, 0.0, 1.0)

    # 轉回 uint8；A_global 回傳 0~255 向量，便於與你原流程一致
    D = (J * 255.0 + 0.5).astype(np.uint8)
    A_vec_255 = (A_global * 255.0).astype(np.float32)

    return D, A_vec_255
