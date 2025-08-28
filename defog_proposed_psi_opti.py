# defog_2023_auto_psi.py
import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter

# ======= 你離線擬合出的線性係數（可改） =======
PSI_A = 1.315537
PSI_B = 0.016293
PSI_STEP = 0.02  # 量化格距（硬體友善）

# ======= 一次掃圖的 haze 指標與 ψ 預測 =======
def compute_haze_index(H: np.ndarray) -> float:
    """
    haze_index = (3 * sum(min(R,G,B))) / sum(R+G+B)
    只在影像結尾做一次除法；H: uint8 RGB -> float
    """
    H32 = H.astype(np.uint32)
    minRGB = np.minimum(np.minimum(H32[:, :, 0], H32[:, :, 1]), H32[:, :, 2])
    sumDark = int(minRGB.sum())
    sumI = int(H32[:, :, 0].sum() + H32[:, :, 1].sum() + H32[:, :, 2].sum())
    return (3.0 * sumDark) / (sumI + 1e-9)

def _quantize_to_grid(x: float, step=0.02, lo=1.0, hi=2.0) -> float:
    x = max(lo, min(hi, float(x)))
    k = round((x - lo) / step)
    return round(lo + k * step, 2)

def _predict_psi_from_image(H: np.ndarray, a=PSI_A, b=PSI_B, step=PSI_STEP):
    h = compute_haze_index(H)
    psi_raw = a * h + b
    psi_q = _quantize_to_grid(psi_raw, step=step, lo=1.0, hi=2.0)
    return psi_q, h, psi_raw

# ======= 除霧主函式（內建 ψ 預測，不需傳 psi） =======
def defog_img(hazy_image: np.ndarray,
              t0: float = 0.2,
              window_size: int = 8,
              epsilon: float = 1e-6,
              a: float = PSI_A,
              b: float = PSI_B,
              step: float = PSI_STEP,
              return_meta: bool = False):
    """
    對輸入 hazy 圖做去霧；ψ 由影像自動預測與量化（0.02格）。
    參數:
        hazy_image: (H,W,3) np.uint8
        t0: 傳輸圖下界
        window_size: 估計 A 的最小濾波窗口
        epsilon: 防除零
        a,b,step: ψ 預測模型係數與量化步距
        return_meta: True 時回傳 (D, A, psi_used, haze_index, psi_raw)
    回傳:
        預設: (D, A)
        return_meta=True: (D, A, psi_used, haze_index, psi_raw)
    """
    # ---- 1) 先從影像預測 ψ（一次掃圖）----
    psi_used, haze_idx, psi_raw = _predict_psi_from_image(hazy_image, a=a, b=b, step=step)

    # ---- 2) 照你原法估計 A 與 t，做去霧 ----
    H = hazy_image.astype(np.float32)

    # 下採樣估 A
    H_ds = H[::2, ::2, :]
    dark_channel_ds = minimum_filter(np.min(H_ds, axis=2), size=window_size)
    idx = np.argmax(dark_channel_ds)
    y, x = np.unravel_index(idx, dark_channel_ds.shape)
    A = H_ds[y, x, :]  # (3,)

    # 歸一化
    H_norm = np.empty_like(H, dtype=np.float32)
    for c in range(3):
        H_norm[:, :, c] = H[:, :, c] / (A[c] + epsilon)

    K = np.mean(H_norm, axis=2)
    min_norm = np.min(H_norm, axis=2)

    # 你的傳輸圖公式（把 psi 換成 psi_used）
    temp = 3 * K + 3 * min_norm
    t = (temp - psi_used * 3 * K * min_norm) / (temp + epsilon)
    t = np.clip(t, t0, 1.0)

    # 恢復
    D = (H - A) / (t[:, :, None]) + A
    D = np.clip(D, 0, 255).astype(np.uint8)

    if return_meta:
        return D, A, psi_used, haze_idx, psi_raw
    else:
        return D, A

# ======= 簡單示例 =======
if __name__ == "__main__":
    # 單檔測試：讀一張圖 -> 自動 ψ -> 除霧 -> 存檔
    in_path = "./dataset/SOTS_in/hazy/001.png"
    out_path = "./dataset/SOTS_in/result_proposed_psi=1.5/001_proposed_psi=1.5.png"

    H = np.array(Image.open(in_path).convert('RGB'))
    D, A, psi_used, haze_idx, psi_raw = defog_img(H, return_meta=True)
    Image.fromarray(D).save(out_path)
    print(f"psi_raw={psi_raw:.4f} -> psi_used={psi_used:.2f}, haze_index={haze_idx:.6f}, "
          f"A=({A[0]:.1f},{A[1]:.1f},{A[2]:.1f})")
    print(f"saved: {out_path}")
