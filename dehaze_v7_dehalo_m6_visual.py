"""Generate M6 results at different max_dev values for visual comparison."""
import numpy as np
import cv2
from scipy.ndimage import minimum_filter
import os

eps = 1e-6

def psnr(a, b):
    mse = np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
    return 10*np.log10(255.0**2/mse) if mse > 1e-10 else 999.0

def ssim_val(a, b):
    from skimage.metrics import structural_similarity
    return np.mean([structural_similarity(
        a[:,:,c].astype(np.float64)/255, b[:,:,c].astype(np.float64)/255,
        data_range=1.0) for c in range(3)])

def box_smooth(src, r=60):
    ks = 2 * r + 1
    return cv2.blur(src.astype(np.float32), (ks, ks)).astype(np.float32)

def airlight_topk(I, k=0.001, ps=15):
    dp = minimum_filter(np.min(I, axis=2), size=ps)
    n = max(1, int(I.shape[0]*I.shape[1]*k))
    fi = dp.flatten(); fI = I.reshape(-1,3)
    top = np.argpartition(fi,-n)[-n:]
    return fI[top[np.argmax(np.max(fI[top], axis=1))]]

def compute_haze_density(I, Ag, r=60):
    dc_global = minimum_filter(np.min(I, axis=2), size=15)
    Ag_max = np.max(Ag)
    haze_density = np.clip(dc_global / (Ag_max + eps), 0, 1)
    haze_smooth = box_smooth(haze_density, r=r)
    return np.clip(haze_smooth, 0, 1)

def dehaze_m6(I_bgr, omega_base, rgb_scale=(0.80, 0.93, 1.05),
              r_base=60, r_trans=9, t0=0.1,
              alpha_clear=0.30, power=3, omega_boost=0.10,
              sat_scale=0.04, sat_power=3, max_dev=0.085):
    I = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    Ag = airlight_topk(I)
    haze_density = compute_haze_density(I, Ag, r=r_base)
    hsv = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1].astype(np.float32) / 255.0
    sat_smooth = box_smooth(sat, r=r_base)

    t_channels = []
    dc_channels = []
    for c, scale_c in enumerate(rgb_scale):
        omega_c = float(np.clip(omega_base * scale_c, 0.10, 0.95))
        norm_c = I[:,:,c] / (Ag[c]+eps)
        dc_c = minimum_filter(norm_c, size=15)
        dc_channels.append(dc_c)
        omega_local = np.clip(omega_c * (1.0 + omega_boost * haze_density), 0.10, 0.95)
        t_c_raw = np.clip(1 - omega_local * dc_c, 0.05, 1.0)
        t_c_ref = box_smooth(t_c_raw, r=r_trans)
        t_c = np.maximum(t_c_ref, t0)
        t_channels.append(t_c)

    t_mean = np.mean(t_channels, axis=0)
    for c in range(3):
        t_channels[c] = np.clip(t_channels[c], t_mean - max_dev, t_mean + max_dev)

    J = np.zeros_like(I)
    for c in range(3):
        dc_smooth = box_smooth(dc_channels[c], r=r_base)
        x = np.clip(1.0 - dc_smooth, 0, 1)
        clear_mod = np.clip(alpha_clear * np.power(x, power), 0, 0.5)
        haze_sat_mod = np.clip(sat_scale * np.power(1.0 - sat_smooth, sat_power), 0, 0.15)
        A_c = Ag[c] * (1.0 - clear_mod - haze_sat_mod)
        J[:,:,c] = np.clip((I[:,:,c] - A_c) / t_channels[c] + A_c, 0, 1)

    return cv2.cvtColor((J*255).astype(np.uint8), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    HAZY  = './dataset/OHaze/hazy/18_hazy.png'
    CLEAR = './dataset/OHaze/clear/18_clear.png'
    OUT_DIR = './dataset/OHaze/result_dehalo_test'
    os.makedirs(OUT_DIR, exist_ok=True)

    I_bgr = cv2.imread(HAZY)
    G = cv2.imread(CLEAR)
    omega_base = 0.85

    from dehaze_v7_3type import dehaze_no_gf
    J_orig = dehaze_no_gf(I_bgr, omega_base)
    p_orig = psnr(J_orig, G)

    print("=== M6 Visual Comparison at Different max_dev ===")
    print(f"Original r=60: PSNR={p_orig:.2f}")
    print()

    configs = [
        (9, 0.02, "tight"),     # tight clamp, should eliminate fringing
        (9, 0.03, "medium"),    # medium clamp
        (9, 0.04, "moderate"),  # moderate
        (9, 0.05, "mild"),      # mild clamp
        (9, 0.085, "best_psnr"), # best PSNR from earlier
    ]

    for r_t, mdev, label in configs:
        J = dehaze_m6(I_bgr, omega_base, r_trans=r_t, max_dev=mdev)
        p = psnr(J, G)
        s = ssim_val(J, G)
        fname = f'18_m6_{label}_mdev{mdev:.3f}.png'
        out_path = os.path.join(OUT_DIR, fname)
        cv2.imwrite(out_path, J)
        print(f"max_dev={mdev:.3f} ({label:>10}): PSNR={p:.2f} SSIM={s:.4f} delta={p-p_orig:+.2f}  -> {fname}")
