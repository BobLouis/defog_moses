"""
Method 2: Bilateral filter ONLY for transmission refinement.
Other box filters keep large r for color stability.
+ Cross-channel deviation clamp to prevent color fringing.
"""
import numpy as np
import cv2
from scipy.ndimage import minimum_filter
import os, time

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

def bilateral_smooth(src, d=9, sigma_color=75, sigma_space=75):
    src_f = np.clip(src.astype(np.float32), 0, 1)
    src_u8 = (src_f * 255).astype(np.uint8)
    out_u8 = cv2.bilateralFilter(src_u8, d, sigma_color, sigma_space)
    return out_u8.astype(np.float32) / 255.0

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

def dehaze_m2(I_bgr, omega_base, rgb_scale=(0.80, 0.93, 1.05),
              r_base=60, t0=0.1,
              alpha_clear=0.30, power=3, omega_boost=0.10,
              sat_scale=0.04, sat_power=3,
              bf_d=9, bf_sc=75, bf_ss=75, max_dev=0.05):
    """
    Only transmission refinement uses bilateral filter (edge-preserving).
    Other 3 box filter locations keep box_smooth(r=r_base) for color stability.
    Cross-channel clamp (max_dev) prevents color fringing at edges.
    """
    I = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    Ag = airlight_topk(I)

    haze_density = compute_haze_density(I, Ag, r=r_base)

    hsv = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1].astype(np.float32) / 255.0
    sat_smooth = box_smooth(sat, r=r_base)

    # First pass: compute per-channel transmissions
    t_channels = []
    dc_channels = []
    for c, scale_c in enumerate(rgb_scale):
        omega_c = float(np.clip(omega_base * scale_c, 0.10, 0.95))
        norm_c = I[:,:,c] / (Ag[c]+eps)
        dc_c = minimum_filter(norm_c, size=15)
        dc_channels.append(dc_c)

        omega_local = np.clip(omega_c * (1.0 + omega_boost * haze_density), 0.10, 0.95)
        t_c_raw = np.clip(1 - omega_local * dc_c, 0.05, 1.0)
        t_c_ref = bilateral_smooth(t_c_raw, d=bf_d, sigma_color=bf_sc, sigma_space=bf_ss)
        t_c = np.maximum(t_c_ref, t0)
        t_channels.append(t_c)

    # Cross-channel clamp
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

    print("=== Method 2: Bilateral + Cross-Channel Clamp ===")
    print(f"Original box r=60: PSNR={p_orig:.2f}")
    print(f"\n{'d':>4} {'sc':>6} {'ss':>6} {'mdev':>6} {'PSNR':>8} {'SSIM':>8} {'dPSNR':>7}")
    print("-" * 52)

    best_psnr = 0
    best_cfg = (9, 75, 75, 0.05)
    best_ssim = 0
    configs = [
        (9, 50, 50),   (9, 75, 75),   (9, 100, 100),
        (15, 50, 50),  (15, 75, 75),  (15, 100, 100),
        (21, 75, 75),  (21, 100, 100),
        (31, 75, 75),  (31, 100, 100),
    ]

    for d, sc, ss in configs:
        for mdev in [0.04, 0.05, 0.06, 0.08]:
            t0_s = time.time()
            J = dehaze_m2(I_bgr, omega_base, bf_d=d, bf_sc=sc, bf_ss=ss, max_dev=mdev)
            elapsed = time.time() - t0_s
            p = psnr(J, G)
            s = ssim_val(J, G)
            dp = p - p_orig
            mark = " *" if p > best_psnr else ""
            print(f"{d:>4} {sc:>6} {ss:>6} {mdev:>6.2f} {p:>8.2f} {s:>8.4f} {dp:>+7.2f}{mark}", flush=True)
            if p > best_psnr:
                best_psnr = p
                best_cfg = (d, sc, ss, mdev)
                best_ssim = s

    bd, bsc, bss, bmd = best_cfg
    print(f"\nBest: d={bd} sc={bsc} ss={bss} max_dev={bmd} PSNR={best_psnr:.2f}")

    # Fine-tune
    print("Fine-tuning...")
    for dd in [max(3, bd-4), max(3, bd-2), bd, bd+2, bd+4]:
        for dsc in [max(20, bsc-25), bsc, bsc+25]:
            for mdev in [max(0.02, bmd-0.01), bmd, bmd+0.01]:
                J = dehaze_m2(I_bgr, omega_base, bf_d=dd, bf_sc=dsc, bf_ss=bss, max_dev=mdev)
                p = psnr(J, G)
                if p > best_psnr:
                    best_psnr = p
                    best_cfg = (dd, dsc, bss, mdev)
                    best_ssim = ssim_val(J, G)

    d, sc, ss, mdev = best_cfg
    print(f"Final: d={d} sc={sc} ss={ss} max_dev={mdev:.3f} PSNR={best_psnr:.2f} SSIM={best_ssim:.4f} (delta={best_psnr-p_orig:+.2f})")

    J_best = dehaze_m2(I_bgr, omega_base, bf_d=d, bf_sc=sc, bf_ss=ss, max_dev=mdev)
    out_path = os.path.join(OUT_DIR, '18_m2_bilateral.png')
    cv2.imwrite(out_path, J_best)
    print(f"Saved: {out_path}")
