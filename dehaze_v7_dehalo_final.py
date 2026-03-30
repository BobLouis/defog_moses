"""
Final dehalo: small-r transmission + tight cross-channel clamp (max_dev=0.04)
+ optimize other parameters (alpha_clear, power, omega_boost) to recover PSNR.
"""
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

def dehaze_final(I_bgr, omega_base, rgb_scale=(0.80, 0.93, 1.05),
                 r_base=60, r_trans=9, t0=0.1,
                 alpha_clear=0.30, power=3, omega_boost=0.10,
                 sat_scale=0.04, sat_power=3, max_dev=0.04):
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

    print("=== Final: Optimize all params with tight clamp ===")
    print(f"Original r=60: PSNR={p_orig:.2f}")

    # Baseline with max_dev=0.04, default params
    J_base = dehaze_final(I_bgr, omega_base, max_dev=0.04)
    p_base = psnr(J_base, G)
    print(f"Baseline (max_dev=0.04, default params): PSNR={p_base:.2f}")

    # Optimize rgb_scale (the main driver of channel divergence)
    print("\n--- Optimizing rgb_scale ---")
    best_psnr = 0
    best_rgb = (0.80, 0.93, 1.05)

    for r_s in [0.75, 0.80, 0.85, 0.90]:
        for g_s in [0.90, 0.93, 0.95, 1.00]:
            for b_s in [1.00, 1.05, 1.10]:
                J = dehaze_final(I_bgr, omega_base, rgb_scale=(r_s, g_s, b_s), max_dev=0.04)
                p = psnr(J, G)
                if p > best_psnr:
                    best_psnr = p
                    best_rgb = (r_s, g_s, b_s)
                    print(f"  rgb=({r_s},{g_s},{b_s}): PSNR={p:.2f} *")

    print(f"Best rgb_scale: {best_rgb} PSNR={best_psnr:.2f}")

    # Optimize alpha_clear, power with best rgb_scale
    print("\n--- Optimizing alpha_clear, power ---")
    for ac in [0.20, 0.25, 0.30, 0.35, 0.40]:
        for pw in [2, 3, 4]:
            J = dehaze_final(I_bgr, omega_base, rgb_scale=best_rgb, max_dev=0.04,
                             alpha_clear=ac, power=pw)
            p = psnr(J, G)
            if p > best_psnr:
                best_psnr = p
                best_ac, best_pw = ac, pw
                print(f"  ac={ac} pw={pw}: PSNR={p:.2f} *")
            else:
                best_ac, best_pw = 0.30, 3  # keep defaults if no improvement

    # Optimize omega_boost, sat_scale
    print("\n--- Optimizing omega_boost, sat_scale ---")
    best_ob, best_ss = 0.10, 0.04
    for ob in [0.05, 0.10, 0.15, 0.20]:
        for ss in [0.02, 0.04, 0.06]:
            J = dehaze_final(I_bgr, omega_base, rgb_scale=best_rgb, max_dev=0.04,
                             alpha_clear=best_ac, power=best_pw,
                             omega_boost=ob, sat_scale=ss)
            p = psnr(J, G)
            if p > best_psnr:
                best_psnr = p
                best_ob, best_ss = ob, ss
                print(f"  ob={ob} ss={ss}: PSNR={p:.2f} *")

    # Optimize max_dev and r_trans with all best params
    print("\n--- Fine-tuning max_dev + r_trans ---")
    best_mdev, best_rt = 0.04, 9
    for rt in [5, 7, 9, 11, 13]:
        for mdev in [0.03, 0.035, 0.04, 0.045, 0.05]:
            J = dehaze_final(I_bgr, omega_base, rgb_scale=best_rgb, max_dev=mdev,
                             r_trans=rt, alpha_clear=best_ac, power=best_pw,
                             omega_boost=best_ob, sat_scale=best_ss)
            p = psnr(J, G)
            if p > best_psnr:
                best_psnr = p
                best_mdev, best_rt = mdev, rt
                print(f"  rt={rt} mdev={mdev}: PSNR={p:.2f} *")

    print(f"\n=== Final Result ===")
    print(f"rgb_scale={best_rgb}, r_trans={best_rt}, max_dev={best_mdev}")
    print(f"alpha_clear={best_ac}, power={best_pw}, omega_boost={best_ob}, sat_scale={best_ss}")
    print(f"PSNR={best_psnr:.2f} (delta={best_psnr-p_orig:+.2f})")

    J_final = dehaze_final(I_bgr, omega_base, rgb_scale=best_rgb, max_dev=best_mdev,
                            r_trans=best_rt, alpha_clear=best_ac, power=best_pw,
                            omega_boost=best_ob, sat_scale=best_ss)
    s_final = ssim_val(J_final, G)
    print(f"SSIM={s_final:.4f}")

    out_path = os.path.join(OUT_DIR, '18_final_dehalo.png')
    cv2.imwrite(out_path, J_final)
    print(f"Saved: {out_path}")
