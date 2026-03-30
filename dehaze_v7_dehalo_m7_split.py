"""
Method 7: Split transmission into unified edge structure + smooth per-channel offset.
- Unified t from true DCP dark channel, refined with small r (sharp edges, no fringing)
- Per-channel offset = (per_ch_t_raw - unified_t_raw), smoothed with large r (no edge artifacts)
- Final: t_c = unified_sharp + smooth_per_channel_offset
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

def dehaze_m7(I_bgr, omega_base, rgb_scale=(0.80, 0.93, 1.05),
              r_base=60, r_trans=10, t0=0.1,
              alpha_clear=0.30, power=3, omega_boost=0.10,
              sat_scale=0.04, sat_power=3,
              delta_scale=1.0):
    """
    Split t = unified_sharp + smooth_per_channel_delta.
    delta_scale: scaling for per-channel offset (0=fully unified, 1=full offset)
    """
    I = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    Ag = airlight_topk(I)

    haze_density = compute_haze_density(I, Ag, r=r_base)

    hsv = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1].astype(np.float32) / 255.0
    sat_smooth = box_smooth(sat, r=r_base)

    # Unified dark channel (min across all normalized channels)
    omega_mean = float(np.clip(omega_base * np.mean(rgb_scale), 0.10, 0.95))
    norm_all = np.stack([I[:,:,c] / (Ag[c]+eps) for c in range(3)], axis=2)
    dc_unified = minimum_filter(np.min(norm_all, axis=2), size=15)

    # Unified omega_local
    omega_local_u = np.clip(omega_mean * (1.0 + omega_boost * haze_density), 0.10, 0.95)
    t_unified_raw = np.clip(1 - omega_local_u * dc_unified, 0.05, 1.0)
    # Sharp refinement with small r
    t_unified_ref = box_smooth(t_unified_raw, r=r_trans)

    J = np.zeros_like(I)
    dc_channels = []
    for c, scale_c in enumerate(rgb_scale):
        omega_c = float(np.clip(omega_base * scale_c, 0.10, 0.95))
        norm_c = I[:,:,c] / (Ag[c]+eps)
        dc_c = minimum_filter(norm_c, size=15)
        dc_channels.append(dc_c)

        # Per-channel raw t
        omega_local_c = np.clip(omega_c * (1.0 + omega_boost * haze_density), 0.10, 0.95)
        t_c_raw = np.clip(1 - omega_local_c * dc_c, 0.05, 1.0)

        # Per-channel delta from unified (captures channel-specific variation)
        delta_c = t_c_raw - t_unified_raw
        # Smooth delta with large r (removes edge artifacts from delta)
        delta_c_smooth = box_smooth(delta_c, r=r_base) * delta_scale

        # Final t = unified sharp edges + smooth per-channel offset
        t_c = np.maximum(t_unified_ref + delta_c_smooth, t0)

        # Dual Local A (keep large r)
        dc_smooth = box_smooth(dc_c, r=r_base)
        x = np.clip(1.0 - dc_smooth, 0, 1)
        clear_mod = np.clip(alpha_clear * np.power(x, power), 0, 0.5)

        haze_sat_mod = np.clip(sat_scale * np.power(1.0 - sat_smooth, sat_power), 0, 0.15)

        A_c = Ag[c] * (1.0 - clear_mod - haze_sat_mod)
        J[:,:,c] = np.clip((I[:,:,c] - A_c) / t_c + A_c, 0, 1)

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
    s_orig = ssim_val(J_orig, G)

    print("=== Method 7: Unified Sharp + Smooth Per-Channel Delta ===")
    print(f"Original box r=60: PSNR={p_orig:.2f} SSIM={s_orig:.4f}")

    print(f"\n{'r_t':>5} {'dScl':>5} {'PSNR':>8} {'SSIM':>8} {'dPSNR':>7}")
    print("-" * 36)

    best_psnr = 0
    best_cfg = (10, 1.0)
    best_ssim = 0

    for r_t in [3, 5, 8, 10, 12, 15, 20, 30, 60]:
        for ds in [0.0, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5]:
            J = dehaze_m7(I_bgr, omega_base, r_trans=r_t, delta_scale=ds)
            p = psnr(J, G)
            s = ssim_val(J, G)
            dp = p - p_orig
            mark = " *" if p > best_psnr else ""
            print(f"{r_t:>5} {ds:>5.1f} {p:>8.2f} {s:>8.4f} {dp:>+7.2f}{mark}")
            if p > best_psnr:
                best_psnr = p
                best_cfg = (r_t, ds)
                best_ssim = s

    br, bds = best_cfg
    print(f"\nBest: r_trans={br} delta_scale={bds} PSNR={best_psnr:.2f}")

    # Fine-tune
    print("Fine-tuning...")
    for r_t in range(max(1, br - 3), br + 4):
        for ds in np.arange(max(0, bds - 0.3), bds + 0.35, 0.05):
            J = dehaze_m7(I_bgr, omega_base, r_trans=r_t, delta_scale=float(ds))
            p = psnr(J, G)
            if p > best_psnr:
                best_psnr = p
                best_cfg = (r_t, float(ds))
                best_ssim = ssim_val(J, G)

    r_t, ds = best_cfg
    print(f"Final: r_trans={r_t} delta_scale={ds:.2f} PSNR={best_psnr:.2f} SSIM={best_ssim:.4f} (delta={best_psnr-p_orig:+.2f})")

    J_best = dehaze_m7(I_bgr, omega_base, r_trans=r_t, delta_scale=ds)
    out_path = os.path.join(OUT_DIR, '18_m7_split.png')
    cv2.imwrite(out_path, J_best)
    print(f"Saved: {out_path}")
