"""
Method 8: Keep original r=60 dehazing (no color fringing), then apply
luminance-only edge sharpening to reduce perceived halo.
Uses unsharp masking on L channel of LAB color space.
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

def dehaze_no_gf(I_bgr, omega_base, rgb_scale=(0.80, 0.93, 1.05), r=60, t0=0.1,
                 alpha_clear=0.30, power=3, omega_boost=0.10,
                 sat_scale=0.04, sat_power=3):
    """Original dehaze with r=60 box filter (from dehaze_v7_3type)."""
    I = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    Ag = airlight_topk(I)
    haze_density = compute_haze_density(I, Ag, r=r)
    hsv = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1].astype(np.float32) / 255.0
    sat_smooth = box_smooth(sat, r=r)
    J = np.zeros_like(I)
    for c, scale_c in enumerate(rgb_scale):
        omega_c = float(np.clip(omega_base * scale_c, 0.10, 0.95))
        norm_c = I[:,:,c] / (Ag[c]+eps)
        dc_c = minimum_filter(norm_c, size=15)
        omega_local = np.clip(omega_c * (1.0 + omega_boost * haze_density), 0.10, 0.95)
        t_c_raw = np.clip(1 - omega_local * dc_c, 0.05, 1.0)
        t_c_ref = box_smooth(t_c_raw, r=r)
        t_c = np.maximum(t_c_ref, t0)
        dc_smooth = box_smooth(dc_c, r=r)
        x = np.clip(1.0 - dc_smooth, 0, 1)
        clear_mod = np.clip(alpha_clear * np.power(x, power), 0, 0.5)
        haze_sat_mod = np.clip(sat_scale * np.power(1.0 - sat_smooth, sat_power), 0, 0.15)
        A_c = Ag[c] * (1.0 - clear_mod - haze_sat_mod)
        J[:,:,c] = np.clip((I[:,:,c] - A_c) / t_c + A_c, 0, 1)
    return cv2.cvtColor((J*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def lum_sharpen(img_bgr, sigma=30, amount=1.0):
    """
    Unsharp mask on L channel only (LAB space).
    sigma: Gaussian blur radius (should match halo radius ~30-60)
    amount: sharpening strength
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:,:,0]
    # Blur L to get low-frequency component
    L_blur = cv2.GaussianBlur(L, (0, 0), sigma)
    # High-frequency detail = L - L_blur
    detail = L - L_blur
    # Add back sharpened detail
    L_sharp = np.clip(L + amount * detail, 0, 255)
    lab[:,:,0] = L_sharp
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

def lum_sharpen_adaptive(img_bgr, sigma=30, amount=1.0, edge_thresh=5):
    """
    Adaptive unsharp mask: only sharpen where there's actual edge detail.
    Prevents noise amplification in flat areas.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:,:,0]
    L_blur = cv2.GaussianBlur(L, (0, 0), sigma)
    detail = L - L_blur

    # Only sharpen where detail is significant (edge regions)
    detail_abs = np.abs(detail)
    mask = np.clip((detail_abs - edge_thresh) / edge_thresh, 0, 1)

    L_sharp = np.clip(L + amount * detail * mask, 0, 255)
    lab[:,:,0] = L_sharp
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


if __name__ == "__main__":
    HAZY  = './dataset/OHaze/hazy/18_hazy.png'
    CLEAR = './dataset/OHaze/clear/18_clear.png'
    OUT_DIR = './dataset/OHaze/result_dehalo_test'
    os.makedirs(OUT_DIR, exist_ok=True)

    I_bgr = cv2.imread(HAZY)
    G = cv2.imread(CLEAR)
    omega_base = 0.85

    # Get original dehazed result (r=60, known good, no fringing)
    J_orig = dehaze_no_gf(I_bgr, omega_base)
    p_orig = psnr(J_orig, G)
    s_orig = ssim_val(J_orig, G)

    print("=== Method 8: Luminance-Only Sharpening (post-process) ===")
    print(f"Original r=60: PSNR={p_orig:.2f} SSIM={s_orig:.4f}")

    # Test basic unsharp mask
    print(f"\n--- Basic Unsharp Mask ---")
    print(f"{'sigma':>6} {'amt':>5} {'PSNR':>8} {'SSIM':>8} {'dPSNR':>7}")
    print("-" * 40)

    best_psnr = p_orig
    best_cfg = None
    best_ssim = s_orig
    best_mode = "orig"

    for sigma in [10, 20, 30, 40, 50, 60, 80]:
        for amt in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
            J = lum_sharpen(J_orig, sigma=sigma, amount=amt)
            p = psnr(J, G)
            s = ssim_val(J, G)
            dp = p - p_orig
            mark = " *" if p > best_psnr else ""
            if mark or (sigma in [30, 60] and amt in [0.3, 0.5, 1.0]):
                print(f"{sigma:>6} {amt:>5.1f} {p:>8.2f} {s:>8.4f} {dp:>+7.2f}{mark}")
            if p > best_psnr:
                best_psnr = p
                best_cfg = (sigma, amt)
                best_ssim = s
                best_mode = "basic"

    if best_cfg:
        print(f"Best basic: sigma={best_cfg[0]} amt={best_cfg[1]} PSNR={best_psnr:.2f}")

    # Test adaptive unsharp mask
    print(f"\n--- Adaptive Unsharp Mask ---")
    print(f"{'sigma':>6} {'amt':>5} {'thr':>5} {'PSNR':>8} {'SSIM':>8} {'dPSNR':>7}")
    print("-" * 45)

    for sigma in [10, 20, 30, 40, 60]:
        for amt in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            for thr in [2, 5, 8, 10]:
                J = lum_sharpen_adaptive(J_orig, sigma=sigma, amount=amt, edge_thresh=thr)
                p = psnr(J, G)
                s = ssim_val(J, G)
                dp = p - p_orig
                mark = " *" if p > best_psnr else ""
                if mark:
                    print(f"{sigma:>6} {amt:>5.1f} {thr:>5} {p:>8.2f} {s:>8.4f} {dp:>+7.2f}{mark}")
                    best_psnr = p
                    best_cfg = (sigma, amt, thr)
                    best_ssim = s
                    best_mode = "adaptive"

    if best_mode == "adaptive":
        sigma, amt, thr = best_cfg
        print(f"Best adaptive: sigma={sigma} amt={amt} thr={thr} PSNR={best_psnr:.2f} SSIM={best_ssim:.4f}")

        # Fine-tune
        print("\nFine-tuning adaptive...")
        bsig, bamt, bthr = best_cfg
        for sigma in [max(5, bsig-5), bsig, bsig+5, bsig+10]:
            for amt in np.arange(max(0.1, bamt-0.3), bamt+0.35, 0.1):
                for thr in [max(1, bthr-2), bthr, bthr+2]:
                    J = lum_sharpen_adaptive(J_orig, sigma=sigma, amount=float(amt), edge_thresh=thr)
                    p = psnr(J, G)
                    if p > best_psnr:
                        best_psnr = p
                        best_cfg = (sigma, float(amt), thr)
                        best_ssim = ssim_val(J, G)

        sigma, amt, thr = best_cfg
        print(f"Final: sigma={sigma} amt={amt:.1f} thr={thr} PSNR={best_psnr:.2f} SSIM={best_ssim:.4f} (delta={best_psnr-p_orig:+.2f})")
        J_best = lum_sharpen_adaptive(J_orig, sigma=sigma, amount=amt, edge_thresh=thr)
    elif best_mode == "basic":
        sigma, amt = best_cfg
        print(f"\nFinal (basic): sigma={sigma} amt={amt} PSNR={best_psnr:.2f} SSIM={best_ssim:.4f} (delta={best_psnr-p_orig:+.2f})")
        J_best = lum_sharpen(J_orig, sigma=sigma, amount=amt)
    else:
        print("No improvement found.")
        J_best = J_orig

    out_path = os.path.join(OUT_DIR, '18_m8_lumsharp.png')
    cv2.imwrite(out_path, J_best)
    print(f"Saved: {out_path}")
