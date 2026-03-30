"""
Method 5: Per-channel t with edge-constrained cross-channel blending.
- Compute per-channel t as usual (with rgb_scale) using small r
- At edges, blend each channel's t toward cross-channel mean
- In flat areas, keep per-channel t for PSNR quality
- This prevents inter-channel t divergence at edges (no blue fringing)
  while preserving per-channel accuracy in flat regions.
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

def compute_edge_weight(guide_bgr, edge_sigma=30, grad_boost=1.0):
    """Compute edge weight: 1 at edges, 0 in flat areas."""
    gray = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2) * grad_boost
    edge_weight = 1.0 - np.exp(-(grad**2) / (2.0 * edge_sigma**2))
    return np.clip(edge_weight, 0, 1).astype(np.float32)

def dehaze_m5(I_bgr, omega_base, rgb_scale=(0.80, 0.93, 1.05),
              r_base=60, r_trans=10, t0=0.1,
              alpha_clear=0.30, power=3, omega_boost=0.10,
              sat_scale=0.04, sat_power=3,
              edge_sigma=30, grad_boost=1.0, blend_strength=1.0):
    """
    Per-channel t with edge-constrained blending.
    At edges: blend toward cross-channel mean t (prevents color fringing).
    In flat areas: keep per-channel t (preserves PSNR).
    blend_strength: how much to blend toward mean at edges (0=none, 1=full)
    """
    I = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    Ag = airlight_topk(I)

    haze_density = compute_haze_density(I, Ag, r=r_base)

    hsv = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1].astype(np.float32) / 255.0
    sat_smooth = box_smooth(sat, r=r_base)

    # Compute edge weight map
    edge_w = compute_edge_weight(I_bgr, edge_sigma=edge_sigma, grad_boost=grad_boost)

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
        t_c_ref = box_smooth(t_c_raw, r=r_trans)
        t_c = np.maximum(t_c_ref, t0)
        t_channels.append(t_c)

    # Cross-channel mean transmission
    t_mean = np.mean(t_channels, axis=0)

    # Blend: at edges -> toward mean, in flat -> keep per-channel
    blend_map = edge_w * blend_strength  # [0,1]

    J = np.zeros_like(I)
    for c, scale_c in enumerate(rgb_scale):
        t_c = (1.0 - blend_map) * t_channels[c] + blend_map * t_mean

        # Dual Local A (keep large r)
        dc_smooth = box_smooth(dc_channels[c], r=r_base)
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

    print("=== Method 5: Per-channel t + Edge Cross-Channel Blend ===")
    print(f"Original box r=60: PSNR={p_orig:.2f} SSIM={s_orig:.4f}")

    # Coarse scan
    print(f"\n{'r_t':>5} {'eSig':>6} {'gB':>5} {'bStr':>5} {'PSNR':>8} {'SSIM':>8} {'dPSNR':>7}")
    print("-" * 50)

    best_psnr = 0
    best_cfg = (10, 30, 1.0, 1.0)
    best_ssim = 0

    for r_t in [5, 8, 10, 15, 20, 30]:
        for eSig in [10, 20, 30, 50]:
            for gB in [1.0, 2.0, 3.0]:
                for bStr in [0.5, 0.8, 1.0]:
                    J = dehaze_m5(I_bgr, omega_base, r_trans=r_t,
                                  edge_sigma=eSig, grad_boost=gB,
                                  blend_strength=bStr)
                    p = psnr(J, G)
                    s = ssim_val(J, G)
                    dp = p - p_orig
                    mark = " *" if p > best_psnr else ""
                    if p > best_psnr or mark:
                        print(f"{r_t:>5} {eSig:>6} {gB:>5.1f} {bStr:>5.1f} {p:>8.2f} {s:>8.4f} {dp:>+7.2f}{mark}")
                    if p > best_psnr:
                        best_psnr = p
                        best_cfg = (r_t, eSig, gB, bStr)
                        best_ssim = s

    br, beSig, bgB, bbStr = best_cfg
    print(f"\nBest: r_t={br} eSig={beSig} gB={bgB} bStr={bbStr} PSNR={best_psnr:.2f}")

    # Fine-tune
    print("Fine-tuning...")
    for r_t in range(max(1, br - 3), br + 4):
        for eSig in [max(5, beSig - 5), beSig, beSig + 5, beSig + 10]:
            for gB in [max(0.5, bgB - 0.5), bgB, bgB + 0.5]:
                for bStr in [max(0.1, bbStr - 0.1), bbStr, min(1.0, bbStr + 0.1)]:
                    J = dehaze_m5(I_bgr, omega_base, r_trans=r_t,
                                  edge_sigma=eSig, grad_boost=gB,
                                  blend_strength=bStr)
                    p = psnr(J, G)
                    if p > best_psnr:
                        best_psnr = p
                        best_cfg = (r_t, eSig, gB, bStr)
                        best_ssim = ssim_val(J, G)

    r_t, eSig, gB, bStr = best_cfg
    print(f"Final: r_t={r_t} eSig={eSig} gB={gB} bStr={bStr} PSNR={best_psnr:.2f} SSIM={best_ssim:.4f} (delta={best_psnr-p_orig:+.2f})")

    J_best = dehaze_m5(I_bgr, omega_base, r_trans=r_t,
                        edge_sigma=eSig, grad_boost=gB,
                        blend_strength=bStr)
    out_path = os.path.join(OUT_DIR, '18_m5_edgeblend.png')
    cv2.imwrite(out_path, J_best)
    print(f"Saved: {out_path}")
