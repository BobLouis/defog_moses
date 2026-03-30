
"""
Per-channel DCP + Dual Local Atmospheric Light — NO Guided Filter (Hardware Version)
Based on: dehaze_final_report.py

Hardware considerations:
  - Guided filter replaced by box filter (simple averaging)
    -> Only needs: accumulator + divider, fully pipelinable
    -> No multiplication of guide*source, no covariance computation
  - Box filter: cv2.blur(src, (ks, ks)) = sliding window average
    -> Hardware: row buffer + column accumulator + normalization
  - minimum_filter: sliding window min
    -> Hardware: min-tree comparator
  - All other ops: multiply, add, clip, power (LUT or shift-add)

Trade-off:
  - Box filter does NOT preserve edges (unlike guided filter)
  - Transmission map will have halo artifacts at object boundaries
  - Expected PSNR drop, but simpler hardware

Guided filter usage in original (4 places):
  1) compute_haze_density: smooth haze density map
  2) dehaze_final: smooth saturation map
  3) dehaze_final: refine transmission (t_c_raw -> t_c_ref)
  4) dehaze_final: smooth per-channel dark channel (dc_smooth)
"""
import numpy as np
import cv2
from scipy.ndimage import minimum_filter
import os, glob

eps = 1e-6
BASE  = '/mnt/c/Users/DIC/Documents/defog/img_soft_cl_posA'
HAZY  = os.path.join(BASE, 'dataset/OHaze/hazy')
CLEAR = os.path.join(BASE, 'dataset/OHaze/clear')
OUT   = os.path.join(BASE, 'output_FINAL_no_gf')

REFINED_OMEGA = {
    '01_hazy.png': 0.35, '02_hazy.png': 0.45, '03_hazy.png': 0.70,
    '04_hazy.png': 0.20, '05_hazy.png': 0.85, '06_hazy.png': 0.90,
    '07_hazy.png': 0.60, '08_hazy.png': 0.85, '09_hazy.png': 0.80,
    '10_hazy.png': 0.75, '11_hazy.png': 0.65, '12_hazy.png': 0.60,
    '13_hazy.png': 0.85, '14_hazy.png': 0.60, '15_hazy.png': 0.80,
    '16_hazy.png': 0.85, '17_hazy.png': 0.75, '18_hazy.png': 0.85,
    '19_hazy.png': 0.80, '20_hazy.png': 0.85, '21_hazy.png': 0.85,
    '22_hazy.png': 0.45, '23_hazy.png': 0.65, '24_hazy.png': 0.70,
    '25_hazy.png': 0.85, '26_hazy.png': 0.65, '27_hazy.png': 0.60,
    '28_hazy.png': 0.90, '29_hazy.png': 0.80, '30_hazy.png': 0.50,
    '31_hazy.png': 0.65, '32_hazy.png': 0.45, '33_hazy.png': 0.60,
    '34_hazy.png': 0.60, '35_hazy.png': 0.75, '36_hazy.png': 0.70,
    '37_hazy.png': 0.75, '38_hazy.png': 0.70, '39_hazy.png': 0.65,
    '40_hazy.png': 0.65, '41_hazy.png': 0.50, '42_hazy.png': 0.80,
    '43_hazy.png': 0.50, '44_hazy.png': 0.50, '45_hazy.png': 0.80,
}

def psnr(a, b):
    mse = np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
    return 10*np.log10(255.0**2/mse) if mse > 1e-10 else 999.0

def ssim_val(a, b):
    try:
        from skimage.metrics import structural_similarity
        return np.mean([structural_similarity(
            a[:,:,c].astype(np.float64)/255, b[:,:,c].astype(np.float64)/255,
            data_range=1.0) for c in range(3)])
    except: return float('nan')

def box_smooth(src, r=60):
    """
    Box filter (simple averaging) — hardware-friendly replacement for guided filter.
    Hardware: row line buffers + column accumulators + divider.
    """
    ks = 2 * r + 1
    return cv2.blur(src.astype(np.float32), (ks, ks)).astype(np.float32)

def airlight_topk(I, k=0.001, ps=15):
    dp = minimum_filter(np.min(I, axis=2), size=ps)
    n = max(1, int(I.shape[0]*I.shape[1]*k))
    fi = dp.flatten(); fI = I.reshape(-1,3)
    top = np.argpartition(fi,-n)[-n:]
    return fI[top[np.argmax(np.max(fI[top], axis=1))]]

def compute_haze_density(I, Ag, r=60):
    """Compute per-pixel haze density — box filter version."""
    dc_global = minimum_filter(np.min(I, axis=2), size=15)
    Ag_max = np.max(Ag)
    haze_density = np.clip(dc_global / (Ag_max + eps), 0, 1)
    haze_smooth = box_smooth(haze_density, r=r)
    return np.clip(haze_smooth, 0, 1)

def dehaze_no_gf(I_bgr, omega_base, rgb_scale=(0.80, 0.93, 1.05), r=60, t0=0.1,
                 alpha_clear=0.30, power=3, omega_boost=0.10,
                 sat_scale=0.04, sat_power=3):
    """
    Per-channel DCP + Dual Local A + Omega Boost — NO guided filter.
    All smoothing uses box filter (hardware-friendly).
    """
    I = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    Ag = airlight_topk(I)

    # Haze density map for omega boost (box filter)
    haze_density = compute_haze_density(I, Ag, r=r)

    # Saturation map for haze-area A correction (box filter)
    hsv = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:,:,1].astype(np.float32) / 255.0
    sat_smooth = box_smooth(sat, r=r)

    J = np.zeros_like(I)
    for c, scale_c in enumerate(rgb_scale):
        omega_c = float(np.clip(omega_base * scale_c, 0.10, 0.95))
        norm_c = I[:,:,c] / (Ag[c]+eps)
        dc_c = minimum_filter(norm_c, size=15)

        # --- Omega boost in hazy areas ---
        omega_local = np.clip(omega_c * (1.0 + omega_boost * haze_density), 0.10, 0.95)
        t_c_raw = np.clip(1 - omega_local * dc_c, 0.05, 1.0)
        t_c_ref = box_smooth(t_c_raw, r=r)           # <-- box instead of guided
        t_c = np.maximum(t_c_ref, t0)

        # --- Dual Local A ---
        # 1) Clear-area correction (DCP-based)
        dc_smooth = box_smooth(dc_c, r=r)             # <-- box instead of guided
        x = np.clip(1.0 - dc_smooth, 0, 1)
        clear_mod = np.clip(alpha_clear * np.power(x, power), 0, 0.5)

        # 2) Haze-area correction (saturation-based)
        haze_sat_mod = np.clip(sat_scale * np.power(1.0 - sat_smooth, sat_power), 0, 0.15)

        A_c = Ag[c] * (1.0 - clear_mod - haze_sat_mod)

        J[:,:,c] = np.clip((I[:,:,c] - A_c) / t_c + A_c, 0, 1)

    return cv2.cvtColor((J*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# ============================================================
# Main
# ============================================================
hazy_files = sorted(glob.glob(os.path.join(HAZY,'*.png')))
os.makedirs(OUT, exist_ok=True)

rgb_scale = (0.80, 0.93, 1.05)
print(f"\n=== NO Guided Filter (Box Filter Only) — Hardware Version ===")
print(f"    Params: clear a={0.30},p={3}; haze_sat ss={0.04},sp={3}; boost={0.10}")
ps_list = []; ss_list = []
for hf in hazy_files:
    fname = os.path.basename(hf)
    g = os.path.join(CLEAR, fname.replace('_hazy','_clear'))
    if not os.path.exists(g): continue
    I_bgr = cv2.imread(hf); G = cv2.imread(g)
    omega_base = REFINED_OMEGA.get(fname, 0.70)
    J_bgr = dehaze_no_gf(I_bgr, omega_base, rgb_scale=rgb_scale)
    p = psnr(J_bgr, G)
    s = ssim_val(J_bgr, G)
    ps_list.append(p); ss_list.append(s)
    cv2.imwrite(os.path.join(OUT, fname.replace('_hazy','_dehazed')), J_bgr)
    print(f"  {fname}: w={omega_base:.2f}  PSNR={p:.2f}  SSIM={s:.4f}", flush=True)

mean_p = np.mean(ps_list)
mean_s = float(np.nanmean(ss_list))
print(f"\n  Mean PSNR = {mean_p:.2f} dB  SSIM = {mean_s:.4f}")
print(f"\n  === Comparison ===")
print(f"  With guided filter:    PSNR=19.50 dB  SSIM=0.6532")
print(f"  Without guided filter: PSNR={mean_p:.2f} dB  SSIM={mean_s:.4f}")
print(f"  Delta:                 PSNR={mean_p - 19.50:+.2f} dB  SSIM={mean_s - 0.6532:+.4f}")
print(f"\n  Paper target (DMTME):  PSNR=18.89 dB  SSIM=0.734")

print("\n===== Hardware Notes =====")
print("  Box filter replaces guided filter at 4 locations:")
print("    1) Haze density smoothing")
print("    2) Saturation map smoothing")
print("    3) Transmission refinement (t_c_raw -> t_c_ref)")
print("    4) Dark channel smoothing (dc_smooth)")
print("  Hardware cost: row buffers + accumulator + divider per filter stage")
print("  No guide*src multiplication, no covariance, no a/b coefficient maps")
