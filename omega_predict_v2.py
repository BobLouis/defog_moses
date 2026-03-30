"""
V2: Build PSNR lookup tables per image, then optimize coefficients by table lookup.
This avoids re-running dehaze during optimization.
"""
import numpy as np
import cv2
import os, glob
from scipy.ndimage import minimum_filter
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr

eps = 1e-6

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

hazy_dir = './dataset/OHaze/hazy'
clear_dir = './dataset/OHaze/clear'
hazy_files = sorted(glob.glob(os.path.join(hazy_dir, '*.png')))

# ========== Step 1: Build PSNR lookup table for each image ==========
print("=== Building PSNR lookup tables ===")
omega_grid = np.arange(0.20, 0.91, 0.01)  # 71 omega values
N_omega = len(omega_grid)

images_data = []  # (fname, features, psnr_table[omega_idx])

for hazy_path in hazy_files:
    fname = os.path.basename(hazy_path)
    bn = fname.split('_')[0]
    cp = os.path.join(clear_dir, f"{bn}_clear.png")
    if not os.path.exists(cp):
        continue

    I_bgr = cv2.imread(hazy_path)
    G = np.array(Image.open(cp).convert('RGB'))

    # Extract features
    I = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    Ag = airlight_topk(I)
    Ag_max = float(np.max(Ag))
    dc = minimum_filter(np.min(I, axis=2), size=15)
    dc_norm = float(np.mean(dc / (Ag_max + eps)))
    hsv = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2HSV)
    sat = float(hsv[:,:,1].mean() / 255.0)
    bright = float(I.mean())
    gray = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dr = float((gray.max() - gray.min()) / 255.0)
    avg_dev = float(np.mean(np.abs(gray - gray.mean())) / 255.0)

    # Build PSNR table
    psnr_table = np.zeros(N_omega)
    for j, omega in enumerate(omega_grid):
        J = dehaze_no_gf(I_bgr, omega)
        J_rgb = cv2.cvtColor(J, cv2.COLOR_BGR2RGB)
        if J_rgb.shape != G.shape:
            G_r = cv2.resize(G, (J_rgb.shape[1], J_rgb.shape[0]))
        else:
            G_r = G
        psnr_table[j] = calculate_psnr(G_r, J_rgb)

    best_idx = np.argmax(psnr_table)
    feats = np.array([dc_norm, sat, bright, dr, avg_dev])
    images_data.append((fname, feats, psnr_table))
    print(f"  {fname}: best omega={omega_grid[best_idx]:.2f} PSNR={psnr_table[best_idx]:.2f}  "
          f"dc={dc_norm:.3f} sat={sat:.3f} bright={bright:.3f}")

N = len(images_data)
print(f"\n{N} images loaded, {N_omega} omega values each")

# ========== Step 2: Optimize using lookup tables ==========
print("\n=== Optimizing coefficients ===")

def eval_coeffs(coeffs, feat_names_idx):
    """Evaluate avg PSNR using lookup tables (instant!)"""
    psnr_sum = 0
    for fname, feats, psnr_table in images_data:
        f_sel = feats[feat_names_idx] if feat_names_idx is not None else feats
        pred_omega = float(np.clip(np.dot(f_sel, coeffs[:-1]) + coeffs[-1], 0.20, 0.90))
        # Find nearest omega in grid
        idx = int(round((pred_omega - 0.20) / 0.01))
        idx = max(0, min(idx, N_omega-1))
        psnr_sum += psnr_table[idx]
    return psnr_sum / N

# Feature sets to try
feat_configs = {
    'M1: dc+sat+bright': [0, 1, 2],
    'M2: dc+sat+bright+dr+dev': [0, 1, 2, 3, 4],
    'M3: dc+bright': [0, 2],
    'M4: dc only': [0],
    'M5: sat+bright': [1, 2],
}

# Also try constant omega
print("\n--- Constant omega baseline ---")
best_const_psnr = 0
best_const_omega = 0.70
for omega in omega_grid:
    psnr_sum = 0
    for fname, feats, psnr_table in images_data:
        idx = int(round((omega - 0.20) / 0.01))
        psnr_sum += psnr_table[idx]
    avg = psnr_sum / N
    if avg > best_const_psnr:
        best_const_psnr = avg
        best_const_omega = omega
print(f"Best constant omega={best_const_omega:.2f} PSNR={best_const_psnr:.4f}")

# For each feature config, do exhaustive grid search (fast with lookup tables)
overall_best_psnr = best_const_psnr
overall_best_name = f"constant {best_const_omega:.2f}"
overall_best_coeffs = None
overall_best_feat_idx = None

for name, feat_idx in feat_configs.items():
    n_feats = len(feat_idx)
    n_params = n_feats + 1  # + bias

    # OLS initialization
    X = np.column_stack([np.array([d[1][feat_idx] for d in images_data]),
                         np.ones(N)])
    Y = np.array([omega_grid[np.argmax(d[2])] for d in images_data])
    ols_coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    best_c = ols_coeffs.copy()
    best_p = eval_coeffs(best_c, feat_idx)
    print(f"\n--- {name} ---")
    print(f"OLS start: PSNR={best_p:.4f} coeffs={best_c}")

    # Coordinate descent - very fast with lookup tables
    for round_num in range(5):
        for step in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
            improved = True
            while improved:
                improved = False
                for dim in range(n_params):
                    for delta in [-step, step]:
                        trial = best_c.copy()
                        trial[dim] += delta
                        p = eval_coeffs(trial, feat_idx)
                        if p > best_p + 1e-6:
                            best_p = p
                            best_c = trial
                            improved = True

    print(f"Optimized: PSNR={best_p:.4f} coeffs={best_c}")
    print(f"  delta from baseline: {best_p - 19.5754:+.4f}")

    if best_p > overall_best_psnr:
        overall_best_psnr = best_p
        overall_best_name = name
        overall_best_coeffs = best_c.copy()
        overall_best_feat_idx = feat_idx

# ========== Step 3: Try non-linear mappings ==========
print("\n=== Non-linear approaches ===")

# Piecewise: use dc_norm to split into bins, different omega per bin
print("\n--- Piecewise by dc_norm ---")
dc_vals = np.array([d[1][0] for d in images_data])
# Try 2, 3, 4 bins
for n_bins in [2, 3, 4]:
    thresholds = np.percentile(dc_vals, np.linspace(0, 100, n_bins+1)[1:-1])
    bin_omegas = []
    for b in range(n_bins):
        mask = np.zeros(N, dtype=bool)
        for i in range(N):
            if b == 0:
                mask[i] = dc_vals[i] <= (thresholds[0] if n_bins > 1 else 999)
            elif b == n_bins - 1:
                mask[i] = dc_vals[i] > thresholds[-1]
            else:
                mask[i] = thresholds[b-1] < dc_vals[i] <= thresholds[b]

        # Find best omega for this bin
        best_bin_psnr = 0
        best_bin_omega = 0.70
        for omega in omega_grid:
            idx_o = int(round((omega - 0.20) / 0.01))
            psnr_sum = sum(images_data[i][2][idx_o] for i in range(N) if mask[i])
            n_in_bin = sum(mask)
            if n_in_bin > 0:
                avg = psnr_sum / n_in_bin
                if avg > best_bin_psnr:
                    best_bin_psnr = avg
                    best_bin_omega = omega
        bin_omegas.append(best_bin_omega)

    # Evaluate full
    psnr_sum = 0
    for i in range(N):
        dc_v = dc_vals[i]
        b = 0
        for t in thresholds:
            if dc_v > t:
                b += 1
        omega = bin_omegas[b]
        idx_o = int(round((omega - 0.20) / 0.01))
        psnr_sum += images_data[i][2][idx_o]
    avg_p = psnr_sum / N
    print(f"  {n_bins} bins: thresholds={[f'{t:.3f}' for t in thresholds]} "
          f"omegas={[f'{o:.2f}' for o in bin_omegas]} PSNR={avg_p:.4f}")

# KNN-style: for each image, find K nearest neighbors by features, use their optimal omega
print("\n--- KNN-style (leave-one-out) ---")
all_feats = np.array([d[1][:3] for d in images_data])  # dc, sat, bright
all_feats_norm = (all_feats - all_feats.mean(0)) / (all_feats.std(0) + 1e-8)
opt_omegas = np.array([omega_grid[np.argmax(d[2])] for d in images_data])

for K in [3, 5, 7, 9]:
    psnr_sum = 0
    for i in range(N):
        dists = np.sum((all_feats_norm - all_feats_norm[i])**2, axis=1)
        dists[i] = 1e10  # exclude self
        nn_idx = np.argsort(dists)[:K]
        pred_omega = np.mean(opt_omegas[nn_idx])
        pred_omega = np.clip(pred_omega, 0.20, 0.90)
        idx_o = int(round((pred_omega - 0.20) / 0.01))
        idx_o = max(0, min(idx_o, N_omega-1))
        psnr_sum += images_data[i][2][idx_o]
    avg_p = psnr_sum / N
    print(f"  K={K}: PSNR={avg_p:.4f}")

# ========== Final summary ==========
print(f"\n{'='*60}")
print(f"Summary:")
print(f"  Baseline (REFINED_OMEGA dict): PSNR=19.5754")
print(f"  Best constant omega={best_const_omega:.2f}: PSNR={best_const_psnr:.4f}")
print(f"  Best linear model ({overall_best_name}): PSNR={overall_best_psnr:.4f}")
if overall_best_coeffs is not None:
    print(f"  Coefficients: {overall_best_coeffs}")
print(f"{'='*60}")

# Print the predict function for the best linear model
if overall_best_coeffs is not None and overall_best_feat_idx is not None:
    feat_names = ['dc_norm', 'sat', 'bright', 'dr', 'avg_dev']
    sel_feats = [feat_names[i] for i in overall_best_feat_idx]
    print(f"\ndef predict_omega(I_bgr):")
    print(f"    I = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0")
    print(f"    Ag = airlight_topk(I)")
    print(f"    Ag_max = float(np.max(Ag))")
    print(f"    dc = minimum_filter(np.min(I, axis=2), size=15)")
    for fn in sel_feats:
        if fn == 'dc_norm':
            print(f"    dc_norm = float(np.mean(dc / (Ag_max + eps)))")
        elif fn == 'sat':
            print(f"    hsv = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2HSV)")
            print(f"    sat = float(hsv[:,:,1].mean() / 255.0)")
        elif fn == 'bright':
            print(f"    bright = float(I.mean())")
        elif fn == 'dr':
            print(f"    gray = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)")
            print(f"    dr = float((gray.max() - gray.min()) / 255.0)")
        elif fn == 'avg_dev':
            print(f"    avg_dev = float(np.mean(np.abs(gray - gray.mean())) / 255.0)")

    terms = []
    for i, fn in enumerate(sel_feats):
        terms.append(f"{overall_best_coeffs[i]:.4f}*{fn}")
    terms.append(f"{overall_best_coeffs[-1]:.4f}")
    print(f"    omega = {' + '.join(terms)}")
    print(f"    return float(np.clip(omega, 0.20, 0.90))")

# Show per-image comparison
print(f"\n{'Image':<15} {'OptΩ':>5} {'PredΩ':>6} {'RefΩ':>5} {'PSNR_opt':>9} {'PSNR_pred':>10} {'PSNR_ref':>9}")
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
for fname, feats, psnr_table in images_data:
    opt_idx = np.argmax(psnr_table)
    opt_omega = omega_grid[opt_idx]
    opt_psnr = psnr_table[opt_idx]

    ref_omega = REFINED_OMEGA.get(fname, 0.70)
    ref_idx = int(round((ref_omega - 0.20) / 0.01))
    ref_idx = max(0, min(ref_idx, N_omega-1))
    ref_psnr = psnr_table[ref_idx]

    if overall_best_coeffs is not None:
        f_sel = feats[overall_best_feat_idx]
        pred_omega = float(np.clip(np.dot(f_sel, overall_best_coeffs[:-1]) + overall_best_coeffs[-1], 0.20, 0.90))
    else:
        pred_omega = best_const_omega
    pred_idx = int(round((pred_omega - 0.20) / 0.01))
    pred_idx = max(0, min(pred_idx, N_omega-1))
    pred_psnr = psnr_table[pred_idx]

    print(f"  {fname:<15} {opt_omega:>5.2f} {pred_omega:>6.2f} {ref_omega:>5.2f} "
          f"{opt_psnr:>9.2f} {pred_psnr:>10.2f} {ref_psnr:>9.2f}")
