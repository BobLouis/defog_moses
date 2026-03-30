"""
omega_predict_v3.py — Train omega prediction on combined datasets
 - OHaze: use REFINED_OMEGA as ground truth (no table needed, images too large)
 - SOTS_out/SOTS_in: build PSNR tables (small images, fast)

Then train models on combined data and optimize via coordinate descent.
"""
import numpy as np
import cv2
from scipy.ndimage import minimum_filter
import os, glob, sys, time

eps = 1e-6

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

def psnr(a, b):
    mse = np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
    return 10*np.log10(255.0**2/mse) if mse > 1e-10 else 999.0

def extract_features(I_bgr, max_pixels=500000):
    """Extract features from hazy image. Downscale if too large."""
    h, w = I_bgr.shape[:2]
    if h * w > max_pixels:
        scale = (max_pixels / (h * w)) ** 0.5
        I_bgr = cv2.resize(I_bgr, (int(w*scale), int(h*scale)))
    I = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    Ag = airlight_topk(I)
    Ag_max = float(np.max(Ag))
    dc = minimum_filter(np.min(I, axis=2), size=15)
    dc_norm = float(np.mean(dc / (Ag_max + eps)))
    hsv = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2HSV)
    sat = float(hsv[:, :, 1].mean() / 255.0)
    bright = float(I.mean())
    gray = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dr = float((gray.max() - gray.min()) / 255.0)
    avg_dev = float(np.mean(np.abs(gray - gray.mean())) / 255.0)
    dc_std = float(np.std(dc / (Ag_max + eps)))
    Ag_mean = float(np.mean(Ag))
    contrast = float(np.std(I))
    return [dc_norm, sat, bright, dr, avg_dev, dc_std, Ag_mean, contrast]

FEAT_NAMES = ['dc_norm', 'sat', 'bright', 'dr', 'avg_dev', 'dc_std', 'Ag_mean', 'contrast']

def get_image_pairs(dataset, max_images=None):
    hazy_dir = f"./dataset/{dataset}/hazy"
    clear_dir = f"./dataset/{dataset}/clear"
    hazy_files = sorted(glob.glob(os.path.join(hazy_dir, "*.png")))
    pairs = []
    for hp in hazy_files:
        base = os.path.splitext(os.path.basename(hp))[0].split('_')[0]
        cp = os.path.join(clear_dir, f"{base}_clear.png")
        if os.path.exists(cp):
            pairs.append((hp, cp))
    if max_images and len(pairs) > max_images:
        indices = np.linspace(0, len(pairs)-1, max_images, dtype=int)
        pairs = [pairs[i] for i in indices]
    return pairs

def eval_psnr_from_table(omega_pred, tables, omega_values):
    total_psnr = 0
    for i, om in enumerate(omega_pred):
        om_clip = np.clip(om, omega_values[0], omega_values[-1])
        idx = np.searchsorted(omega_values, om_clip)
        idx = min(idx, len(omega_values)-1)
        if idx == 0:
            total_psnr += tables[i, 0]
        else:
            w = (om_clip - omega_values[idx-1]) / (omega_values[idx] - omega_values[idx-1])
            total_psnr += tables[i, idx-1] * (1-w) + tables[i, idx] * w
    return total_psnr / len(omega_pred)

if __name__ == "__main__":
    omega_values = np.arange(0.20, 0.91, 0.05)  # 15 values
    print(f"Omega values ({len(omega_values)}): {omega_values}")

    # ============================================================
    # Step 1: Collect OHaze features + optimal omega (from REFINED_OMEGA)
    # ============================================================
    print(f"\n=== Step 1: OHaze features (no table needed) ===")
    ohaze_pairs = get_image_pairs("OHaze")
    ohaze_features = []
    ohaze_omega = []
    for hp, cp in ohaze_pairs:
        fname = os.path.basename(hp)
        if fname not in REFINED_OMEGA:
            continue
        I_bgr = cv2.imread(hp)
        feats = extract_features(I_bgr)
        ohaze_features.append(feats)
        ohaze_omega.append(REFINED_OMEGA[fname])
    print(f"  OHaze: {len(ohaze_features)} images, omega mean={np.mean(ohaze_omega):.3f}")

    # ============================================================
    # Step 2: Build PSNR tables for SOTS (small images, fast)
    # ============================================================
    sots_features = {ds: [] for ds in ["SOTS_out", "SOTS_in"]}
    sots_tables = {ds: [] for ds in ["SOTS_out", "SOTS_in"]}
    sots_optimal = {ds: [] for ds in ["SOTS_out", "SOTS_in"]}

    for dataset in ["SOTS_out", "SOTS_in"]:
        pairs = get_image_pairs(dataset, max_images=80)
        print(f"\n=== Step 2: {dataset} PSNR tables ({len(pairs)} images) ===")
        t_start = time.time()

        for i, (hp, cp) in enumerate(pairs):
            fname = os.path.basename(hp)
            I_bgr = cv2.imread(hp)
            G_bgr = cv2.imread(cp)
            if I_bgr is None or G_bgr is None:
                continue
            if I_bgr.shape[:2] != G_bgr.shape[:2]:
                G_bgr = cv2.resize(G_bgr, (I_bgr.shape[1], I_bgr.shape[0]))

            feats = extract_features(I_bgr)
            psnrs = []
            for omega in omega_values:
                J = dehaze_no_gf(I_bgr, omega)
                p = psnr(J, G_bgr)
                psnrs.append(p)
            psnrs = np.array(psnrs)
            best_idx = np.argmax(psnrs)
            opt_omega = omega_values[best_idx]

            sots_features[dataset].append(feats)
            sots_tables[dataset].append(psnrs)
            sots_optimal[dataset].append(opt_omega)

            if i % 20 == 0 or i == len(pairs)-1:
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{len(pairs)}] {fname}: opt={opt_omega:.2f} "
                      f"best_psnr={psnrs[best_idx]:.2f} ({elapsed:.0f}s)")
            sys.stdout.flush()

        opt = np.array(sots_optimal[dataset])
        print(f"  {dataset}: opt_omega mean={opt.mean():.3f} std={opt.std():.3f} "
              f"range=[{opt.min():.2f}, {opt.max():.2f}]")

    # ============================================================
    # Step 3: Combine data
    # ============================================================
    print(f"\n=== Step 3: Combined training ===")

    # For OHaze, we don't have PSNR tables, so we'll train regression on omega directly
    # For SOTS, we have tables for PSNR-based optimization

    X_all = np.array(ohaze_features +
                     sots_features["SOTS_out"] + sots_features["SOTS_in"])
    Y_all = np.array(ohaze_omega +
                     sots_optimal["SOTS_out"] + sots_optimal["SOTS_in"])

    n_oh = len(ohaze_features)
    n_so = len(sots_features["SOTS_out"])
    n_si = len(sots_features["SOTS_in"])
    print(f"  Total: {len(X_all)} (OHaze={n_oh}, SOTS_out={n_so}, SOTS_in={n_si})")
    print(f"  Overall omega: mean={Y_all.mean():.3f} std={Y_all.std():.3f}")

    # Feature subsets
    X5 = X_all[:, :5]
    X5b = np.column_stack([X5, np.ones(len(X5))])
    X8b = np.column_stack([X_all, np.ones(len(X_all))])

    # ============================================================
    # Model A: 5-feature linear regression
    # ============================================================
    print(f"\n--- Model A: 5-feature linear ---")
    coefA, _, _, _ = np.linalg.lstsq(X5b, Y_all, rcond=None)
    Y_predA = np.clip(X5b @ coefA, 0.20, 0.90)
    r2A = 1 - np.sum((Y_all - Y_predA)**2) / np.sum((Y_all - Y_all.mean())**2)
    mae_A = np.mean(np.abs(Y_all - Y_predA))
    print(f"  Coef: {coefA}")
    print(f"  R²={r2A:.4f}, MAE={mae_A:.4f}")

    # Per-dataset R²
    for name, s, e in [("OHaze", 0, n_oh), ("SOTS_out", n_oh, n_oh+n_so),
                        ("SOTS_in", n_oh+n_so, n_oh+n_so+n_si)]:
        ys = Y_all[s:e]
        yp = Y_predA[s:e]
        r2 = 1 - np.sum((ys - yp)**2) / (np.sum((ys - ys.mean())**2) + 1e-10)
        print(f"    {name}: pred_mean={yp.mean():.3f} true_mean={ys.mean():.3f} R²={r2:.4f}")

    # ============================================================
    # Model B: 8-feature linear regression
    # ============================================================
    print(f"\n--- Model B: 8-feature linear ---")
    coefB, _, _, _ = np.linalg.lstsq(X8b, Y_all, rcond=None)
    Y_predB = np.clip(X8b @ coefB, 0.20, 0.90)
    r2B = 1 - np.sum((Y_all - Y_predB)**2) / np.sum((Y_all - Y_all.mean())**2)
    mae_B = np.mean(np.abs(Y_all - Y_predB))
    print(f"  Coef: {coefB}")
    print(f"  R²={r2B:.4f}, MAE={mae_B:.4f}")

    for name, s, e in [("OHaze", 0, n_oh), ("SOTS_out", n_oh, n_oh+n_so),
                        ("SOTS_in", n_oh+n_so, n_oh+n_so+n_si)]:
        ys = Y_all[s:e]
        yp = Y_predB[s:e]
        r2 = 1 - np.sum((ys - yp)**2) / (np.sum((ys - ys.mean())**2) + 1e-10)
        print(f"    {name}: pred_mean={yp.mean():.3f} true_mean={ys.mean():.3f} R²={r2:.4f}")

    # ============================================================
    # Model C: Polynomial degree 2 on 5 features
    # ============================================================
    print(f"\n--- Model C: Polynomial (deg2) on 5 features ---")
    poly_cols = [X5]
    for i in range(5):
        for j in range(i, 5):
            poly_cols.append((X5[:, i] * X5[:, j]).reshape(-1, 1))
    Xpoly = np.column_stack(poly_cols + [np.ones(len(X5))])
    coefC, _, _, _ = np.linalg.lstsq(Xpoly, Y_all, rcond=None)
    Y_predC = np.clip(Xpoly @ coefC, 0.20, 0.90)
    r2C = 1 - np.sum((Y_all - Y_predC)**2) / np.sum((Y_all - Y_all.mean())**2)
    mae_C = np.mean(np.abs(Y_all - Y_predC))
    print(f"  Num features: {Xpoly.shape[1]}, R²={r2C:.4f}, MAE={mae_C:.4f}")

    for name, s, e in [("OHaze", 0, n_oh), ("SOTS_out", n_oh, n_oh+n_so),
                        ("SOTS_in", n_oh+n_so, n_oh+n_so+n_si)]:
        ys = Y_all[s:e]
        yp = Y_predC[s:e]
        r2 = 1 - np.sum((ys - yp)**2) / (np.sum((ys - ys.mean())**2) + 1e-10)
        print(f"    {name}: pred_mean={yp.mean():.3f} true_mean={ys.mean():.3f} R²={r2:.4f}")

    # ============================================================
    # Evaluate PSNR on SOTS tables
    # ============================================================
    print(f"\n{'='*60}")
    print("=== PSNR evaluation on SOTS tables ===")

    # Build combined SOTS tables
    tables_so = np.array(sots_tables["SOTS_out"])
    tables_si = np.array(sots_tables["SOTS_in"])

    def sots_psnr(omega_pred_so, omega_pred_si):
        p_so = eval_psnr_from_table(omega_pred_so, tables_so, omega_values)
        p_si = eval_psnr_from_table(omega_pred_si, tables_si, omega_values)
        return p_so, p_si

    # Constant baselines
    print("\nConstant omega baselines:")
    for om in np.arange(0.50, 0.85, 0.05):
        p_so, p_si = sots_psnr(np.full(n_so, om), np.full(n_si, om))
        print(f"  omega={om:.2f}: SOTS_out={p_so:.4f} SOTS_in={p_si:.4f}")

    # Model predictions on SOTS
    print("\nModel predictions:")
    for name, Y_pred in [("Model A (5-feat)", Y_predA),
                          ("Model B (8-feat)", Y_predB),
                          ("Model C (poly)", Y_predC)]:
        pred_so = Y_pred[n_oh:n_oh+n_so]
        pred_si = Y_pred[n_oh+n_so:]
        p_so, p_si = sots_psnr(pred_so, pred_si)
        print(f"  {name}: SOTS_out={p_so:.4f} (pred_mean={pred_so.mean():.3f}) "
              f"SOTS_in={p_si:.4f} (pred_mean={pred_si.mean():.3f})")

    # ============================================================
    # Coordinate descent on SOTS PSNR (5-feature model)
    # ============================================================
    print(f"\n{'='*60}")
    print("Coordinate descent: optimize for SOTS PSNR (5-feat)")

    best_coef = coefA.copy()

    def total_sots_psnr(coef):
        pred_all = np.clip(X5b @ coef, 0.20, 0.90)
        pred_so = pred_all[n_oh:n_oh+n_so]
        pred_si = pred_all[n_oh+n_so:]
        p_so = eval_psnr_from_table(pred_so, tables_so, omega_values)
        p_si = eval_psnr_from_table(pred_si, tables_si, omega_values)
        # Also penalize deviation from REFINED_OMEGA for OHaze
        pred_oh = pred_all[:n_oh]
        oh_penalty = np.mean(np.abs(pred_oh - Y_all[:n_oh])) * 2.0  # weight
        return (p_so + p_si) / 2.0 - oh_penalty

    best_score = total_sots_psnr(best_coef)
    print(f"  Start score: {best_score:.4f}")

    for rnd in range(8):
        improved = False
        for ci in range(len(best_coef)):
            base_val = best_coef[ci]
            for step in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
                for direction in [1, -1]:
                    trial = best_coef.copy()
                    trial[ci] = base_val + direction * step
                    sc = total_sots_psnr(trial)
                    if sc > best_score:
                        best_score = sc
                        best_coef = trial
                        improved = True
        if not improved:
            break
        pred_all = np.clip(X5b @ best_coef, 0.20, 0.90)
        p_so, p_si = sots_psnr(pred_all[n_oh:n_oh+n_so], pred_all[n_oh+n_so:])
        oh_mae = np.mean(np.abs(pred_all[:n_oh] - Y_all[:n_oh]))
        print(f"  Round {rnd+1}: score={best_score:.4f} SOTS_out={p_so:.4f} SOTS_in={p_si:.4f} OH_MAE={oh_mae:.4f}")
        sys.stdout.flush()

    print(f"\n  Final 5-feat CD coefficients: {list(best_coef)}")
    print(f"  Features: {FEAT_NAMES[:5]}")

    # Show predictions
    pred_all = np.clip(X5b @ best_coef, 0.20, 0.90)
    print(f"  OHaze pred mean={pred_all[:n_oh].mean():.3f} (true={Y_all[:n_oh].mean():.3f})")
    print(f"  SOTS_out pred mean={pred_all[n_oh:n_oh+n_so].mean():.3f} (true={Y_all[n_oh:n_oh+n_so].mean():.3f})")
    print(f"  SOTS_in pred mean={pred_all[n_oh+n_so:].mean():.3f} (true={Y_all[n_oh+n_so:].mean():.3f})")

    p_so, p_si = sots_psnr(pred_all[n_oh:n_oh+n_so], pred_all[n_oh+n_so:])
    print(f"  SOTS PSNR: out={p_so:.4f} in={p_si:.4f}")

    # ============================================================
    # Also try CD on 8-feature model
    # ============================================================
    print(f"\n{'='*60}")
    print("Coordinate descent: optimize for SOTS PSNR (8-feat)")

    best_coef8 = coefB.copy()

    def total_sots_psnr8(coef):
        pred_all = np.clip(X8b @ coef, 0.20, 0.90)
        pred_so = pred_all[n_oh:n_oh+n_so]
        pred_si = pred_all[n_oh+n_so:]
        p_so = eval_psnr_from_table(pred_so, tables_so, omega_values)
        p_si = eval_psnr_from_table(pred_si, tables_si, omega_values)
        pred_oh = pred_all[:n_oh]
        oh_penalty = np.mean(np.abs(pred_oh - Y_all[:n_oh])) * 2.0
        return (p_so + p_si) / 2.0 - oh_penalty

    best_score8 = total_sots_psnr8(best_coef8)
    print(f"  Start score: {best_score8:.4f}")

    for rnd in range(8):
        improved = False
        for ci in range(len(best_coef8)):
            base_val = best_coef8[ci]
            for step in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
                for direction in [1, -1]:
                    trial = best_coef8.copy()
                    trial[ci] = base_val + direction * step
                    sc = total_sots_psnr8(trial)
                    if sc > best_score8:
                        best_score8 = sc
                        best_coef8 = trial
                        improved = True
        if not improved:
            break
        pred_all = np.clip(X8b @ best_coef8, 0.20, 0.90)
        p_so, p_si = sots_psnr(pred_all[n_oh:n_oh+n_so], pred_all[n_oh+n_so:])
        oh_mae = np.mean(np.abs(pred_all[:n_oh] - Y_all[:n_oh]))
        print(f"  Round {rnd+1}: score={best_score8:.4f} SOTS_out={p_so:.4f} SOTS_in={p_si:.4f} OH_MAE={oh_mae:.4f}")
        sys.stdout.flush()

    print(f"\n  Final 8-feat CD coefficients: {list(best_coef8)}")
    print(f"  Features: {FEAT_NAMES}")

    pred_all8 = np.clip(X8b @ best_coef8, 0.20, 0.90)
    print(f"  OHaze pred mean={pred_all8[:n_oh].mean():.3f} (true={Y_all[:n_oh].mean():.3f})")
    print(f"  SOTS_out pred mean={pred_all8[n_oh:n_oh+n_so].mean():.3f}")
    print(f"  SOTS_in pred mean={pred_all8[n_oh+n_so:].mean():.3f}")

    p_so8, p_si8 = sots_psnr(pred_all8[n_oh:n_oh+n_so], pred_all8[n_oh+n_so:])
    print(f"  SOTS PSNR: out={p_so8:.4f} in={p_si8:.4f}")

    # ============================================================
    # Final Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("=== FINAL SUMMARY ===")
    print(f"{'='*60}")
    print(f"\nBest model for predict_omega():")
    if best_score8 > best_score:
        print(f"  8-feature model")
        print(f"  Coefficients: {list(best_coef8)}")
        print(f"  Features: {FEAT_NAMES}")
    else:
        print(f"  5-feature model")
        print(f"  Coefficients: {list(best_coef)}")
        print(f"  Features: {FEAT_NAMES[:5]}")

    print(f"\nFor dehaze_v7_3type_omega.py, update predict_omega() with these coefficients.")
