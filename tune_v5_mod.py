"""Tune defog_avsd_v5_mod.py parameters - 50+ attempts, PSNR-only for speed"""
import numpy as np
import os
from PIL import Image
from glob import glob
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from scipy.ndimage import minimum_filter

epsilon = 1e-6

# Load all images
print("Loading images...", flush=True)
data = {}
for ds in ['OHaze', 'SOTS_out', 'SOTS_in']:
    hazy_files = sorted(glob(f'./dataset/{ds}/hazy/*.png'))
    pairs = []
    for f in hazy_files:
        bn = os.path.splitext(os.path.basename(f))[0].split('_')[0]
        cp = f'./dataset/{ds}/clear/{bn}_clear.png'
        if os.path.exists(cp):
            pairs.append((np.array(Image.open(f).convert('RGB')),
                          np.array(Image.open(cp).convert('RGB'))))
    data[ds] = pairs
    print(f'  {ds}: {len(pairs)} pairs', flush=True)


def defog_v5(hazy, psi_slope=0.009308, psi_intercept=0.927009, psi_min=0.5, psi_max=1.2,
             t0=0.2, window_size=8,
             intensity_boost=0.0, intensity_threshold=0.45, intensity_divisor=0.35,
             intensity_power=1.0, dark_boost=0.0):
    """V5 defog with tunable parameters"""
    H = hazy.astype(np.float32)

    # PSI prediction (vectorized, not pixel-by-pixel loop)
    gray = hazy[:, :, 0]
    mx, mn = int(np.max(gray)), int(np.min(gray))
    dr = mx - mn
    if dr >= 240: fs = 0
    elif dr <= 100: fs = 100
    else: fs = (240 - dr) >> 1
    fs = max(0, min(100, fs))
    psi = max(psi_min, min(psi_max, psi_slope * fs + psi_intercept))

    # Intensity-based adaptive boost
    if intensity_boost > 0:
        mean_intensity = np.mean(H) / 255.0
        factor = max(0, (mean_intensity - intensity_threshold) / intensity_divisor) ** intensity_power
        psi = min(psi_max, psi + intensity_boost * factor)

    # Dark channel based boost
    dc = np.min(H, axis=2)
    dm = minimum_filter(dc, size=window_size)
    idx = np.argmax(dm)
    y, x = np.unravel_index(idx, dm.shape)
    A = H[y, x, :].copy()

    if dark_boost > 0:
        dark_mean = np.mean(dc) / (np.mean(A) + epsilon)
        psi = min(psi_max, psi + dark_boost * dark_mean)

    # Defogging
    H_norm = H / (A + epsilon)
    K = np.mean(H_norm, axis=2)
    mn2 = np.min(H_norm, axis=2)
    temp = 3 * K + 3 * mn2
    t = (temp - psi * 3 * K * mn2) / (temp + epsilon)
    t = np.clip(t, t0, 1)
    D = (H - A) / t[:, :, np.newaxis] + A
    return np.clip(D, 0, 255).astype(np.uint8)


def test_config(**kwargs):
    results = {}
    for ds in ['OHaze', 'SOTS_out', 'SOTS_in']:
        psnr_sum = 0
        for hazy, clear in data[ds]:
            D = defog_v5(hazy, **kwargs)
            if D.shape != clear.shape:
                h = min(D.shape[0], clear.shape[0])
                w = min(D.shape[1], clear.shape[1])
                D = D[:h, :w]; clear = clear[:h, :w]
            psnr_sum += calculate_psnr(clear, D)
        results[ds] = psnr_sum / len(data[ds])
    return results


# Baseline reference
BL_OH = 16.6067
BL_SO = 22.3646
BL_SI = 17.9952

configs = [
    # === GROUP 1: Baseline & simple PSI formula changes ===
    ("G1-01 baseline", {}),
    ("G1-02 slope+0.001", dict(psi_slope=0.010308)),
    ("G1-03 slope+0.002", dict(psi_slope=0.011308)),
    ("G1-04 intercept+0.03", dict(psi_intercept=0.957009)),
    ("G1-05 intercept+0.05", dict(psi_intercept=0.977009)),
    ("G1-06 psi_max=1.3", dict(psi_max=1.3)),
    ("G1-07 psi_max=1.35", dict(psi_max=1.35)),

    # === GROUP 2: Intensity boost (proven best from prev session) ===
    ("G2-01 ib=0.20 th=0.40", dict(intensity_boost=0.20, intensity_threshold=0.40, psi_max=1.35)),
    ("G2-02 ib=0.22 th=0.40", dict(intensity_boost=0.22, intensity_threshold=0.40, psi_max=1.35)),
    ("G2-03 ib=0.25 th=0.40", dict(intensity_boost=0.25, intensity_threshold=0.40, psi_max=1.35)),
    ("G2-04 ib=0.25 th=0.45", dict(intensity_boost=0.25, intensity_threshold=0.45, psi_max=1.35)),
    ("G2-05 ib=0.26 th=0.45", dict(intensity_boost=0.26, intensity_threshold=0.45, psi_max=1.35)),
    ("G2-06 ib=0.28 th=0.45", dict(intensity_boost=0.28, intensity_threshold=0.45, psi_max=1.35)),
    ("G2-07 ib=0.30 th=0.45", dict(intensity_boost=0.30, intensity_threshold=0.45, psi_max=1.35)),
    ("G2-08 ib=0.25 th=0.50 d=0.30", dict(intensity_boost=0.25, intensity_threshold=0.50, intensity_divisor=0.30, psi_max=1.35)),
    ("G2-09 ib=0.30 th=0.50 d=0.30", dict(intensity_boost=0.30, intensity_threshold=0.50, intensity_divisor=0.30, psi_max=1.35)),
    ("G2-10 ib=0.35 th=0.50 d=0.30", dict(intensity_boost=0.35, intensity_threshold=0.50, intensity_divisor=0.30, psi_max=1.35)),
    ("G2-11 ib=0.40 th=0.50 d=0.30", dict(intensity_boost=0.40, intensity_threshold=0.50, intensity_divisor=0.30, psi_max=1.35)),

    # === GROUP 3: Squared/cubed intensity factor (more selective) ===
    ("G3-01 ib=0.35 th=0.40 p=2", dict(intensity_boost=0.35, intensity_threshold=0.40, intensity_power=2.0, psi_max=1.35)),
    ("G3-02 ib=0.40 th=0.40 p=2", dict(intensity_boost=0.40, intensity_threshold=0.40, intensity_power=2.0, psi_max=1.35)),
    ("G3-03 ib=0.50 th=0.40 p=2", dict(intensity_boost=0.50, intensity_threshold=0.40, intensity_power=2.0, psi_max=1.35)),
    ("G3-04 ib=0.50 th=0.45 p=2", dict(intensity_boost=0.50, intensity_threshold=0.45, intensity_power=2.0, psi_max=1.35)),
    ("G3-05 ib=0.60 th=0.45 p=2", dict(intensity_boost=0.60, intensity_threshold=0.45, intensity_power=2.0, psi_max=1.35)),
    ("G3-06 ib=0.60 th=0.50 p=2", dict(intensity_boost=0.60, intensity_threshold=0.50, intensity_power=2.0, intensity_divisor=0.30, psi_max=1.35)),
    ("G3-07 ib=0.80 th=0.50 p=2 d=0.30", dict(intensity_boost=0.80, intensity_threshold=0.50, intensity_power=2.0, intensity_divisor=0.30, psi_max=1.35)),
    ("G3-08 ib=0.60 th=0.40 p=3", dict(intensity_boost=0.60, intensity_threshold=0.40, intensity_power=3.0, psi_max=1.35)),
    ("G3-09 ib=0.80 th=0.45 p=3", dict(intensity_boost=0.80, intensity_threshold=0.45, intensity_power=3.0, psi_max=1.35)),
    ("G3-10 ib=1.00 th=0.50 p=3 d=0.30", dict(intensity_boost=1.00, intensity_threshold=0.50, intensity_power=3.0, intensity_divisor=0.30, psi_max=1.35)),

    # === GROUP 4: Dark channel boost ===
    ("G4-01 db=0.15", dict(dark_boost=0.15, psi_max=1.3)),
    ("G4-02 db=0.20", dict(dark_boost=0.20, psi_max=1.35)),
    ("G4-03 db=0.25", dict(dark_boost=0.25, psi_max=1.35)),
    ("G4-04 db=0.18", dict(dark_boost=0.18, psi_max=1.3)),

    # === GROUP 5: Combined intensity + PSI formula ===
    ("G5-01 ib=0.25 th=0.45 slope+0.001", dict(intensity_boost=0.25, intensity_threshold=0.45, psi_slope=0.010308, psi_max=1.35)),
    ("G5-02 ib=0.25 th=0.45 int+0.02", dict(intensity_boost=0.25, intensity_threshold=0.45, psi_intercept=0.947009, psi_max=1.35)),
    ("G5-03 ib=0.22 th=0.45 int+0.03", dict(intensity_boost=0.22, intensity_threshold=0.45, psi_intercept=0.957009, psi_max=1.35)),
    ("G5-04 ib=0.20 th=0.45 int+0.05", dict(intensity_boost=0.20, intensity_threshold=0.45, psi_intercept=0.977009, psi_max=1.35)),
    ("G5-05 ib=0.20 th=0.40 int+0.03", dict(intensity_boost=0.20, intensity_threshold=0.40, psi_intercept=0.957009, psi_max=1.35)),

    # === GROUP 6: Fine-tune around best configs ===
    ("G6-01 ib=0.24 th=0.43", dict(intensity_boost=0.24, intensity_threshold=0.43, psi_max=1.35)),
    ("G6-02 ib=0.26 th=0.43", dict(intensity_boost=0.26, intensity_threshold=0.43, psi_max=1.35)),
    ("G6-03 ib=0.25 th=0.42 d=0.38", dict(intensity_boost=0.25, intensity_threshold=0.42, intensity_divisor=0.38, psi_max=1.35)),
    ("G6-04 ib=0.27 th=0.44", dict(intensity_boost=0.27, intensity_threshold=0.44, psi_max=1.35)),
    ("G6-05 ib=0.23 th=0.46 d=0.33", dict(intensity_boost=0.23, intensity_threshold=0.46, intensity_divisor=0.33, psi_max=1.35)),
    ("G6-06 ib=0.28 th=0.48 d=0.32", dict(intensity_boost=0.28, intensity_threshold=0.48, intensity_divisor=0.32, psi_max=1.35)),
    ("G6-07 ib=0.30 th=0.48 d=0.32", dict(intensity_boost=0.30, intensity_threshold=0.48, intensity_divisor=0.32, psi_max=1.35)),
    ("G6-08 ib=0.32 th=0.50 d=0.30", dict(intensity_boost=0.32, intensity_threshold=0.50, intensity_divisor=0.30, psi_max=1.35)),
    ("G6-09 ib=0.35 th=0.48 d=0.32", dict(intensity_boost=0.35, intensity_threshold=0.48, intensity_divisor=0.32, psi_max=1.35)),
    ("G6-10 ib=0.38 th=0.50 d=0.28", dict(intensity_boost=0.38, intensity_threshold=0.50, intensity_divisor=0.28, psi_max=1.35)),
    ("G6-11 ib=0.26 th=0.44 d=0.36", dict(intensity_boost=0.26, intensity_threshold=0.44, intensity_divisor=0.36, psi_max=1.35)),
    ("G6-12 ib=0.28 th=0.46 d=0.34", dict(intensity_boost=0.28, intensity_threshold=0.46, intensity_divisor=0.34, psi_max=1.35)),
]

log_path = "tune_v5_mod_log.txt"
log = open(log_path, "w")
log.write("Tuning Log for defog_avsd_v5_mod.py (no map, parameter-only)\n")
log.write(f"Baseline: OH={BL_OH:.4f} SO={BL_SO:.4f} SI={BL_SI:.4f}\n")
log.write(f"Target: SI>18.8000, OH>={BL_OH:.4f}, SO>={BL_SO:.4f}\n")
log.write("=" * 100 + "\n\n")

best_si = 0
best_name = ""
best_balanced = None  # Best that keeps OH and SO close to baseline

print(f"\n{'#':>3} {'Config':<35} {'OH':>8} {'dOH':>7} {'SO':>8} {'dSO':>7} {'SI':>8} {'dSI':>7} {'OK':>4}", flush=True)
print("-" * 100, flush=True)

for i, (name, kwargs) in enumerate(configs):
    r = test_config(**kwargs)
    oh, so, si = r['OHaze'], r['SOTS_out'], r['SOTS_in']
    doh = oh - BL_OH
    dso = so - BL_SO
    dsi = si - BL_SI

    si_ok = si > 18.8
    oh_ok = oh >= BL_OH - 0.02  # allow tiny tolerance
    so_ok = so >= BL_SO - 0.02
    all_ok = "***" if (si_ok and oh_ok and so_ok) else ("SI+" if si_ok else "")

    line = f"{i+1:>3} {name:<35} {oh:>8.4f} {doh:>+7.4f} {so:>8.4f} {dso:>+7.4f} {si:>8.4f} {dsi:>+7.4f} {all_ok:>4}"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()

    if si > best_si:
        best_si = si
        best_name = name

    # Track best balanced (SI>18.8 and min OH/SO drop)
    if si > 18.8:
        penalty = max(0, BL_OH - oh) + max(0, BL_SO - so)
        if best_balanced is None or penalty < best_balanced[1]:
            best_balanced = (name, penalty, oh, so, si)

summary = f"\nBest SOTS_in: {best_si:.4f} ({best_name})\n"
if best_balanced:
    summary += f"Best balanced: {best_balanced[0]} OH={best_balanced[2]:.4f} SO={best_balanced[3]:.4f} SI={best_balanced[4]:.4f}\n"
print(summary, flush=True)
log.write(summary)
log.close()
