# method7_hardware.py
# Based on defog_v6_method7.py — Hardware-Friendly Version
#
# All cv2 calls replaced with pure mathematical formulas.
# Each function includes hardware architecture notes for FPGA/ASIC.
#
# Removed:
#   cv2.cvtColor      → rgb_to_lab() / lab_to_rgb() (manual matrix + LUT)
#   cv2.split/merge   → numpy stack/unstack
#   cv2.createCLAHE   → clahe_channel() (histogram + CDF + bilinear interp)
#   cv2.boxFilter     → box_filter_2d() (summed area table / SAT)
#   cv2.LUT           → numpy fancy indexing  table[image]
#   cv2.COLOR_RGB2GRAY→ BT.601 luma formula

import numpy as np
from scipy.ndimage import minimum_filter


# =============================================================================
# MODULE 1: Precomputed LUTs  (ROM in hardware)
# =============================================================================

# sRGB linearization LUT: 256 entries × 16-bit fixed-point
# Hardware: 256-word ROM, 1-cycle latency
_SRGB_TO_LINEAR = np.array([
    (((i / 255.0 + 0.055) / 1.055) ** 2.4
     if i / 255.0 > 0.04045
     else i / 255.0 / 12.92)
    for i in range(256)
], dtype=np.float32)


# =============================================================================
# MODULE 2: RGB ↔ CIE LAB Color Conversion
#
# Hardware pipeline (4 stages):
#   Stage 1  Gamma LUT       256 × 16b ROM            (1 cycle)
#   Stage 2  RGB → XYZ       3×3 fixed-point Q1.15    (3 cycles)
#   Stage 3  Normalize        /Xn /Yn /Zn              (1 cycle)
#   Stage 4  f(t) cube-root  1024-entry LUT or         (4 cycles)
#            + LAB formula    Newton-Raphson ×3 iter
#   Throughput: 1 pixel/cycle after pipeline fill
# =============================================================================

def _f_lab(t):
    """
    CIE LAB nonlinearity (IEC 61966-7-1):
        f(t) = t^(1/3)              if t > (6/29)^3  ≈ 0.008856
             = t / (3·δ²) + 4/29   otherwise   (δ = 6/29)
    Hardware: 1024-entry LUT for cube root, or Newton-Raphson (3 iterations)
    """
    delta   = 6.0 / 29.0         # ≈ 0.20690
    delta3  = delta ** 3          # ≈ 0.00886
    c3d2    = 3.0 * delta ** 2    # 3δ²
    return np.where(t > delta3,
                    np.cbrt(t),
                    t / c3d2 + 4.0 / 29.0)


def _f_lab_inv(t):
    """
    Inverse of _f_lab:
        f⁻¹(t) = t³               if t > δ = 6/29
               = 3·δ²·(t − 4/29)  otherwise
    """
    delta  = 6.0 / 29.0
    c3d2   = 3.0 * delta ** 2
    return np.where(t > delta,
                    t ** 3,
                    c3d2 * (t - 4.0 / 29.0))


def rgb_to_lab(image_uint8):
    """
    sRGB uint8 [0,255] → CIE L*a*b* float32
        L ∈ [0, 100],  a ∈ [−128, 127],  b ∈ [−128, 127]

    Step 1: sRGB gamma removal (via LUT)
        c_lin = c/12.92                         if c_norm ≤ 0.04045
              = ((c_norm + 0.055) / 1.055)^2.4  otherwise

    Step 2: Linear RGB → CIE XYZ (D65, IEC 61966-2-1)
        [X]   [0.4124564  0.3575761  0.1804375] [R_lin]
        [Y] = [0.2126729  0.7151522  0.0721750] [G_lin]
        [Z]   [0.0193339  0.1191920  0.9503041] [B_lin]

    Step 3: Normalize by D65 white point
        Xn=0.95047, Yn=1.00000, Zn=1.08883

    Step 4: Apply f() nonlinearity and compute L*a*b*
        L* = 116·f(Y/Yn) − 16
        a* = 500·(f(X/Xn) − f(Y/Yn))
        b* = 200·(f(Y/Yn) − f(Z/Zn))
    """
    # Stage 1: Linearize gamma (LUT lookup per channel)
    R = _SRGB_TO_LINEAR[image_uint8[:, :, 0]]
    G = _SRGB_TO_LINEAR[image_uint8[:, :, 1]]
    B = _SRGB_TO_LINEAR[image_uint8[:, :, 2]]

    # Stage 2: Linear RGB → XYZ
    X = 0.4124564 * R + 0.3575761 * G + 0.1804375 * B
    Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B
    Z = 0.0193339 * R + 0.1191920 * G + 0.9503041 * B

    # Stage 3 + 4: Normalize and apply f()
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    fx = _f_lab(X / Xn)
    fy = _f_lab(Y / Yn)
    fz = _f_lab(Z / Zn)

    L    = 116.0 * fy - 16.0
    A_ch = 500.0 * (fx - fy)
    B_ch = 200.0 * (fy - fz)

    return L.astype(np.float32), A_ch.astype(np.float32), B_ch.astype(np.float32)


def lab_to_rgb(L, A_ch, B_ch):
    """
    CIE L*a*b* float32 → sRGB uint8  (exact inverse of rgb_to_lab)

    Step 1: Recover f() values
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy − b / 200

    Step 2: f⁻¹() → XYZ
        X = Xn · f⁻¹(fx),  Y = Yn · f⁻¹(fy),  Z = Zn · f⁻¹(fz)

    Step 3: XYZ → Linear sRGB (D65 inverse matrix)
        [R]   [ 3.2404542  −1.5371385  −0.4985314] [X]
        [G] = [−0.9692660   1.8760108   0.0415560] [Y]
        [B]   [ 0.0556434  −0.2040259   1.0572252] [Z]

    Step 4: Gamma encoding
        c_srgb = 12.92·c                        if c ≤ 0.0031308
               = 1.055·c^(1/2.4) − 0.055        otherwise
    """
    fy = (L + 16.0) / 116.0
    fx = A_ch / 500.0 + fy
    fz = fy - B_ch / 200.0

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    X = Xn * _f_lab_inv(fx)
    Y = Yn * _f_lab_inv(fy)
    Z = Zn * _f_lab_inv(fz)

    R_lin =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    G_lin = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    B_lin =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

    R_lin = np.clip(R_lin, 0.0, 1.0)
    G_lin = np.clip(G_lin, 0.0, 1.0)
    B_lin = np.clip(B_lin, 0.0, 1.0)

    def _gamma_encode(c):
        return np.where(c <= 0.0031308,
                        12.92 * c,
                        1.055 * np.power(c, 1.0 / 2.4) - 0.055)

    rgb = np.stack([_gamma_encode(R_lin),
                    _gamma_encode(G_lin),
                    _gamma_encode(B_lin)], axis=2)
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


# =============================================================================
# MODULE 3: Box Filter via Summed Area Table (SAT)
# Replaces cv2.boxFilter
#
# Hardware architecture:
#   Row SAT:  cascaded adder, 1 cycle/pixel (no multiplier)
#   Col SAT:  requires (2r+1) line buffers × width × word-width
#   Lookup:   4 BRAM reads + 3 adders per pixel output
#   Memory:   H × W × 32-bit  (720p → ~3.3 MB per SAT instance)
# =============================================================================

def box_filter_2d(src, r):
    """
    2D mean filter, window = (2r+1) × (2r+1), using integral image.

    For interior pixel (y, x):
        sum = SAT[y+r+1, x+r+1] − SAT[y−r, x+r+1]
                                 − SAT[y+r+1, x−r]
                                 + SAT[y−r,   x−r]
        mean = sum / ((2r+1)²)

    Border pixels use smaller windows (clamped indices).
    Equivalent to cv2.boxFilter(src, cv2.CV_64F, (r, r), normalize=True).
    """
    src_f = src.astype(np.float64)
    h, w  = src_f.shape

    # Build summed area table (padded with zeros at row 0 and col 0)
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(src_f, axis=0), axis=1)

    # Clamped window boundary indices (inclusive top/left, exclusive bottom/right)
    y_lo = np.maximum(np.arange(h) - r,     0)
    y_hi = np.minimum(np.arange(h) + r + 1, h)
    x_lo = np.maximum(np.arange(w) - r,     0)
    x_hi = np.minimum(np.arange(w) + r + 1, w)

    YLO, XLO = np.meshgrid(y_lo, x_lo, indexing='ij')
    YHI, XHI = np.meshgrid(y_hi, x_hi, indexing='ij')

    # 4-corner SAT lookup
    box_sum = (ii[YHI, XHI]
             - ii[YLO, XHI]
             - ii[YHI, XLO]
             + ii[YLO, XLO])

    # Normalize by actual window size (smaller at borders)
    count = ((YHI - YLO) * (XHI - XLO)).astype(np.float64)
    return box_sum / count


# =============================================================================
# MODULE 4: CLAHE — Contrast Limited Adaptive Histogram Equalization
# Replaces cv2.createCLAHE
#
# Hardware overview:
#   Phase 1  Tile analysis:   N_tile engines (parallelizable)
#              Each engine:   BRAM histogram (256×16b)
#                           + clip/redistribute logic
#                           + prefix-sum CDF
#                           → 256-entry mapping LUT (256×8b)
#   Phase 2  Reconstruction: bilinear interpolation unit
#              Per pixel:    4 LUT reads + 4 multipliers + 3 adders
# =============================================================================

def _tile_equalize(tile_pixels, clip_limit_abs, num_bins=256):
    """
    Compute the CLAHE histogram equalization mapping for one tile.

    Step 1 — Histogram accumulation:
        hist[i] = count of pixels with value i
        Hardware: 256 counters in BRAM, scan pixels sequentially (1 cycle/px)

    Step 2 — Clip and redistribute excess:
        excess       = Σ max(hist[i] − clip_limit_abs, 0)
        hist[i]      = min(hist[i], clip_limit_abs)
        add_per_bin  = excess // 256
        remainder    = excess %  256
        hist[i]     += add_per_bin
        hist[0..R-1]+= 1           (distribute R leftover counts to lowest bins)
        Hardware: comparator per bin + adder tree for redistribution

    Step 3 — CDF and normalize:
        cdf[i]     = Σ_{j=0}^{i} hist[j]   (prefix sum)
        mapping[i] = round(cdf[i] × 255 / cdf[255])
        Hardware: carry-propagate adder (log₂256 = 8 stages) + multiplier
    """
    # Step 1: Histogram
    hist, _ = np.histogram(tile_pixels, bins=num_bins, range=(0, 256))
    hist = hist.astype(np.int32)

    # Step 2: Clip and redistribute
    excess       = int(np.sum(np.maximum(hist - clip_limit_abs, 0)))
    hist         = np.minimum(hist, clip_limit_abs)
    add_per_bin  = excess // num_bins
    remainder    = excess %  num_bins
    hist        += add_per_bin
    hist[:remainder] += 1           # remainder to lowest bins

    # Step 3: CDF → mapping [0, 255]
    cdf   = np.cumsum(hist)
    total = int(cdf[-1])
    if total == 0:
        return np.arange(num_bins, dtype=np.uint8)
    mapping = np.round(cdf.astype(np.float32) * (255.0 / total)).astype(np.uint8)
    return mapping


def clahe_channel(channel_uint8, clip_limit=2.0, tile_size=16):
    """
    Apply CLAHE to a single uint8 channel.

    Phase 1 — Per-tile mapping LUT computation:
        clip_limit_abs = max(1, round(clip_limit × tile_size² / 256))
        For each tile (ty, tx):
            tile_lut[ty, tx] = _tile_equalize(tile_pixels, clip_limit_abs)

    Phase 2 — Bilinear reconstruction:
        Tile center of tile (ty, tx) is at pixel
            (ty·tile_size + tile_size/2,  tx·tile_size + tile_size/2)

        For pixel (y, x) with value p:
            cy = (y − tile_size/2) / tile_size   ← continuous tile coord
            cx = (x − tile_size/2) / tile_size
            ty0 = ⌊cy⌋,  wy1 = cy − ty0,  wy0 = 1 − wy1
            tx0 = ⌊cx⌋,  wx1 = cx − tx0,  wx0 = 1 − wx1
            out = wy0·wx0·M[ty0,tx0,p]   (top-left  tile mapping)
                + wy0·wx1·M[ty0,tx1,p]   (top-right tile mapping)
                + wy1·wx0·M[ty1,tx0,p]   (bot-left  tile mapping)
                + wy1·wx1·M[ty1,tx1,p]   (bot-right tile mapping)

        At image borders: tile indices clamped → effectively nearest-tile copy.

    Hardware memory (1080p, tile=16):
        Tile grid : 68 × 120 = 8,160 tiles
        LUT RAM   : 8,160 × 256 B = 2.04 MB  (on-chip SRAM or external SDRAM)
    """
    H, W = channel_uint8.shape
    num_bins       = 256
    clip_limit_abs = max(1, int(clip_limit * tile_size * tile_size / num_bins))

    n_ty = (H + tile_size - 1) // tile_size
    n_tx = (W + tile_size - 1) // tile_size

    # Reflect-pad so tiles cover the full image evenly
    ph     = n_ty * tile_size - H
    pw     = n_tx * tile_size - W
    padded = np.pad(channel_uint8, ((0, ph), (0, pw)), mode='reflect')

    # Phase 1: compute 256-entry mapping LUT for every tile
    tile_lut = np.zeros((n_ty, n_tx, num_bins), dtype=np.uint8)
    for ty in range(n_ty):
        for tx in range(n_tx):
            tile = padded[ty * tile_size:(ty + 1) * tile_size,
                          tx * tile_size:(tx + 1) * tile_size]
            tile_lut[ty, tx] = _tile_equalize(tile.ravel(), clip_limit_abs)

    # Phase 2: bilinear interpolation (fully vectorized for SW speed)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    cy  = (yy.astype(np.float32) - tile_size * 0.5) / tile_size
    cx  = (xx.astype(np.float32) - tile_size * 0.5) / tile_size

    ty0 = np.floor(cy).astype(np.int32)
    tx0 = np.floor(cx).astype(np.int32)
    wy1 = (cy - ty0).astype(np.float32)
    wx1 = (cx - tx0).astype(np.float32)
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1

    ty0c = np.clip(ty0,     0, n_ty - 1)
    ty1c = np.clip(ty0 + 1, 0, n_ty - 1)
    tx0c = np.clip(tx0,     0, n_tx - 1)
    tx1c = np.clip(tx0 + 1, 0, n_tx - 1)

    pix = channel_uint8   # H×W, used as LUT index

    # 4-corner lookup
    v00 = tile_lut[ty0c, tx0c, pix].astype(np.float32)
    v01 = tile_lut[ty0c, tx1c, pix].astype(np.float32)
    v10 = tile_lut[ty1c, tx0c, pix].astype(np.float32)
    v11 = tile_lut[ty1c, tx1c, pix].astype(np.float32)

    out = wy0 * wx0 * v00 + wy0 * wx1 * v01 + wy1 * wx0 * v10 + wy1 * wx1 * v11
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def apply_clahe(image, clip_limit=2.0, tile_size=16):
    """
    CLAHE applied in LAB space (L channel only).
    Drop-in replacement for the cv2.createCLAHE version.

    Pipeline: RGB → LAB → scale L to [0,255] → CLAHE → scale back → RGB

    Why LAB: CLAHE on L* only enhances luminance contrast
             without distorting hue or saturation.
    """
    L, A_ch, B_ch = rgb_to_lab(image)

    # Scale L: [0, 100] → [0, 255]  (8-bit histogram domain)
    L_u8 = np.clip(np.round(L * (255.0 / 100.0)), 0, 255).astype(np.uint8)
    L_eq = clahe_channel(L_u8, clip_limit=clip_limit, tile_size=tile_size)

    # Scale back: [0, 255] → [0, 100]
    L_out = L_eq.astype(np.float32) * (100.0 / 255.0)
    return lab_to_rgb(L_out, A_ch, B_ch)


# =============================================================================
# MODULE 5: Gamma Correction LUT
# Replaces cv2.LUT
#
# Hardware: 256 × 8-bit ROM, addressed by pixel value, 1-cycle latency
# =============================================================================

def apply_gamma_correction(image, gamma=1.0):
    """
    Per-pixel gamma correction via lookup table.
        mapping[i] = round((i / 255)^(1/gamma) × 255)

    Hardware: 256-word × 8-bit ROM loaded at startup.
    Apply: out_pixel = ROM[in_pixel]   (1-cycle latency, no arithmetic)
    """
    if gamma == 1.0:
        return image

    inv_gamma = 1.0 / gamma
    table = np.array(
        [int(round(((i / 255.0) ** inv_gamma) * 255)) for i in range(256)],
        dtype=np.uint8
    )
    # numpy fancy indexing ≡ cv2.LUT
    return table[image]


# =============================================================================
# MODULE 6: Guided Filter
# Replaces cv2.boxFilter inside guided_filter
#
# Algorithm (He et al., 2013 — TPAMI):
#   mean_I  = box(I,  r)
#   mean_p  = box(p,  r)
#   cov_Ip  = box(I·p, r) − mean_I · mean_p
#   var_I   = box(I², r)  − mean_I²
#   a       = cov_Ip / (var_I + ε)
#   b       = mean_p − a · mean_I
#   q       = box(a, r) · I + box(b, r)
#
# Hardware:
#   5 × SAT box filter units (each needs (2r+1) line buffers × width × 64b)
#   For r=15, 720p: 31 rows × 1280 × 8B ≈ 316 KB per SAT unit → ~1.6 MB total
#   Elementwise ops: 5 multipliers + 5 adders per output pixel
# =============================================================================

def guided_filter(I, p, r=15, eps=0.01):
    """
    Edge-preserving guided filter.
    Smooths transmission map t while preserving scene edges.
    Eliminates halo artifacts caused by naive box-filtering of t.
    """
    I = I.astype(np.float64)
    p = p.astype(np.float64)

    mean_I  = box_filter_2d(I,     r)
    mean_p  = box_filter_2d(p,     r)
    mean_Ip = box_filter_2d(I * p, r)
    cov_Ip  = mean_Ip - mean_I * mean_p

    mean_II = box_filter_2d(I * I, r)
    var_I   = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter_2d(a, r)
    mean_b = box_filter_2d(b, r)

    return mean_a * I + mean_b


# =============================================================================
# CORE ALGORITHM: AVSD Defogging (identical logic to method7)
# No cv2 usage in this section; hardware notes added for new modules above.
#
# Physics model:
#   H(x) = D(x)·t(x) + A·(1−t(x))
#   D(x) = (H(x) − A) / t(x) + A
#
# Transmission map:
#   t = (3K + 3m − ψ·3·K·m) / (3K + 3m)
#   K = mean(H/A),  m = min(H/A)
# =============================================================================

def predict_psi(image):
    """
    Predict best PSI from hardware fog score.
    Regression: BestPsi = 0.009308 × fog_score + 0.927009
    Clipped to [0.5, 1.2]

    Hardware: min/max tree over pixel scan (H×W cycles), then
              two multiply-add operations.
    """
    gray = image[:, :, 0]
    dynamic_range = float(np.max(gray)) - float(np.min(gray))

    if   dynamic_range >= 240: fog_score = 0
    elif dynamic_range <= 100: fog_score = 100
    else:                      fog_score = int(240 - dynamic_range) >> 1

    fog_score = max(0, min(100, fog_score))
    BestPsi   = float(np.clip(0.009308 * fog_score + 0.927009, 0.5, 1.2))
    return BestPsi


def defog_img_basic(hazy_image, psi=1, t0=0.2, window_size=8, epsilon=1e-6):
    """
    Core AVSD defogging — no post-processing.
    Returns (D, A, BestPsi, t).
    """
    H       = hazy_image.astype(np.float32)
    BestPsi = predict_psi(H)
    psi     = BestPsi

    dark_channel = np.min(H, axis=2)
    dark_min     = minimum_filter(dark_channel, size=window_size)
    y, x         = np.unravel_index(np.argmax(dark_min), dark_min.shape)
    A            = H[y, x, :].copy()

    H_norm   = H / (A + epsilon)
    K        = np.mean(H_norm, axis=2)
    min_norm = np.min(H_norm,  axis=2)
    temp     = 3 * K + 3 * min_norm
    t        = (temp - psi * 3 * K * min_norm) / (temp + epsilon)
    t        = np.clip(t, t0, 1)

    t_exp = t[:, :, np.newaxis]
    D     = np.clip((H - A) / t_exp + A, 0, 255).astype(np.uint8)
    return D, A, BestPsi, t


def defog_img_enhanced(hazy_image,
                       psi=1,
                       t0=0.2,
                       window_size=8,
                       use_clahe=True,
                       clahe_clip_limit=2.0,
                       clahe_tile_size=16,
                       use_gamma=False,
                       gamma_value=1.0,
                       use_guided_filter=False,
                       guided_filter_radius=15):
    """
    Enhanced AVSD defogging with optional post-processing.
    All cv2 calls replaced — suitable as hardware reference model.
    """
    H       = hazy_image.astype(np.float32)
    epsilon = 1e-6
    BestPsi = predict_psi(H)
    psi     = BestPsi

    dark_channel = np.min(H, axis=2)
    dark_min     = minimum_filter(dark_channel, size=window_size)
    y, x         = np.unravel_index(np.argmax(dark_min), dark_min.shape)
    A            = H[y, x, :].copy()

    H_norm   = H / (A + epsilon)
    K        = np.mean(H_norm, axis=2)
    min_norm = np.min(H_norm,  axis=2)
    temp     = 3 * K + 3 * min_norm
    t        = (temp - psi * 3 * K * min_norm) / (temp + epsilon)
    t        = np.clip(t, t0, 1)

    if use_guided_filter:
        # Grayscale: BT.601 luma  (replaces cv2.COLOR_RGB2GRAY)
        gray = (0.299 * hazy_image[:, :, 0].astype(np.float64)
              + 0.587 * hazy_image[:, :, 1].astype(np.float64)
              + 0.114 * hazy_image[:, :, 2].astype(np.float64)) / 255.0
        t = guided_filter(gray, t, r=guided_filter_radius, eps=0.01)
        t = np.clip(t, t0, 1)

    t_exp = t[:, :, np.newaxis]
    D     = np.clip((H - A) / t_exp + A, 0, 255).astype(np.uint8)

    if use_gamma and gamma_value != 1.0:
        D = apply_gamma_correction(D, gamma_value)

    if use_clahe:
        D = apply_clahe(D, clip_limit=clahe_clip_limit, tile_size=clahe_tile_size)

    info = {
        'psi':                BestPsi,
        'airlight':           A,
        'used_clahe':         use_clahe,
        'used_gamma':         use_gamma,
        'used_guided_filter': use_guided_filter,
        't_mean':             float(np.mean(t)),
        't_min':              float(np.min(t)),
        't_max':              float(np.max(t)),
    }
    return D, A, BestPsi, info


def defog_img(hazy_image, psi=1, t0=0.2, window_size=8, epsilon=1e-6):
    """Public API (backward compatible with method7)."""
    D, A, BestPsi, _ = defog_img_enhanced(
        hazy_image, psi=psi, t0=t0, window_size=window_size,
        use_clahe=True, clahe_clip_limit=2.0, clahe_tile_size=16,
        use_gamma=False, use_guided_filter=False
    )
    return D, A, BestPsi


# ── Preset configurations (identical to method7) ──────────────────────────────

def defog_best_quality(hazy_image):
    """Best-quality config (recommended for evaluation)."""
    return defog_img_enhanced(hazy_image,
                              use_clahe=True, clahe_clip_limit=2.0,
                              clahe_tile_size=16, use_gamma=False,
                              use_guided_filter=False)


def defog_high_contrast(hazy_image):
    """High-contrast config (dense fog)."""
    return defog_img_enhanced(hazy_image, t0=0.1,
                              use_clahe=True, clahe_clip_limit=3.0,
                              clahe_tile_size=8, use_gamma=False,
                              use_guided_filter=True, guided_filter_radius=15)


def defog_conservative(hazy_image):
    """Conservative config (light fog)."""
    return defog_img_enhanced(hazy_image, t0=0.3,
                              use_clahe=True, clahe_clip_limit=1.5,
                              clahe_tile_size=16, use_gamma=False,
                              use_guided_filter=False)


def defog_fast(hazy_image):
    """Fast mode (no post-processing)."""
    return defog_img_basic(hazy_image)[:3]


def batch_process(image_list, config='best'):
    configs = {
        'best':          defog_best_quality,
        'high_contrast': defog_high_contrast,
        'conservative':  defog_conservative,
        'fast':          defog_fast,
    }
    func = configs.get(config, defog_best_quality)
    return [func(img) for img in image_list]


# =============================================================================
# Evaluation (same as method7, no cv2 in this section)
# =============================================================================

if __name__ == "__main__":
    import os
    import time
    from PIL import Image
    from glob import glob
    from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
    from skimage.metrics import structural_similarity as calculate_ssim
    from skimage.color import rgb2lab, deltaE_ciede2000
    import pandas as pd
    from tqdm import tqdm

    defog_version = "defog_v6_method7_hw"
    datasets = ["OHaze", "SOTS_out", "SOTS_in"]

    targets = {
        "OHaze":    {"PSNR": 16.7290, "SSIM": 0.5942, "CIEDE2000": 15.3479},
        "SOTS_out": {"PSNR": 22.1355, "SSIM": 0.8840, "CIEDE2000":  6.0956},
        "SOTS_in":  {"PSNR": 17.1906, "SSIM": 0.7856, "CIEDE2000": 10.4843},
    }

    def compute_psnr(defogged_image, clear_image_path, Xsize, Ysize):
        clear_img = Image.open(clear_image_path).convert('RGB')
        if clear_img.width != Xsize or clear_img.height != Ysize:
            clear_img = clear_img.resize((Xsize, Ysize))
        clear_array = np.array(clear_img)
        if defogged_image.shape != clear_array.shape:
            mh = min(defogged_image.shape[0], clear_array.shape[0])
            mw = min(defogged_image.shape[1], clear_array.shape[1])
            defogged_image, clear_array = defogged_image[:mh, :mw], clear_array[:mh, :mw]
        try:    return calculate_psnr(clear_array, defogged_image)
        except: return 0

    def compute_ssim(defogged_image, clear_image_path, Xsize, Ysize):
        clear_img = Image.open(clear_image_path).convert('RGB')
        if clear_img.width != Xsize or clear_img.height != Ysize:
            clear_img = clear_img.resize((Xsize, Ysize))
        clear_array = np.array(clear_img)
        if defogged_image.shape != clear_array.shape:
            mh = min(defogged_image.shape[0], clear_array.shape[0])
            mw = min(defogged_image.shape[1], clear_array.shape[1])
            defogged_image, clear_array = defogged_image[:mh, :mw], clear_array[:mh, :mw]
        try:    return calculate_ssim(clear_array, defogged_image, channel_axis=-1)
        except: return 0

    def compute_ciede2000(defogged_image, clear_image_path, Xsize, Ysize):
        clear_img = Image.open(clear_image_path).convert('RGB')
        if clear_img.width != Xsize or clear_img.height != Ysize:
            clear_img = clear_img.resize((Xsize, Ysize))
        clear_array = np.array(clear_img)
        if defogged_image.shape != clear_array.shape:
            mh = min(defogged_image.shape[0], clear_array.shape[0])
            mw = min(defogged_image.shape[1], clear_array.shape[1])
            defogged_image, clear_array = defogged_image[:mh, :mw], clear_array[:mh, :mw]
        try:    return float(np.mean(deltaE_ciede2000(rgb2lab(clear_array), rgb2lab(defogged_image))))
        except: return 0

    def main(dataset):
        hazy_dir = f"./dataset/{dataset}/hazy"
        out_dir  = f"./dataset/{dataset}/result_{defog_version}"
        os.makedirs(out_dir, exist_ok=True)
        hazy_files = sorted(glob(os.path.join(hazy_dir, "*.png")))

        for hazy_path in hazy_files:
            full_name = os.path.splitext(os.path.basename(hazy_path))[0]
            base_name = full_name.split('_')[0]
            out_path  = os.path.join(out_dir, f"{base_name}_{defog_version}.png")
            try:
                img = Image.open(hazy_path).convert('RGB')
                H   = np.array(img)
                t0  = time.time()
                D, A, BestPsi = defog_img(H)
                elapsed = time.time() - t0
                Image.fromarray(D).save(out_path)
            except Exception:
                import traceback; traceback.print_exc()

    def score(dataset):
        clear_dir   = f"./dataset/{dataset}/clear"
        defog_dir   = f"./dataset/{dataset}/result_{defog_version}"
        defog_files = sorted(glob(os.path.join(defog_dir, "*.png")))

        results    = []
        avg_scores = {"PSNR": 0, "SSIM": 0, "CIEDE2000": 0}
        total      = 0

        for dp in tqdm(defog_files, desc=f"Scoring {dataset}"):
            base = os.path.splitext(os.path.basename(dp))[0].split('_')[0]
            cp   = os.path.join(clear_dir, f"{base}_clear.png")
            if not os.path.exists(cp):
                continue
            da = np.array(Image.open(dp).convert('RGB'))
            Xs, Ys = da.shape[1], da.shape[0]
            psnr  = compute_psnr(da, cp, Xs, Ys)
            ssim  = compute_ssim(da, cp, Xs, Ys)
            ciede = compute_ciede2000(da, cp, Xs, Ys)
            results.append({"Image": base, "PSNR": psnr, "SSIM": ssim, "CIEDE2000": ciede})
            total += 1
            for k, v in zip(["PSNR", "SSIM", "CIEDE2000"], [psnr, ssim, ciede]):
                avg_scores[k] = (avg_scores[k] * (total - 1) + v) / total

        if total > 0:
            df = pd.concat([pd.DataFrame(results),
                            pd.DataFrame([{"Image": "AVERAGE", **avg_scores}])],
                           ignore_index=True)
            os.makedirs(f"./dataset/{dataset}/report", exist_ok=True)
            df.to_csv(f"./dataset/{dataset}/report/score_{defog_version}.csv",
                      index=False, float_format="%.4f")
        return avg_scores if total > 0 else None

    all_scores = {}
    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"### Processing: {dataset}")
        print(f"{'#'*60}")
        main(dataset)
        avg = score(dataset)
        if avg:
            all_scores[dataset] = avg

    print(f"\n\n{'='*70}")
    print(f"Method 7 HW Summary vs Targets")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} | {'PSNR':>8} ({'Tgt':>8}) | {'SSIM':>8} ({'Tgt':>8}) | {'CIEDE':>8} ({'Tgt':>8})")
    print(f"{'-'*15}-+-{'-'*19}-+-{'-'*19}-+-{'-'*19}")
    for ds in datasets:
        if ds in all_scores:
            s = all_scores[ds]
            t = targets[ds]
            p_ok = "+" if s["PSNR"]      > t["PSNR"]      else "-"
            s_ok = "+" if s["SSIM"]      > t["SSIM"]      else "-"
            c_ok = "+" if s["CIEDE2000"] < t["CIEDE2000"] else "-"
            print(f"{ds:<15} | {s['PSNR']:>7.4f}{p_ok} ({t['PSNR']:>7.4f}) "
                  f"| {s['SSIM']:>7.4f}{s_ok} ({t['SSIM']:>7.4f}) "
                  f"| {s['CIEDE2000']:>7.4f}{c_ok} ({t['CIEDE2000']:>7.4f})")
    print(f"{'='*70}")
