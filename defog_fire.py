 # defog_2023.py
# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import minimum_filter, uniform_filter

_EPS = 1e-6

# ---------- 基本工具 ----------
def _to_float(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)

def _to_uint8(img_f32):
    img = np.clip(img_f32 * 255.0, 0, 255)
    return img.astype(np.uint8)

def _rgb_to_y(img):
    return 0.299 * img[...,0] + 0.587 * img[...,1] + 0.114 * img[...,2]

def _rgb_to_yuv(img):
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.14713 * R - 0.28886 * G + 0.436 * B
    V = 0.615 * R - 0.51499 * G - 0.10001 * B
    return np.stack([Y, U, V], axis=-1)

def _yuv_to_rgb(yuv):
    Y, U, V = yuv[..., 0], yuv[..., 1], yuv[..., 2]
    R = Y + 1.13983 * V
    G = Y - 0.39465 * U - 0.58060 * V
    B = Y + 2.03211 * U
    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0.0, 1.0)

def _hist_equalize_channel(ch):
    ch = np.clip(ch, 0.0, 1.0)
    hist, bins = np.histogram(ch.flatten(), bins=256, range=(0.0, 1.0))
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) / max(cdf.max() - cdf.min(), _EPS)
    return np.interp(ch.flatten(), bins[:-1], cdf).reshape(ch.shape)

def _saturate(rgb, gain=1.06):
    yuv = _rgb_to_yuv(rgb)
    yuv[...,1] *= gain
    yuv[...,2] *= gain
    return np.clip(_yuv_to_rgb(yuv), 0.0, 1.0)

def _dark_channel(norm_img, size=15):
    min_per_pixel = np.min(norm_img, axis=2)
    return minimum_filter(min_per_pixel, size=size, mode='nearest')

def _bright_channel(norm_img, size=15):
    max_per_pixel = np.max(norm_img, axis=2)
    return -minimum_filter(-max_per_pixel, size=size, mode='nearest')  # local max

def _guided_filter_gray(guide, src, radius=20, eps=1e-3):
    r = int(radius)
    def boxf(x): return uniform_filter(x, size=2*r+1, mode='nearest')
    I, P = guide, src
    mean_I, mean_P = boxf(I), boxf(P)
    corr_I, corr_IP = boxf(I*I), boxf(I*P)
    var_I = corr_I - mean_I*mean_I
    cov_IP = corr_IP - mean_I*mean_P
    a = cov_IP / (var_I + eps)
    b = mean_P - a * mean_I
    mean_a, mean_b = boxf(a), boxf(b)
    return mean_a * I + mean_b

def _grad_mag(gray):
    dx = np.zeros_like(gray); dy = np.zeros_like(gray)
    dx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    dy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    g = np.sqrt(dx*dx + dy*dy)
    g /= (g.max() + _EPS)
    return g

def _local_variance(gray, r=5):
    mean = uniform_filter(gray, size=2*r+1, mode='nearest')
    mean2 = uniform_filter(gray*gray, size=2*r+1, mode='nearest')
    var = np.clip(mean2 - mean*mean, 0.0, 1.0)
    vnorm = var / (var.max() + _EPS)
    return vnorm

# ---------- 大氣光（分群平均） ----------
def _estimate_atmospheric_light(img, patch=15, top_percent=0.005, downsample=2,
                                kmeans_k=3, kmeans_iters=10):
    img_ds = img[::downsample, ::downsample, :]
    dc = _dark_channel(img_ds, size=patch)
    thresh = np.quantile(dc, 1.0 - top_percent)
    mask = dc >= thresh
    candidates = img_ds[mask]
    if candidates.size == 0:
        flat = img_ds.reshape(-1, 3)
        lum = flat.sum(axis=1)
        idx = np.argsort(lum)[int(0.995 * len(lum)):]
        return np.clip(np.mean(flat[idx], axis=0), 0.03, 1.0)
    X = candidates
    rng = np.random.default_rng(42)
    centers = (np.vstack([X[rng.integers(0, len(X))] for _ in range(kmeans_k)])
               if len(X) < kmeans_k else
               X[rng.choice(len(X), size=kmeans_k, replace=False)])
    for _ in range(kmeans_iters):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        new_centers = []
        for ki in range(kmeans_k):
            sel = X[labels == ki]
            new_centers.append(centers[ki] if len(sel) == 0 else sel.mean(axis=0))
        new_centers = np.stack(new_centers, axis=0)
        if np.allclose(new_centers, centers, atol=1e-4): break
        centers = new_centers
    brightness = centers.sum(axis=1)
    best_k = int(np.argmax(brightness))
    sel = X[labels == best_k]
    A = centers[best_k] if len(sel) == 0 else sel.mean(axis=0)
    return np.clip(A, 0.03, 1.0)

# ---------- t(x) 初估（混合下界，自適應 ω） ----------
def _estimate_transmission(img, A, patch=15, omega_min=0.85, omega_max=0.95, t0=0.34):
    norm = img / (A[None, None, :] + _EPS)
    min_c = np.min(norm, axis=2)
    min_patch = minimum_filter(min_c, size=patch, mode='nearest')
    mean_patch = uniform_filter(min_c, size=patch, mode='nearest')
    bright_loc = np.clip(_bright_channel(norm, size=patch), 0.0, 1.0)
    omega = omega_max - (omega_max - omega_min) * bright_loc
    t1 = 1.0 - omega * min_patch
    t2 = 1.0 - omega * mean_patch
    t = np.maximum(t1, t2)
    return np.clip(t, t0, 1.0)

# ---------- t(x) 邊緣自適應精煉（抑制暈白） ----------
def _refine_t_edgeaware(t_coarse, guide_gray,
                        r_small=10, eps_small=1e-4,
                        r_large=28, eps_large=5e-3,
                        mix_strength=0.9):
    t_s = _guided_filter_gray(guide_gray, t_coarse, radius=r_small, eps=eps_small)
    t_l = _guided_filter_gray(guide_gray, t_coarse, radius=r_large, eps=eps_large)
    edge = _grad_mag(guide_gray)          # 0~1
    w = np.clip(edge * mix_strength, 0.0, 1.0)
    t_ref = w * t_s + (1.0 - w) * t_l
    return np.clip(t_ref, t_coarse.min(), 1.0)

# ---------- 噪聲自適應平滑（只在厚煙+低方差） ----------
def _denoise_adaptive(rgb, guide_gray, t, strength=0.6, r=6, eps=1e-3):
    var = _local_variance(guide_gray, r=3)
    noise_w = (1.0 - var) * np.power(1.0 - t, 1.2)
    noise_w = np.clip(noise_w * strength, 0.0, 1.0)
    out = np.empty_like(rgb)
    for c in range(3):
        base = _guided_filter_gray(guide_gray, rgb[..., c], radius=r, eps=eps)
        out[..., c] = (1.0 - noise_w) * rgb[..., c] + noise_w * base
    return np.clip(out, 0.0, 1.0)

# ---------- 受控銳化（避免白邊） ----------
def _unsharp_on_y_adaptive(img_rgb, t, amount=0.45, r_base=8, eps_base=1e-3,
                           detail_clip=0.05, edge_protect=0.6):
    yuv = _rgb_to_yuv(img_rgb)
    Y = yuv[..., 0]
    base = _guided_filter_gray(Y, Y, radius=r_base, eps=eps_base)
    detail = Y - base
    var = _local_variance(Y, r=3)
    edge = _grad_mag(Y)
    mask = (var ** 0.8) * (t ** 0.6) * (1.0 - edge_protect * edge)
    detail = np.clip(detail, -detail_clip, detail_clip)
    Y_sharp = np.clip(Y + amount * mask * detail, 0.0, 1.0)
    yuv[..., 0] = Y_sharp
    return _yuv_to_rgb(yuv)

# ---------- 火焰軟遮罩 ----------
def _flame_soft_mask(img, ksize=9, r_min=0.75, dom_rg=1.10, dom_rb=1.25, lum_sum=1.8):
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    cond = (R > r_min) & (R > dom_rg * (G + _EPS)) & (R > dom_rb * (B + _EPS)) & ((R + G + B) > lum_sum)
    m = uniform_filter(cond.astype(np.float32), size=ksize, mode='nearest')
    return np.clip(m, 0.0, 1.0)

# ---------- 多尺度細節增強（邊界銳利度提升） ----------
def _ms_detail_boost_y(img_rgb, t,
                       amounts=(0.55, 0.35),
                       radii=(2, 6),
                       eps=(1e-4, 4e-4),
                       clip=(0.035, 0.05)):
    yuv = _rgb_to_yuv(img_rgb)
    Y = yuv[..., 0]
    B1 = _guided_filter_gray(Y, Y, radius=radii[0], eps=eps[0])
    B2 = _guided_filter_gray(Y, Y, radius=radii[1], eps=eps[1])
    d1 = np.clip(Y - B1, -clip[0], clip[0])      # 高頻
    d2 = np.clip(B1 - B2, -clip[1], clip[1])     # 中字頻
    var = _local_variance(Y, r=3)
    edge = _grad_mag(Y)
    w = (var ** 0.8) * (t ** 0.7) * (1.0 - 0.3 * edge)
    Yb = np.clip(Y + w * (amounts[0] * d1 + amounts[1] * d2), 0.0, 1.0)
    yuv[..., 0] = Yb
    return _yuv_to_rgb(yuv)

# ---------- 8×8 自適應去方塊（新增；減少「方塊感」與邊界模糊） ----------
def _deblock_8x8(rgb, luma=None, alpha=0.6, diff_thr=0.02, var_thr=0.008):
    """
    只在疑似壓縮方塊邊界上做方向性融合：
      - 若跨邊界亮度差小（非真邊緣）且局部方差低 -> 融合兩側像素
      - 否則保留（避免真邊緣被抹）
    alpha: 融合強度
    diff_thr: 邊界兩側允許的亮度差門檻（越小越保守）
    var_thr: 局部方差門檻（越小越只在平坦區作用）
    """
    H, W, _ = rgb.shape
    if luma is None:
        luma = _rgb_to_y(rgb)
    # 局部方差圖
    var = uniform_filter(luma*luma, size=9, mode='nearest') - \
          uniform_filter(luma, size=9, mode='nearest')**2
    var = np.clip(var, 0.0, 1.0)

    out = rgb.copy()

    # 垂直方塊邊界（x % 8 == 0）
    for x in range(8, W, 8):
        if x-2 < 0 or x+1 >= W: continue
        l = luma[:, x-1]; r = luma[:, x]
        diff = np.abs(r - l)
        m = (diff < diff_thr) & (var[:, x] < var_thr)
        if not np.any(m): continue
        m3 = m[:, None]
        left_avg  = (rgb[:, x-2, :] + rgb[:, x,   :]) * 0.5
        right_avg = (rgb[:, x-1, :] + rgb[:, x+1, :]) * 0.5
        out[:, x-1, :] = np.where(m3, (1-alpha)*rgb[:, x-1, :] + alpha*left_avg,  out[:, x-1, :])
        out[:, x,   :] = np.where(m3, (1-alpha)*rgb[:, x,   :] + alpha*right_avg, out[:, x,   :])

    # 水平方塊邊界（y % 8 == 0）
    for y in range(8, H, 8):
        if y-2 < 0 or y+1 >= H: continue
        u = luma[y-1, :]; d = luma[y, :]
        diff = np.abs(d - u)
        m = (diff < diff_thr) & (var[y, :] < var_thr)
        if not np.any(m): continue
        m3 = m[None, :, None]
        up_avg   = (rgb[y-2, :, :] + rgb[y,   :, :]) * 0.5
        down_avg = (rgb[y-1, :, :] + rgb[y+1, :, :]) * 0.5
        out[y-1, :, :] = np.where(m3, (1-alpha)*rgb[y-1, :, :] + alpha*up_avg,   out[y-1, :, :])
        out[y,   :, :] = np.where(m3, (1-alpha)*rgb[y,   :, :] + alpha*down_avg, out[y,   :, :])

    return np.clip(out, 0.0, 1.0)

# ---------- 主流程 ----------
def defog_img(
    hazy_image,
    patch=15,
    downsample=2,
    top_percent=0.005,
    kmeans_k=3,
    kmeans_iters=10,
    omega_min=0.85,
    omega_max=0.95,
    t0=0.34,
    guide_radius_small=10,
    guide_eps_small=1e-4,
    guide_radius_large=28,
    guide_eps_large=5e-3,
    mix_strength=0.9,
    gamma_up=0.9,
    he_blend=0.62,      # 降一點等化強度，讓細節保持
    flame_blend=0.5,
    shadow_lift=0.04,
    max_gain=3.2,
    denoise_strength=0.55,   # 降低去噪以保細節
    sharpen_amount=0.5,
    # 多尺度細節增強
    ms_amounts=(0.60, 0.38),
    ms_radii=(2, 6),
    ms_eps=(1e-4, 4e-4),
    ms_clip=(0.032, 0.045),
    # 去方塊參數
    deblock_alpha=0.65,
    deblock_diff_thr=0.018,
    deblock_var_thr=0.006,
):
    """
    回傳：
        out_uint8: 去霧 + 防暈白 + 去方塊 + 提升邊界銳利度 的圖 (uint8)
        A: 大氣光 (float32, 3) in [0,1]
    """
    H = _to_float(hazy_image)

    # 1) A
    A = _estimate_atmospheric_light(H, patch=patch, top_percent=top_percent,
                                    downsample=downsample, kmeans_k=kmeans_k, kmeans_iters=kmeans_iters).astype(np.float32)

    # 2) t 初估
    t_coarse = _estimate_transmission(H, A, patch=patch, omega_min=omega_min, omega_max=omega_max, t0=t0)

    # 3) t 精煉（邊緣自適應）
    guide_gray = _rgb_to_y(H)
    t_ref = _refine_t_edgeaware(t_coarse, guide_gray,
                                r_small=guide_radius_small, eps_small=guide_eps_small,
                                r_large=guide_radius_large, eps_large=guide_eps_large,
                                mix_strength=mix_strength)
    t = np.clip(t_ref, t0, 1.0)

    # 4) 還原（限制最大放大量）
    gain = 1.0 / np.maximum(t, _EPS)
    gain = np.clip(gain, 1.0, max_gain)
    J = (H - A[None, None, :]) * gain[..., None] + A[None, None, :]
    J = np.clip(J, 0.0, 1.0)

    # 5) 噪聲自適應平滑
    J = _denoise_adaptive(J, guide_gray=_rgb_to_y(J), t=t,
                          strength=denoise_strength, r=6, eps=1e-3)

    # 6) 厚煙區微提亮
    J = np.clip(J + (shadow_lift * (1.0 - t)**2)[..., None], 0.0, 1.0)

    # 7) 火焰保護混合
    if flame_blend > 0.0:
        m = _flame_soft_mask(H, ksize=9)
        J = np.clip((1.0 - flame_blend*m[...,None]) * J + (flame_blend*m[...,None]) * H, 0.0, 1.0)

    # 8) 溫和亮度/對比（γ + 邊緣抑制等化）
    yuv = _rgb_to_yuv(J)
    Y = np.clip(yuv[...,0], 0.0, 1.0) ** gamma_up
    Y_eq = _hist_equalize_channel(Y)
    edge = _grad_mag(Y)
    he_local = he_blend * (1.0 - 0.7 * edge)
    yuv[...,0] = (1.0 - he_local) * Y + he_local * Y_eq
    J2 = _yuv_to_rgb(yuv)

    # 9) 先去方塊（只動 8×8 邊界、保邊緣）
    J2 = _deblock_8x8(J2, luma=_rgb_to_y(J2),
                      alpha=deblock_alpha, diff_thr=deblock_diff_thr, var_thr=deblock_var_thr)

    # 10) 輕度飽和度
    J2 = _saturate(J2, gain=1.06)

    # 11) 多尺度細節增強（在去方塊後）
    J3 = _ms_detail_boost_y(J2, t, amounts=ms_amounts, radii=ms_radii, eps=ms_eps, clip=ms_clip)

    # 12) 受控銳化（少量，避免白邊）
    J_sharp = _unsharp_on_y_adaptive(J3, t, amount=sharpen_amount, r_base=8, eps_base=1e-3,
                                     detail_clip=0.045, edge_protect=0.65)

    return _to_uint8(J_sharp), A
