#!/usr/bin/env python3
"""
Hardware-friendly ψ (haze strength) predictor training script — v2

✅ 支援兩種資料來源：
A) 目錄分級：
   dataset_root/
     0.5/*.png 0.6/*.png ... 1.7/*.png  （資料夾名即 ψ 標籤）
B) CSV 標註：
   --csv /path/to/score_optimize_psi_grid.csv  +  --hazy_dir /path/to/hazy
   讀取 CSV 欄位 Image, BestPsi，對應檔名 `${Image}_hazy.png`。

流程：
1) 讀資料（A 或 B）
2) 估 A（暗通道 + top-k 平均）
3) 萃取「可硬體化」全域特徵
4) K-Fold 評估（Linear & RandomForest）
5) 以 Linear+StandardScaler 當最終模型
6) 匯出：psi_linear_float.json / psi_linear_q.json / psi_linear_q.h（可直接接韌體/RTL）

依賴：numpy, pillow, scipy(建議), scikit-learn
"""
import os
import json
import glob
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

try:
    from scipy.ndimage import minimum_filter
except Exception:
    minimum_filter = None
    print("[WARN] scipy not available; using numpy fallback for dark channel (slower)")

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ----------------------------- IO ----------------------------- #

def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.asarray(img, dtype=np.uint8)


def downsample2(img: np.ndarray) -> np.ndarray:
    return img[::2, ::2, :]


def dark_channel(img: np.ndarray, win: int = 15) -> np.ndarray:
    # img: float32 [0,255]
    m = img.min(axis=2)
    if win <= 1:
        return m
    if minimum_filter is not None:
        return minimum_filter(m, size=win, mode='nearest')
    # numpy fallback: exact但較慢
    h, w = m.shape
    r = win // 2
    padded = np.pad(m, ((r, r), (r, r)), mode='edge')
    out = np.empty_like(m)
    for y in range(h):
        for x in range(w):
            out[y, x] = padded[y:y+win, x:x+win].min()
    return out


def estimate_atmospheric_light(H: np.ndarray, patch: int = 15, topk: int = 16) -> np.ndarray:
    H_ds = downsample2(H)
    dc = dark_channel(H_ds, win=patch)
    flat_idx = np.argpartition(dc.ravel(), -topk)[-topk:]
    ys, xs = np.unravel_index(flat_idx, dc.shape)
    samples = H_ds[ys, xs, :].astype(np.float32)
    A = samples.mean(axis=0)
    return A

# ---------------------- Feature Extraction --------------------- #
@dataclass
class Features:
    vec: np.ndarray
    names: List[str]


def fixed_point_reciprocal(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return 1.0 / (x + eps)


def _conv2_sep7_box(img: np.ndarray) -> np.ndarray:
    # separable 7x7 box blur
    k = np.ones((7,), dtype=np.float32) / 7.0
    pad_h = np.pad(img, ((0, 0), (3, 3)), mode='edge')
    tmp = np.zeros_like(img, dtype=np.float32)
    # horizontal
    for y in range(img.shape[0]):
        # 利用卷積式滑窗避免 Python 迴圈太慢（簡化版）
        row = pad_h[y]
        acc = np.convolve(row, k, mode='valid')
        tmp[y] = acc
    # vertical
    pad_v = np.pad(tmp, ((3, 3), (0, 0)), mode='edge')
    out = np.zeros_like(img, dtype=np.float32)
    for x in range(img.shape[1]):
        col = pad_v[:, x]
        acc = np.convolve(col, k, mode='valid')
        out[:, x] = acc
    return out


def extract_features(H_u8: np.ndarray, A: np.ndarray) -> Features:
    H = H_u8.astype(np.float32)

    # 1) Dark channel stats
    dc = H.min(axis=2) / 255.0
    dc_mean = float(dc.mean()); dc_std = float(dc.std())
    dc_p10, dc_p50, dc_p90 = [float(np.percentile(dc, p)) for p in (10, 50, 90)]
    dc_ratio_005 = float((dc < 0.05).mean()); dc_ratio_01 = float((dc < 0.10).mean())

    # 2) Edge density via Sobel (簡潔向量化)
    gray = (0.299*H[:, :, 0] + 0.587*H[:, :, 1] + 0.114*H[:, :, 2])
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    # 使用 scipy.signal.convolve2d 會更快，但為了少依賴，這裡用 numpy padding + einsum 寫法
    def conv2(img, k):
        r = k.shape[0]//2
        pad = np.pad(img, ((r, r), (r, r)), mode='edge')
        # 展成視窗區塊
        Hh, Ww = img.shape
        # 提取每個 (y,x) 的 3x3 patch
        s0 = pad.strides[0]; s1 = pad.strides[1]
        shape = (Hh, Ww, 3, 3)
        strides = (s0, s1, s0, s1)
        patches = np.lib.stride_tricks.as_strided(pad, shape=shape, strides=strides)
        return np.einsum('ij,xyij->xy', k, patches)
    gx = conv2(gray, kx); gy = conv2(gray, ky)
    mag = np.sqrt(gx*gx + gy*gy)
    thr = max(1.0, np.percentile(mag, 75))
    edge_density = float((mag > thr).mean())
    edge_mean = float(mag.mean()); edge_p90 = float(np.percentile(mag, 90))

    # 3) Saturation-like stats
    mx = H.max(axis=2); mn = H.min(axis=2)
    sat = (mx - mn) * fixed_point_reciprocal(mx)
    sat = np.clip(sat, 0, 1)
    s_mean = float(sat.mean()); s_std = float(sat.std()); s_p90 = float(np.percentile(sat, 90))

    # 4) A-normalized mean intensity K
    invA = fixed_point_reciprocal(A.astype(np.float32))
    Hn = H * invA[None, None, :]
    Hn = np.clip(Hn, 0, 4.0)
    K = Hn.mean(axis=2)
    K_mean = float(K.mean()); K_p90 = float(np.percentile(K, 90)); K_low_ratio = float((K < 0.3).mean())

    # 5) Low-frequency ratio (7x7 box blur)
    B = _conv2_sep7_box(gray)
    low_ratio = float((np.abs(gray - B) < 5.0).mean())

    # 6) A stats
    A0, A1, A2 = [float(A[i]/255.0) for i in range(3)]
    A_norm = float(np.linalg.norm(A)/255.0)

    vec = np.array([
        dc_mean, dc_std, dc_p10, dc_p50, dc_p90, dc_ratio_005, dc_ratio_01,
        edge_density, edge_mean, edge_p90,
        s_mean, s_std, s_p90,
        K_mean, K_p90, K_low_ratio,
        low_ratio,
        A0, A1, A2, A_norm
    ], dtype=np.float32)

    names = [
        'dc_mean','dc_std','dc_p10','dc_p50','dc_p90','dc_ratio_005','dc_ratio_01',
        'edge_density','edge_mean','edge_p90',
        's_mean','s_std','s_p90',
        'K_mean','K_p90','K_low_ratio',
        'low_ratio',
        'A0','A1','A2','A_norm'
    ]
    return Features(vec=vec, names=names)

# --------------------------- Dataset --------------------------- #

def scan_dataset_folders(root: str, exts=(".png",".jpg",".jpeg")) -> Tuple[List[str], np.ndarray]:
    paths: List[str] = []
    labels: List[float] = []
    for d in sorted(os.listdir(root)):
        try:
            psi = float(d)
        except Exception:
            continue
        dd = os.path.join(root, d)
        if not os.path.isdir(dd):
            continue
        for p in sorted(glob.glob(os.path.join(dd, "**", "*"), recursive=True)):
            if os.path.splitext(p)[1].lower() in exts:
                paths.append(p)
                labels.append(psi)
    if not paths:
        raise RuntimeError(f"No images found under {root}")
    return paths, np.array(labels, dtype=np.float32)


def scan_dataset_csv(csv_path: str, hazy_dir: str) -> Tuple[List[str], np.ndarray]:
    import csv
    paths: List[str] = []
    labels: List[float] = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Image') == 'AVERAGE':
                continue
            image_id = row['Image']
            best_psi = float(row['BestPsi'])
            p = os.path.join(hazy_dir, f"{image_id}_hazy.png")
            if os.path.exists(p):
                paths.append(p)
                labels.append(best_psi)
            else:
                # 忽略缺檔，或可在此列印警告
                pass
    if not paths:
        raise RuntimeError(f"No images found by CSV={csv_path} + hazy_dir={hazy_dir}")
    return paths, np.array(labels, dtype=np.float32)


def build_X(paths: List[str]) -> Tuple[np.ndarray, List[str]]:
    feats = []
    names: Optional[List[str]] = None
    for i, p in enumerate(paths):
        H = load_image_rgb(p)
        A = estimate_atmospheric_light(H, patch=15, topk=16)
        f = extract_features(H, A)
        feats.append(f.vec)
        if names is None:
            names = f.names
        if (i+1) % 50 == 0:
            print(f"[feat] {i+1}/{len(paths)} done")
    X = np.vstack(feats)
    return X, names or []

# ------------------------- Training/Eval ------------------------ #

def kfold_eval(X: np.ndarray, y: np.ndarray, k: int = 5) -> Dict[str, float]:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    res = {"Linear": [], "RF": []}
    for tr, va in kf.split(X):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)

        lr = LinearRegression()
        lr.fit(Xtr_s, ytr)
        p_lr = lr.predict(Xva_s)
        res["Linear"].append(mean_absolute_error(yva, p_lr))

        rf = RandomForestRegressor(n_estimators=600, min_samples_leaf=3, random_state=42, n_jobs=-1)
        rf.fit(Xtr, ytr)
        p_rf = rf.predict(Xva)
        res["RF"].append(mean_absolute_error(yva, p_rf))

    return {k: float(np.mean(v)) for k, v in res.items()}


def train_final_and_export(X: np.ndarray, y: np.ndarray, outdir: str, feature_names: List[str], psi_min: float, psi_max: float, q_frac_bits: int = 14):
    os.makedirs(outdir, exist_ok=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lr = LinearRegression()
    lr.fit(Xs, y)

    # Float export
    model = {
        "type": "linear_scaled",
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights": lr.coef_.tolist(),
        "bias": float(lr.intercept_),
        "q_format": {"int_bits": 2, "frac_bits": q_frac_bits},
        "psi_range": [psi_min, psi_max]
    }
    with open(os.path.join(outdir, "psi_linear_float.json"), 'w') as f:
        json.dump(model, f, indent=2)

    # Fixed-point export (Q format)
    FRAC_BITS = q_frac_bits
    SCALE = 1 << FRAC_BITS
    def q(val):
        return int(np.round(val * SCALE))

    q_weights = [q(w) for w in lr.coef_.tolist()]
    q_bias = q(lr.intercept_)
    q_mean = [q(m) for m in scaler.mean_.tolist()]
    q_invscale = [q(1.0/s) for s in scaler.scale_.tolist()]

    q_model = {
        "type": "linear_scaled_Q",
        "feature_names": feature_names,
        "mean_q": q_mean,
        "invscale_q": q_invscale,
        "weights_q": q_weights,
        "bias_q": q_bias,
        "frac_bits": FRAC_BITS,
        "psi_min": psi_min,
        "psi_max": psi_max
    }
    with open(os.path.join(outdir, "psi_linear_q.json"), 'w') as f:
        json.dump(q_model, f, indent=2)

    hdr = [
        "#pragma once",
        "// Auto-generated ψ linear model (scaled + Q format)",
        f"#define PSI_Q_FRAC {FRAC_BITS}",
        f"#define PSI_NFEAT {len(feature_names)}",
        f"#define PSI_MIN {psi_min} // clamp lower bound",
        f"#define PSI_MAX {psi_max} // clamp upper bound",
        "static const int32_t PSI_MEAN_Q[PSI_NFEAT] = {" + ", ".join(map(str, q_mean)) + "};",
        "static const int32_t PSI_INV_SCALE_Q[PSI_NFEAT] = {" + ", ".join(map(str, q_invscale)) + "};",
        "static const int32_t PSI_W_Q[PSI_NFEAT] = {" + ", ".join(map(str, q_weights)) + "};",
        f"static const int32_t PSI_B_Q = {q_bias};",
    ]
    with open(os.path.join(outdir, "psi_linear_q.h"), 'w') as f:
        f.write("\n".join(hdr) + "\n")

    print(f"[export] Wrote {outdir}/psi_linear_float.json, psi_linear_q.json, psi_linear_q.h")

# ------------------------------ CLI ---------------------------- #

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--root', help='Foldered dataset root with subfolders 0.5..1.7 (images inside)')
    src.add_argument('--csv', help='CSV file path with columns: Image, BestPsi')
    ap.add_argument('--hazy_dir', help='Required if --csv is used: directory containing *_hazy.png', default=None)
    ap.add_argument('--kfold', type=int, default=5)
    ap.add_argument('--out', default='models')
    ap.add_argument('--psi_min', type=float, default=0.5)
    ap.add_argument('--psi_max', type=float, default=1.7)
    args = ap.parse_args()

    if args.csv and not args.hazy_dir:
        raise SystemExit("--hazy_dir is required when --csv is provided")

    if args.root:
        print(f"[scan] scanning folders at {args.root}")
        paths, y = scan_dataset_folders(args.root)
    else:
        print(f"[scan] scanning CSV {args.csv} with hazy_dir={args.hazy_dir}")
        paths, y = scan_dataset_csv(args.csv, args.hazy_dir)

    # clamp y into range if需要
    y = np.clip(y, args.psi_min, args.psi_max)

    print(f"[scan] found {len(paths)} images; ψ levels (unique) = {len(np.unique(np.round(y,1)))}")

    print("[feat] extracting features...")
    X, names = build_X(paths)

    print("[eval] K-Fold evaluation...")
    scores = kfold_eval(X, y, k=args.kfold)
    for k, v in scores.items():
        print(f"  {k}: MAE={v:.4f}")

    print("[train] training final linear model and exporting...")
    train_final_and_export(X, y, args.out, names, psi_min=args.psi_min, psi_max=args.psi_max)

    print("[done] ✅")

if __name__ == '__main__':
    main()

# [eval] K-Fold evaluation...
#   Linear: MAE=0.1708
#   RF: MAE=0.1214
# [train] training final linear model and exporting...
# [export] Wrote models/psi_linear_float.json, psi_linear_q.json, psi_linear_q.h
# [done] ✅
