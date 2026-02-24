# fit_linear_psi_from_dark_ratio.py
import os, glob, argparse
import numpy as np
import pandas as pd
from PIL import Image

def load_rgb(p): return np.array(Image.open(p).convert('RGB'))

def compute_haze_index(H):
    # H: uint8 RGB
    H = H.astype(np.uint32)
    minRGB = np.minimum(np.minimum(H[:,:,0], H[:,:,1]), H[:,:,2]).astype(np.uint32)
    sumDark = int(minRGB.sum())
    sumI = int(H[:,:,0].sum() + H[:,:,1].sum() + H[:,:,2].sum())
    # 3*sumDark / sumI  （避免每像素除法）
    return (3.0 * sumDark) / (sumI + 1e-9)

def find_hazy(hazy_dir, base):
    # 兼容 001.png / 001_*.png
    cands = []
    for ptn in [f"{base}.png", f"{base}_*.png", f"{base}*.png"]:
        cands += glob.glob(os.path.join(hazy_dir, ptn))
    cands = [p for p in cands if "clear" not in p.lower() and "result" not in p.lower()]
    return sorted(cands)[0] if cands else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="包含 Image,BestPsi 的彙總CSV（你貼的那份）")
    ap.add_argument("--hazy_dir", required=True, help="./dataset/SOTS_in/hazy")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    assert "Image" in df.columns and "BestPsi" in df.columns

    X, y, used = [], [], []
    for _, r in df.iterrows():
        base = f"{int(r['Image']):03d}" if str(r["Image"]).isdigit() else str(r["Image"])
        p = find_hazy(args.hazy_dir, base)
        if not p: 
            print(f"skip {base}: hazy not found"); 
            continue
        H = load_rgb(p)
        h = compute_haze_index(H)
        X.append(h); y.append(float(r["BestPsi"])); used.append(base)

    X = np.array(X); y = np.array(y)
    # 線性回歸閉式解
    a = np.cov(X, y, bias=True)[0,1] / (np.var(X) + 1e-12)
    b = float(y.mean() - a * X.mean())

    # 評估 & 換算為「幾格(0.02)」
    pred = np.clip(a*X + b, 1.0, 2.0)
    mae = np.mean(np.abs(pred - y))
    print(f"a={a:.6f}, b={b:.6f}")
    print(f"MAE(psi)={mae:.4f}  ≈ {mae/0.02:.2f} 格距")

    # 建議固化為 Q16.16
    AQ = int(round(a * (1<<16)))
    BQ = int(round(b * (1<<16)))
    print(f"a_Q16.16={AQ}, b_Q16.16={BQ}")

if __name__ == "__main__":
    main()
