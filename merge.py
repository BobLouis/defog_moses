from pathlib import Path
from PIL import Image
import numpy as np

# --- 路徑設定 ---
base_dir  = Path("dataset/fire")
hazy_p    = base_dir / "hazy"  / "frame_000223.jpg"
water_p   = base_dir / "result_water" / "frame_000223.jpg"
defog_p   = base_dir / "result_test" / "frame_test.jpg"
out_dir   = base_dir / "repair"
mask_dir  = out_dir / "mask"
out_dir.mkdir(parents=True, exist_ok=True)
mask_dir.mkdir(parents=True, exist_ok=True)

THRESHOLD = 220  # 嚴格依照需求：pixel > 220 才算亮部

# --- 檔案存在性檢查 ---
for p in [hazy_p, water_p, defog_p]:
    if not p.exists():
        raise FileNotFoundError(f"找不到檔案：{p}")

# --- 讀圖（以 hazy 尺寸為基準對齊） ---
hazy_img  = Image.open(hazy_p).convert("RGB")
water_img = Image.open(water_p).convert("RGB").resize(hazy_img.size, Image.BICUBIC)
defog_img = Image.open(defog_p).convert("RGB").resize(hazy_img.size, Image.BICUBIC)

# --- 生成 mask：water 的灰階 > 220 ---
water_gray = water_img.convert("L")  # 8-bit, 0~255
g = np.array(water_gray, dtype=np.uint8)
mask = (g > THRESHOLD)  # shape [H, W], bool

# --- 套用規則：mask 為 True 的位置，用 hazy；其他地方保持 defog ---
hazy_np  = np.array(hazy_img,  dtype=np.uint8)
defog_np = np.array(defog_img, dtype=np.uint8)
out_np   = np.where(mask[..., None], hazy_np, defog_np)  # broadcast 到 3 通道

# --- 輸出結果與 mask ---
out_img  = Image.fromarray(out_np, mode="RGB")
out_img.save(out_dir / "test.png")  # 用 PNG 避免 jpg 壓縮影響

mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
mask_img.save(mask_dir / "test_mask.png")

print("完成：")
print(f"  修補圖：{out_dir / 'test.png'}")
print(f"  Mask：{mask_dir / 'test_mask.png'}")
