from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import time

# ===== 路徑設定（照你現在的相對路徑）=====
img_path = Path("../../dataset/fire/hazy")             # 輸入資料夾
res_path = Path("../../dataset/fire/result_water")     # 輸出資料夾
model_path = Path("../model/std_L_ch_based3K.pth")     # 模型檔
# ==========================================

# === 尺寸策略 ===
# 1) 原生尺寸推論（推薦）：INFER_SIZE = None，並用 PAD_TO_MULTIPLE 做對齊
# 2) 固定網路尺寸推論（若模型只能吃固定尺寸）：INFER_SIZE = (256, 256)
INFER_SIZE = None           # 改成 (256, 256) 可維持你原流程，但輸出會縮回原圖大小
PAD_TO_MULTIPLE = 32        # 多數編碼器下採樣步長的公倍數；不確定就用 32，關閉可設為 None

# 裝置：優先用 GPU
USE_CUDA = True
device = "cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu"
print(f"Using {device} for inference.")

# 準備輸出資料夾
res_path.mkdir(parents=True, exist_ok=True)

# 只吃影像副檔名
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 抓檔案（先非遞迴，沒有就遞迴）
files = sorted(p for p in img_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)
if not files:
    files = sorted(p for p in img_path.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS)

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# === 載入模型 ===
uwmodel = torch.load(model_path, map_location=device, weights_only=False)
uwmodel = uwmodel.to(device).eval()

time_list = []

def pad_to_multiple_both_sides(x, multiple=32):
    """
    x: [B,C,H,W] tensor
    回傳: padded_x, (left, right, top, bottom), (orig_h, orig_w)
    """
    b, c, h, w = x.shape
    if multiple is None:
        return x, (0,0,0,0), (h,w)

    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if top or bottom or left or right:
        x = F.pad(x, (left, right, top, bottom), mode="reflect")
    return x, (left, right, top, bottom), (h, w)

def unpad(x, pads, orig_hw):
    """
    x: [B,C,H,W], pads=(left,right,top,bottom), orig_hw=(orig_h, orig_w)
    以 pads 和原尺寸裁回
    """
    left, right, top, bottom = pads
    _, _, H, W = x.shape
    h0, w0 = orig_hw
    # 先去除 padding，再保底用原尺寸裁切，避免 off-by-one
    x = x[..., top:H-bottom if bottom>0 else H, left:W-right if right>0 else W]
    x = x[..., :h0, :w0]
    return x

with torch.no_grad():
    for idx, img in enumerate(files, 1):
        try:
            # 讀原圖（不 resize）
            read_img = Image.open(str(img)).convert("RGB")
            orig_w, orig_h = read_img.size

            # 準備推論輸入
            if INFER_SIZE is not None:
                # 固定網路尺寸（例如 256x256），後面會把輸出縮回原尺寸
                infer_img = read_img.resize(INFER_SIZE, Image.BICUBIC)
                img_tensor = to_tensor(infer_img).unsqueeze(0).to(device)  # [1,3,H,W]
                # 固定尺寸模式下通常不需要 padding，但留著一致介面
                img_tensor, pads, orig_hw = pad_to_multiple_both_sides(img_tensor, multiple=None)
            else:
                # 原生尺寸推論 + 對齊到 multiple（例如 32）
                img_tensor = to_tensor(read_img).unsqueeze(0).to(device)
                img_tensor, pads, orig_hw = pad_to_multiple_both_sides(img_tensor, multiple=PAD_TO_MULTIPLE)

            # 推論
            starttime = time.time()
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out_tensor = uwmodel(img_tensor)
                torch.cuda.synchronize()
            else:
                out_tensor = uwmodel(img_tensor)
            endtime = time.time()
            time_list.append(endtime - starttime)

            # 去 padding / 還原尺寸
            # 模型輸出保守處理為 float32 以轉 PIL
            out_tensor = out_tensor.float()
            out_tensor = torch.clamp(out_tensor, 0.0, 1.0)

            # 先去 padding
            if INFER_SIZE is None:
                out_tensor = unpad(out_tensor, pads, orig_hw)  # [1,3,orig_h,orig_w]
                out_img = to_pil(out_tensor.squeeze(0).cpu())
            else:
                # 固定尺寸模式：輸出先是 INFER_SIZE，再縮回原圖尺寸
                out_img_infer = to_pil(out_tensor.squeeze(0).cpu())
                out_img = out_img_infer.resize((orig_w, orig_h), Image.LANCZOS)

            # 保留子資料夾結構輸出
            try:
                rel = img.relative_to(img_path)
                (res_path / rel.parent).mkdir(parents=True, exist_ok=True)
                out_img.save(res_path / rel)
            except ValueError:
                out_img.save(res_path / img.name)

            if idx % 20 == 0 or idx == len(files):
                avg = sum(time_list) / len(time_list) if time_list else 0.0
                fps = (1.0 / avg) if avg > 0 else 0.0
                print(f"[{idx}/{len(files)}] avg_time={avg:.4f}s ~{fps:.2f} FPS -> {img.name}")

        except Exception as e:
            print(f"⚠️ 跳過 {img}: {e}")

if time_list:
    avg = sum(time_list) / len(time_list)
    fps = (1.0 / avg) if avg > 0 else 0.0
    print(f"✅ 完成：共 {len(time_list)} 張，Avg={avg:.4f}s/張，~{fps:.2f} FPS，輸出在：{res_path}")
else:
    print("⚠️ 沒有任何影像被處理，請檢查資料夾與副檔名。")
