# interactive_defog_v6_method1.py
import numpy as np
from scipy.ndimage import minimum_filter
from PIL import Image
import cv2
import os
from glob import glob


# ========== Configuration ==========
DATASET_NAME = "OHaze_lite"
INPUT_DIR = f"./dataset/{DATASET_NAME}/hazy"
OUTPUT_DIR = f"./dataset/{DATASET_NAME}/interactive_v6_output"

# Default parameters (scaled for trackbar)
DEFAULT_PSI_MIN = 52       # 0.52 * 100
DEFAULT_PSI_MAX = 138      # 1.38 * 100
DEFAULT_T0 = 25            # 0.25 * 100
DEFAULT_BUFFER_SIZE = 16
DEFAULT_WINDOW_SIZE = 8
DEFAULT_STRENGTH = 100     # 1.0 * 100 (除霧強度)
DEFAULT_STRENGTH_CURVE = 300  # 3.0 * 100 (強度曲線) - 更陡峭的預設值
DEFAULT_PSI_THRESHOLD = 50    # 0.50 * 100 (PSI閾值)

# Trackbar ranges
PSI_MIN_RANGE = 100        # Max 1.0
PSI_MAX_RANGE = 200        # Max 2.0
T0_RANGE = 100             # Max 1.0
BUFFER_SIZE_RANGE = 64     # Max 64
WINDOW_SIZE_RANGE = 32     # Max 32
STRENGTH_RANGE = 300       # Max 3.0 (除霧強度)
STRENGTH_CURVE_RANGE = 1000 # Max 10.0 (強度曲線) - 允許更陡峭
PSI_THRESHOLD_RANGE = 100   # Max 1.0 (PSI閾值)

# Window size for preview
PREVIEW_MAX_HEIGHT = 600


def compute_atmospheric_light(image, window_size=8):
    """
    計算大氣光 A
    從暗通道中選擇最亮的像素作為 A
    """
    dark_channel = np.min(image, axis=2)
    dark_min = minimum_filter(dark_channel, size=window_size)
    idx = np.argmax(dark_min)
    y, x = np.unravel_index(idx, dark_min.shape)
    A = image[y, x, :].copy()
    return A


def compute_psi_map_method1(H, A, psi_min=0.52, psi_max=1.38, buffer_size=8, epsilon=1e-6):
    """
    方法一：大氣光比較法 (Atmospheric Light Comparison)
    
    原理：
    - 霧氣會使像素值趨近大氣光 A
    - 計算每個像素與 A 的相對距離: |pixel - A| / A
    - 距離小 = 霧濃 = PSI 高
    - 距離大 = 無霧 = PSI 低
    
    參數:
        H: 輸入圖像 (float32, shape: height x width x 3)
        A: 大氣光向量 (shape: 3,)
        psi_min: PSI 最小值
        psi_max: PSI 最大值
        buffer_size: line buffer 大小
        epsilon: 防除零
    
    返回:
        psi_map: PSI 分布圖 (shape: height x width)
    """
    height, width = H.shape[:2]
    psi_range = psi_max - psi_min
    
    # 計算霧濃度: 0=接近A(霧濃), 100=遠離A(無霧)
    # 使用 min 通道計算霧濃度 (暗通道原理)
    diff = np.abs(H - A)
    relative_diff = diff / (A + epsilon)
    fog_density = np.min(relative_diff, axis=2) * 100
    # 輕微的 power 變換
    fog_density = np.power(fog_density / 100.0, 1.05) * 100
    fog_density = np.clip(fog_density, 0, 100)
    
    # 計算 raw PSI: 霧濃 → PSI 高
    raw_psi = psi_max - (fog_density / 100.0) * psi_range
    
    # Line buffer 平滑
    psi_map = np.zeros((height, width), dtype=np.float32)
    
    for i in range(height):
        line_buffer = np.zeros(buffer_size, dtype=np.float32)
        buffer_sum = 0.0
        buffer_count = 0
        head = 0
        
        for j in range(width):
            current_psi = raw_psi[i, j]
            
            if buffer_count >= buffer_size:
                buffer_sum -= line_buffer[head]
            else:
                buffer_count += 1
            
            line_buffer[head] = current_psi
            buffer_sum += current_psi
            head = (head + 1) % buffer_size
            
            psi_map[i, j] = np.clip(buffer_sum / buffer_count, psi_min, psi_max)
    
    return psi_map


def defog_v6_method1(hazy_image, A, psi_min=0.52, psi_max=1.38, t0=0.25, 
                     buffer_size=8, defogging_strength=1.0, strength_curve=3.0, 
                     psi_threshold=0.5, epsilon=1e-6):
    """
    V6 Method 1: 大氣光比較法除霧（自適應強度版 - 陡峭曲線）
    
    參數:
        hazy_image: 輸入圖像（RGB，float32）
        A: 大氣光向量（3,）
        psi_min: PSI 最小值 (無霧時)
        psi_max: PSI 最大值 (濃霧時)
        t0: 傳輸圖的下界（預設 0.25）
        buffer_size: line buffer 大小
        defogging_strength: 最大除霧強度 (預設 1.0)
            - 1.0 = 標準除霧
            - >1.0 = 更強除霧 (如 2.0, 3.0) - 僅應用在霧濃區域
        strength_curve: 強度曲線指數 (預設 3.0)
            - 1.0 = 線性分布
            - 3.0 = 陡峭曲線（推薦）
            - 5.0~10.0 = 極陡峭（幾乎二元化）
        psi_threshold: PSI 閾值 (預設 0.5)
            - PSI < threshold 的區域會被抑制強度
            - 有助於保護無霧/淡霧區域
        epsilon: 防止除零的小常數
    
    返回:
        D: 去霧後的圖像（float32, 0-255）
        psi_map: PSI 分布圖
        t: 傳輸圖
        strength_map: 自適應強度分布圖
    """
    H = hazy_image.astype(np.float32)
    
    # ========== 計算歸一化圖像 ==========
    H_norm = H / (A + epsilon)
    K = np.mean(H_norm, axis=2)
    min_norm = np.min(H_norm, axis=2)
    
    # ========== 計算 PSI Map (方法一：大氣光比較法) ==========
    psi_map = compute_psi_map_method1(H, A, psi_min, psi_max, buffer_size, epsilon)
    
    # ========== 計算傳輸圖 t (使用動態 PSI) ==========
    temp = 3 * K + 3 * min_norm
    t = (temp - psi_map * 3 * K * min_norm) / (temp + epsilon)
    
    # 限制傳輸圖的下界
    t = np.clip(t, t0, 1)
    
    # ========== 自適應除霧強度調整（陡峭曲線版） ==========
    # 1. 將 PSI 正規化到 0-1 範圍（0=無霧, 1=濃霧）
    psi_normalized = (psi_map - psi_min) / (psi_max - psi_min + epsilon)
    psi_normalized = np.clip(psi_normalized, 0, 1)
    
    # 2. 應用閾值遮罩（保護低 PSI 區域）
    # PSI < threshold 的區域會被額外抑制
    threshold_mask = np.where(psi_normalized < psi_threshold, 
                             psi_normalized / (psi_threshold + epsilon),  # 線性衰減
                             1.0)  # 高於閾值的保持原值
    psi_normalized_masked = psi_normalized * threshold_mask
    
    # 3. 應用超陡峭曲線
    # 使用 power 函數：curve 越大，曲線越陡峭
    psi_weighted = np.power(psi_normalized_masked, strength_curve)
    
    # 4. 計算自適應強度
    # 無霧區域 (psi=0) → strength = 1.0 (不處理)
    # 濃霧區域 (psi=1) → strength = defogging_strength (完整強度)
    adaptive_strength = 1.0 + (defogging_strength - 1.0) * psi_weighted
    
    # 5. 應用自適應強度到傳輸圖
    t_adjusted = np.power(t, adaptive_strength)
    t_adjusted = np.clip(t_adjusted, t0, 1)
    
    # ========== 利用傳輸圖恢復無霧圖像 ==========
    t_expanded = t_adjusted[:, :, np.newaxis]
    D = (H - A) / t_expanded + A
    D = np.clip(D, 0, 255)
    
    return D, psi_map, t_adjusted, adaptive_strength


def create_heatmap(data, colormap=cv2.COLORMAP_JET):
    """Convert data to color heatmap"""
    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)
    data_uint8 = (np.clip(data_normalized, 0, 1) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(data_uint8, colormap)
    return heatmap


def add_border(image, color=(200, 200, 200), thickness=2):
    """Add border to image"""
    return cv2.copyMakeBorder(image, thickness, thickness, thickness, thickness,
                             cv2.BORDER_CONSTANT, value=color)


def add_label(image, text, bg_color=(50, 50, 50), text_color=(255, 255, 255)):
    """Add label on top of image"""
    h, w = image.shape[:2]
    label_height = 50
    label = np.full((label_height, w, 3), bg_color, dtype=np.uint8)
    
    # Add text
    font_scale = 0.8
    thickness = 2
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (label_height + text_size[1]) // 2
    
    cv2.putText(label, text, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
    
    return np.vstack([label, image])


def create_stats_panel(stats_lines, width=450, bg_color=(40, 40, 40), text_color=(255, 255, 255)):
    """Create statistics panel"""
    line_height = 28
    padding = 20
    height = len(stats_lines) * line_height + padding * 2
    
    panel = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    y = padding + 20
    for line in stats_lines:
        if line.strip() == "":
            y += line_height // 2
            continue
            
        # Bold for section headers
        if line.endswith(":") and not line.startswith("  "):
            font_scale = 0.65
            thickness = 2
            color = (100, 200, 255)
        else:
            font_scale = 0.55
            thickness = 1
            color = text_color
        
        cv2.putText(panel, line, (padding, y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height
    
    return panel


class InteractiveDehazingV6:
    """Interactive Dehazing Application - V6 Method 1"""
    
    def __init__(self, image_path):
        # Load image
        self.image_path = image_path
        self.image_name = os.path.basename(image_path)
        
        img = Image.open(image_path).convert('RGB')
        self.image_original = np.array(img).astype(np.float32)
        
        # Resize for preview keeping aspect ratio
        h, w = self.image_original.shape[:2]
        if h > PREVIEW_MAX_HEIGHT:
            scale = PREVIEW_MAX_HEIGHT / h
            new_h = PREVIEW_MAX_HEIGHT
            new_w = int(w * scale)
            self.image = cv2.resize(self.image_original, (new_w, new_h), 
                                   interpolation=cv2.INTER_AREA).astype(np.float32)
            print(f"Resized for preview: {w}x{h} -> {new_w}x{new_h}")
        else:
            self.image = self.image_original.copy()
        
        # Current parameters
        self.psi_min = DEFAULT_PSI_MIN
        self.psi_max = DEFAULT_PSI_MAX
        self.t0 = DEFAULT_T0
        self.buffer_size = DEFAULT_BUFFER_SIZE
        self.window_size = DEFAULT_WINDOW_SIZE
        self.defogging_strength = DEFAULT_STRENGTH
        self.strength_curve = DEFAULT_STRENGTH_CURVE
        self.psi_threshold = DEFAULT_PSI_THRESHOLD
        
        # Compute atmospheric light
        self.update_atmospheric_light()
        
        # Create window and trackbars
        self.window_name = "Interactive Dehazing V6 - Method 1 (Atmospheric Light Comparison)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1800, 1000)
        
        self.create_trackbars()
        
        # Initial update
        self.needs_A_update = False
        self.update_display()
    
    def create_trackbars(self):
        """Create trackbars"""
        cv2.createTrackbar('PSI Min (x0.01)', self.window_name, 
                          self.psi_min, PSI_MIN_RANGE, self.on_trackbar_change)
        
        cv2.createTrackbar('PSI Max (x0.01)', self.window_name, 
                          self.psi_max, PSI_MAX_RANGE, self.on_trackbar_change)
        
        cv2.createTrackbar('PSI Threshold (x0.01)', self.window_name, 
                          self.psi_threshold, PSI_THRESHOLD_RANGE, self.on_trackbar_change)
        
        cv2.createTrackbar('t0 (x0.01)', self.window_name, 
                          self.t0, T0_RANGE, self.on_trackbar_change)
        
        cv2.createTrackbar('Max Strength (x0.01)', self.window_name, 
                          self.defogging_strength, STRENGTH_RANGE, self.on_trackbar_change)
        
        cv2.createTrackbar('Strength Curve (x0.01)', self.window_name, 
                          self.strength_curve, STRENGTH_CURVE_RANGE, self.on_trackbar_change)
        
        cv2.createTrackbar('Buffer Size', self.window_name, 
                          self.buffer_size, BUFFER_SIZE_RANGE, self.on_trackbar_change)
        
        cv2.createTrackbar('Window Size', self.window_name, 
                          self.window_size, WINDOW_SIZE_RANGE, self.on_window_size_change)
    
    def on_trackbar_change(self, val):
        """Callback for trackbar change"""
        self.update_display()
    
    def on_window_size_change(self, val):
        """Callback for window size change (requires A recomputation)"""
        self.needs_A_update = True
        self.update_display()
    
    def update_atmospheric_light(self):
        """Update atmospheric light A"""
        self.A = compute_atmospheric_light(self.image, window_size=self.window_size)
    
    def update_display(self):
        """Update display"""
        # Get current parameters from trackbars
        self.psi_min = cv2.getTrackbarPos('PSI Min (x0.01)', self.window_name)
        self.psi_max = cv2.getTrackbarPos('PSI Max (x0.01)', self.window_name)
        self.psi_threshold = cv2.getTrackbarPos('PSI Threshold (x0.01)', self.window_name)
        self.t0 = cv2.getTrackbarPos('t0 (x0.01)', self.window_name)
        self.defogging_strength = cv2.getTrackbarPos('Max Strength (x0.01)', self.window_name)
        self.strength_curve = cv2.getTrackbarPos('Strength Curve (x0.01)', self.window_name)
        self.buffer_size = cv2.getTrackbarPos('Buffer Size', self.window_name)
        self.window_size = cv2.getTrackbarPos('Window Size', self.window_name)
        
        # Ensure minimum values and logical constraints
        if self.buffer_size < 4:
            self.buffer_size = 4
            cv2.setTrackbarPos('Buffer Size', self.window_name, 4)
        
        if self.window_size < 3:
            self.window_size = 3
            cv2.setTrackbarPos('Window Size', self.window_name, 3)
        
        if self.psi_min >= self.psi_max:
            self.psi_max = self.psi_min + 10
            cv2.setTrackbarPos('PSI Max (x0.01)', self.window_name, self.psi_max)
        
        if self.t0 < 1:
            self.t0 = 1
            cv2.setTrackbarPos('t0 (x0.01)', self.window_name, 1)
        
        if self.defogging_strength < 10:
            self.defogging_strength = 10
            cv2.setTrackbarPos('Max Strength (x0.01)', self.window_name, 10)
        
        if self.strength_curve < 10:
            self.strength_curve = 10
            cv2.setTrackbarPos('Strength Curve (x0.01)', self.window_name, 10)
        
        if self.psi_threshold < 0:
            self.psi_threshold = 0
            cv2.setTrackbarPos('PSI Threshold (x0.01)', self.window_name, 0)
        
        # Update atmospheric light if window size changed
        if self.needs_A_update:
            self.update_atmospheric_light()
            self.needs_A_update = False
        
        # Convert trackbar values to actual parameters
        psi_min_val = self.psi_min / 100.0
        psi_max_val = self.psi_max / 100.0
        psi_threshold_val = self.psi_threshold / 100.0
        t0_val = self.t0 / 100.0
        strength_val = self.defogging_strength / 100.0
        curve_val = self.strength_curve / 100.0
        
        # Perform dehazing using V6 Method 1
        dehazed, psi_map, transmission, strength_map = defog_v6_method1(
            self.image,
            self.A,
            psi_min=psi_min_val,
            psi_max=psi_max_val,
            t0=t0_val,
            buffer_size=self.buffer_size,
            defogging_strength=strength_val,
            strength_curve=curve_val,
            psi_threshold=psi_threshold_val
        )
        
        self.current_result = dehazed
        self.current_psi_map = psi_map
        self.current_transmission = transmission
        self.current_strength_map = strength_map
        
        # Create display
        self.create_display(psi_min_val, psi_max_val, psi_threshold_val, t0_val, strength_val, curve_val)
    
    def create_display(self, psi_min_val, psi_max_val, psi_threshold_val, t0_val, strength_val, curve_val):
        """Create display layout"""
        # Convert to uint8 for display
        img_display = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        result_display = cv2.cvtColor(self.current_result.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Create heatmaps
        psi_heatmap = create_heatmap(self.current_psi_map, cv2.COLORMAP_JET)
        strength_heatmap = create_heatmap(self.current_strength_map, cv2.COLORMAP_HOT)
        
        # Add labels and borders
        img1 = add_label(add_border(img_display), "Original Hazy Image")
        img2 = add_label(add_border(psi_heatmap), f"PSI Map (Fog Density)")
        img3 = add_label(add_border(strength_heatmap), "Adaptive Strength Map (STEEP)")
        img4 = add_label(add_border(result_display), "Dehazed Result (Adaptive)")
        
        # Create 2x2 grid
        top_row = np.hstack([img1, img2])
        bottom_row = np.hstack([img3, img4])
        image_grid = np.vstack([top_row, bottom_row])
        
        # Create statistics panel
        stats = [
            f"Image: {self.image_name}",
            "",
            "Steep Curve Parameters:",
            f"  PSI Min: {psi_min_val:.2f}",
            f"  PSI Max: {psi_max_val:.2f}",
            f"  PSI Threshold: {psi_threshold_val:.2f}",
            f"    (below this: suppressed)",
            f"  t0: {t0_val:.2f}",
            f"  Max Strength: {strength_val:.2f}",
            f"  Curve Steepness: {curve_val:.2f}",
            f"    (higher = steeper)",
            f"  Buffer: {self.buffer_size}",
            f"  Window: {self.window_size}",
            "",
            "Curve Effect:",
            f"  PSI < {psi_threshold_val:.2f}:",
            "     Linearly suppressed",
            f"  PSI >= {psi_threshold_val:.2f}:",
            f"     Power curve ^{curve_val:.1f}",
            "",
            "Strength Distribution:",
            f"  Min: {np.min(self.current_strength_map):.3f}",
            f"  Avg: {np.mean(self.current_strength_map):.3f}",
            f"  Max: {np.max(self.current_strength_map):.3f}",
            "",
            "PSI Statistics:",
            f"  Min: {np.min(self.current_psi_map):.3f}",
            f"  Avg: {np.mean(self.current_psi_map):.3f}",
            f"  Max: {np.max(self.current_psi_map):.3f}",
            "",
            "Atmospheric Light:",
            f"  R: {self.A[0]:.1f}",
            f"  G: {self.A[1]:.1f}",
            f"  B: {self.A[2]:.1f}",
            "",
            "Recommended for Heavy Fog:",
            "  Max Strength: 2.5~3.0",
            "  Curve: 4.0~8.0 (steep!)",
            "  PSI Threshold: 0.5~0.7",
            "  t0: 0.10~0.15",
            "",
            "Curve Steepness Guide:",
            "  1.0 - Linear (gentle)",
            "  3.0 - Cubic (moderate)",
            "  5.0 - Very steep",
            "  8.0-10.0 - Near binary",
            "",
            "Keys: S=Save, R=Reset, Q=Quit",
        ]
        
        stats_panel = create_stats_panel(stats, width=500)
        
        # Resize stats panel to match image grid height
        stats_h = image_grid.shape[0]
        stats_w = stats_panel.shape[1]
        stats_panel_resized = cv2.resize(stats_panel, (stats_w, stats_h), 
                                        interpolation=cv2.INTER_LINEAR)
        
        # Combine image grid and stats panel
        display = np.hstack([image_grid, stats_panel_resized])
        
        cv2.imshow(self.window_name, display)
    
    def reset_parameters(self):
        """Reset all parameters to default"""
        cv2.setTrackbarPos('PSI Min (x0.01)', self.window_name, DEFAULT_PSI_MIN)
        cv2.setTrackbarPos('PSI Max (x0.01)', self.window_name, DEFAULT_PSI_MAX)
        cv2.setTrackbarPos('PSI Threshold (x0.01)', self.window_name, DEFAULT_PSI_THRESHOLD)
        cv2.setTrackbarPos('t0 (x0.01)', self.window_name, DEFAULT_T0)
        cv2.setTrackbarPos('Max Strength (x0.01)', self.window_name, DEFAULT_STRENGTH)
        cv2.setTrackbarPos('Strength Curve (x0.01)', self.window_name, DEFAULT_STRENGTH_CURVE)
        cv2.setTrackbarPos('Buffer Size', self.window_name, DEFAULT_BUFFER_SIZE)
        cv2.setTrackbarPos('Window Size', self.window_name, DEFAULT_WINDOW_SIZE)
        self.needs_A_update = True
    
    def save_result(self):
        """Save current result"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_name = os.path.splitext(self.image_name)[0]
        
        # Save full resolution result if image was resized
        if self.image.shape != self.image_original.shape:
            print("Computing full resolution result...")
            A_full = compute_atmospheric_light(self.image_original, window_size=self.window_size)
            dehazed_full, psi_map_full, transmission_full, strength_map_full = defog_v6_method1(
                self.image_original,
                A_full,
                psi_min=self.psi_min / 100.0,
                psi_max=self.psi_max / 100.0,
                t0=self.t0 / 100.0,
                buffer_size=self.buffer_size,
                defogging_strength=self.defogging_strength / 100.0,
                strength_curve=self.strength_curve / 100.0,
                psi_threshold=self.psi_threshold / 100.0
            )
            result_to_save = dehazed_full
            psi_to_save = psi_map_full
            trans_to_save = transmission_full
            strength_to_save = strength_map_full
        else:
            result_to_save = self.current_result
            psi_to_save = self.current_psi_map
            trans_to_save = self.current_transmission
            strength_to_save = self.current_strength_map
        
        # Save dehazed image
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_v6_steep.png")
        result_img = Image.fromarray(result_to_save.astype(np.uint8))
        result_img.save(output_path)
        
        # Save PSI heatmap
        psi_heatmap_path = os.path.join(OUTPUT_DIR, f"{base_name}_psi_heatmap.png")
        psi_heatmap_full = create_heatmap(psi_to_save, cv2.COLORMAP_JET)
        cv2.imwrite(psi_heatmap_path, psi_heatmap_full)
        
        # Save strength heatmap
        strength_heatmap_path = os.path.join(OUTPUT_DIR, f"{base_name}_strength_heatmap.png")
        strength_heatmap_full = create_heatmap(strength_to_save, cv2.COLORMAP_HOT)
        cv2.imwrite(strength_heatmap_path, strength_heatmap_full)
        
        # Save parameters
        params_path = os.path.join(OUTPUT_DIR, f"{base_name}_parameters.txt")
        with open(params_path, 'w', encoding='utf-8') as f:
            f.write(f"V6 Method 1 - Steep Curve Adaptive Defogging\n")
            f.write(f"={'='*50}\n")
            f.write(f"Image: {self.image_name}\n")
            f.write(f"PSI Min: {self.psi_min / 100.0:.3f}\n")
            f.write(f"PSI Max: {self.psi_max / 100.0:.3f}\n")
            f.write(f"PSI Threshold: {self.psi_threshold / 100.0:.3f}\n")
            f.write(f"t0 (transmission lower bound): {self.t0 / 100.0:.3f}\n")
            f.write(f"Max Defogging Strength: {self.defogging_strength / 100.0:.3f}\n")
            f.write(f"Strength Curve (steepness): {self.strength_curve / 100.0:.3f}\n")
            f.write(f"Line Buffer Size: {self.buffer_size}\n")
            f.write(f"Window Size (for A): {self.window_size}\n")
            f.write(f"Atmospheric Light: R={self.A[0]:.2f}, G={self.A[1]:.2f}, B={self.A[2]:.2f}\n")
            f.write(f"\n")
            f.write(f"PSI Map Statistics:\n")
            f.write(f"  Average: {np.mean(psi_to_save):.6f}\n")
            f.write(f"  Min: {np.min(psi_to_save):.6f}\n")
            f.write(f"  Max: {np.max(psi_to_save):.6f}\n")
            f.write(f"  Std: {np.std(psi_to_save):.6f}\n")
            f.write(f"\n")
            f.write(f"Adaptive Strength Map Statistics:\n")
            f.write(f"  Average: {np.mean(strength_to_save):.6f}\n")
            f.write(f"  Min: {np.min(strength_to_save):.6f}\n")
            f.write(f"  Max: {np.max(strength_to_save):.6f}\n")
        
        print(f"✅ Saved: {output_path}")
        print(f"✅ Saved: {psi_heatmap_path}")
        print(f"✅ Saved: {strength_heatmap_path}")
        print(f"✅ Saved: {params_path}")
    
    def run(self):
        """Run the application"""
        print("\n" + "="*70)
        print("Interactive Dehazing V6 - STEEP CURVE Adaptive Method")
        print("="*70)
        print("🔥 Key Innovation: ULTRA STEEP curves for precise fog targeting")
        print("\nAlgorithm:")
        print("  1. Calculate PSI (fog density)")
        print("  2. Apply PSI Threshold (suppress low fog areas)")
        print("  3. Apply STEEP power curve (psi^curve)")
        print("  4. Map to adaptive strength: 1.0 (clear) → max_strength (foggy)")
        print("  5. Apply: t_adjusted = t^adaptive_strength")
        print("\nParameter Ranges:")
        print(f"  PSI Min: 0.00 - {PSI_MIN_RANGE/100.0:.2f}")
        print(f"  PSI Max: 0.00 - {PSI_MAX_RANGE/100.0:.2f}")
        print(f"  PSI Threshold: 0.00 - {PSI_THRESHOLD_RANGE/100.0:.2f}")
        print(f"  t0: 0.00 - {T0_RANGE/100.0:.2f}")
        print(f"  Max Strength: 0.10 - {STRENGTH_RANGE/100.0:.2f}")
        print(f"  Curve Steepness: 0.10 - {STRENGTH_CURVE_RANGE/100.0:.2f} ⚡")
        print(f"  Buffer Size: 4 - {BUFFER_SIZE_RANGE}")
        print(f"  Window Size: 3 - {WINDOW_SIZE_RANGE}")
        print("\n💡 RECOMMENDED Settings for Heavy Fog:")
        print("  ⚡ Max Strength: 2.5 ~ 3.0")
        print("  ⚡ Curve Steepness: 5.0 ~ 8.0 (STEEP!)")
        print("  ⚡ PSI Threshold: 0.5 ~ 0.7 (protect clear areas)")
        print("  ⚡ t0: 0.10 ~ 0.15")
        print("\n💡 Understanding Curve Steepness:")
        print("  1.0 - Linear (even distribution)")
        print("  3.0 - Cubic (moderate focus)")
        print("  5.0 - Very steep (strong focus)")
        print("  8.0 - Extreme steep (near binary)")
        print("  10.0 - Nearly binary (on/off)")
        print("\n💡 PSI Threshold Effect:")
        print("  - PSI < threshold: linearly suppressed strength")
        print("  - PSI ≥ threshold: full curve applied")
        print("  - Higher threshold = more protection for clear areas")
        print("\n🎯 What to Watch:")
        print("  - Bottom-left: Adaptive Strength Map")
        print("    • Red/bright = high strength (foggy areas)")
        print("    • Blue/dark = low strength (clear areas)")
        print("  - Adjust curve until strength map shows clear separation")
        print("\nControls:")
        print("  - Adjust trackbars to see real-time effects")
        print("  - Press 'S' to save result")
        print("  - Press 'R' to reset to defaults")
        print("  - Press 'Q' or ESC to quit")
        print("="*70 + "\n")
        
        while True:
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('s'):  # Save
                self.save_result()
            elif key == ord('r'):  # Reset
                self.reset_parameters()
        
        cv2.destroyAllWindows()


def main():
    """Main function"""
    import sys
    
    # Check if image path provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use first image in dataset
        image_files = sorted(glob(os.path.join(INPUT_DIR, "*.png")))
        if not image_files:
            print(f"❌ No PNG files found in {INPUT_DIR}")
            print(f"Usage: python interactive_defog_v6_method1.py [image_path]")
            return
        image_path = image_files[0]
        print(f"Using first image: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Create and run app
    app = InteractiveDehazingV6(image_path)
    app.run()


if __name__ == "__main__":
    main()