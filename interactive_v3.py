# interactive_defog_v6_method1.py
import numpy as np
from scipy.ndimage import minimum_filter
from PIL import Image
import cv2
import os
from glob import glob
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2lab, deltaE_ciede2000


# ========== Configuration ==========
DATASET_NAME = "OHaze"
INPUT_DIR = f"./dataset/{DATASET_NAME}/hazy"
OUTPUT_DIR = f"./dataset/{DATASET_NAME}/interactive_v7_output"
CLEAR_DIR = f"./dataset/{DATASET_NAME}/clear"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Default parameters (scaled for trackbar)
DEFAULT_PSI_MIN = 52       # 0.52 * 100
DEFAULT_PSI_MAX = 138      # 1.38 * 100
DEFAULT_T0 = 25            # 0.25 * 100
DEFAULT_BUFFER_SIZE = 4
DEFAULT_WINDOW_SIZE = 8
DEFAULT_STRENGTH = 200    # 1.0 * 100
DEFAULT_STRENGTH_CURVE = 400  # 3.0 * 100
DEFAULT_PSI_THRESHOLD = 50    # 0.50 * 100

# Trackbar ranges
PSI_MIN_RANGE = 100
PSI_MAX_RANGE = 200
T0_RANGE = 100
BUFFER_SIZE_RANGE = 64
WINDOW_SIZE_RANGE = 32
STRENGTH_RANGE = 300
STRENGTH_CURVE_RANGE = 1000
PSI_THRESHOLD_RANGE = 100

# Window size for preview
PREVIEW_MAX_HEIGHT = 600

# Modern UI Color Palette
UI_COLORS = {
    'bg_dark': (25, 25, 30),
    'bg_medium': (40, 42, 48),
    'bg_light': (55, 58, 64),
    'accent_blue': (255, 180, 80),
    'accent_green': (120, 220, 120),
    'accent_orange': (80, 140, 255),
    'text_primary': (255, 255, 255),
    'text_secondary': (180, 185, 200),
    'text_dim': (120, 125, 140),
    'border': (70, 75, 85),
    'highlight': (255, 200, 100),
}


def list_image_files(directory, extensions=IMAGE_EXTENSIONS):
    if not os.path.isdir(directory):
        return []
    files = []
    for path in sorted(glob(os.path.join(directory, "*"))):
        if os.path.splitext(path)[1].lower() in extensions:
            files.append(os.path.abspath(path))
    return files


def normalize_filename_key(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    lowered = base.lower()
    suffixes = [
        "_hazy", "_clear", "_gt", "_ref", "_input", "_noisy",
        "-hazy", "-clear", "-gt", "_h", "_c"
    ]
    for suffix in suffixes:
        if lowered.endswith(suffix):
            base = base[: -len(suffix)]
            lowered = lowered[: -len(suffix)]
    return lowered


def build_clear_lookup(clear_dir):
    lookup = {}
    if not os.path.isdir(clear_dir):
        return lookup
    for path in list_image_files(clear_dir):
        lookup[normalize_filename_key(path)] = path
    return lookup


def compute_atmospheric_light(image, window_size=8):
    dark_channel = np.min(image, axis=2)
    dark_min = minimum_filter(dark_channel, size=window_size)
    idx = np.argmax(dark_min)
    y, x = np.unravel_index(idx, dark_min.shape)
    A = image[y, x, :].copy()
    return A


def compute_psi_map_method1(H, A, psi_min=0.52, psi_max=1.38, buffer_size=8, epsilon=1e-6):
    height, width = H.shape[:2]
    psi_range = psi_max - psi_min

    diff = np.abs(H - A)
    relative_diff = diff / (A + epsilon)
    fog_density = np.min(relative_diff, axis=2) * 100
    fog_density = np.power(fog_density / 100.0, 1.05) * 100
    fog_density = np.clip(fog_density, 0, 100)

    raw_psi = psi_max - (fog_density / 100.0) * psi_range

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
    H = hazy_image.astype(np.float32)

    H_norm = H / (A + epsilon)
    K = np.mean(H_norm, axis=2)
    min_norm = np.min(H_norm, axis=2)

    psi_map = compute_psi_map_method1(H, A, psi_min, psi_max, buffer_size, epsilon)

    temp = 3 * K + 3 * min_norm
    t = (temp - psi_map * 3 * K * min_norm) / (temp + epsilon)
    t = np.clip(t, t0, 1)

    psi_normalized = (psi_map - psi_min) / (psi_max - psi_min + epsilon)
    psi_normalized = np.clip(psi_normalized, 0, 1)

    threshold_mask = np.where(
        psi_normalized < psi_threshold,
        psi_normalized / (psi_threshold + epsilon),
        1.0
    )
    psi_normalized_masked = psi_normalized * threshold_mask

    psi_weighted = np.power(psi_normalized_masked, strength_curve)

    adaptive_strength = 1.0 + (defogging_strength - 1.0) * psi_weighted

    t_adjusted = np.power(t, adaptive_strength)
    t_adjusted = np.clip(t_adjusted, t0, 1)

    t_expanded = t_adjusted[:, :, np.newaxis]
    D = (H - A) / t_expanded + A
    D = np.clip(D, 0, 255)

    return D, psi_map, t_adjusted, adaptive_strength


def create_heatmap(data, colormap=cv2.COLORMAP_JET):
    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)
    data_uint8 = (np.clip(data_normalized, 0, 1) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(data_uint8, colormap)
    return heatmap


def add_border(image, color=(70, 75, 85), thickness=3):
    return cv2.copyMakeBorder(image, thickness, thickness, thickness, thickness,
                              cv2.BORDER_CONSTANT, value=color)


def add_label(image, text, bg_color=None, text_color=None, accent=False):
    if bg_color is None:
        bg_color = UI_COLORS['bg_medium']
    if text_color is None:
        text_color = UI_COLORS['text_primary']

    h, w = image.shape[:2]
    label_height = 55

    label = np.zeros((label_height, w, 3), dtype=np.uint8)
    for i in range(label_height):
        alpha = i / label_height
        color = tuple(int(bg_color[j] * (1 - alpha * 0.15)) for j in range(3))
        label[i, :] = color

    if accent:
        accent_color = UI_COLORS['accent_blue']
        cv2.rectangle(label, (0, 0), (w, 4), accent_color, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (label_height + text_size[1]) // 2 + 2

    cv2.putText(label, text, (text_x + 2, text_y + 2),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(label, text, (text_x, text_y),
                font, font_scale, text_color, thickness, cv2.LINE_AA)

    return np.vstack([label, image])


def add_footer(image, text_lines, bg_color=None, text_color=None):
    if not text_lines:
        return image

    if bg_color is None:
        bg_color = UI_COLORS['bg_dark']
    if text_color is None:
        text_color = UI_COLORS['text_primary']

    h, w = image.shape[:2]
    line_height = 32
    padding = 15
    footer_height = padding * 2 + line_height * len(text_lines)

    footer = np.zeros((footer_height, w, 3), dtype=np.uint8)
    for i in range(footer_height):
        alpha = i / footer_height
        color = tuple(int(bg_color[j] * (1 + alpha * 0.2)) for j in range(3))
        footer[i, :] = color

    y = padding + 22
    for line in text_lines:
        if "PSNR" in line:
            metric_color = UI_COLORS['accent_green']
            bullet = "[P]"
        elif "SSIM" in line:
            metric_color = UI_COLORS['accent_blue']
            bullet = "[S]"
        elif "CIEDE" in line:
            metric_color = UI_COLORS['accent_orange']
            bullet = "[C]"
        else:
            metric_color = text_color
            bullet = "[-]"

        cv2.putText(footer, bullet, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, metric_color, 2, cv2.LINE_AA)
        cv2.putText(footer, line, (50, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, metric_color, 2, cv2.LINE_AA)
        y += line_height

    return np.vstack([image, footer])


def pad_panel_to_height(image, target_height, color=None):
    if color is None:
        color = UI_COLORS['bg_dark']

    current_height = image.shape[0]
    if current_height == target_height:
        return image
    if current_height > target_height:
        return cv2.resize(image, (image.shape[1], target_height),
                          interpolation=cv2.INTER_AREA)

    pad_bottom = target_height - current_height
    return cv2.copyMakeBorder(image, 0, pad_bottom, 0, 0,
                              cv2.BORDER_CONSTANT, value=color)


def create_navigation_bar(width, current_index, total_images, bg_color=None):
    if bg_color is None:
        bg_color = UI_COLORS['bg_medium']

    bar_height = 70
    nav_bar = np.zeros((bar_height, width, 3), dtype=np.uint8)

    for i in range(bar_height):
        alpha = i / bar_height
        color = tuple(int(bg_color[j] * (1 + alpha * 0.15)) for j in range(3))
        nav_bar[i, :] = color

    button_width = 120
    button_height = 45
    button_y = (bar_height - button_height) // 2
    button_spacing = 20

    prev_x = 30
    prev_button = (prev_x, button_y, prev_x + button_width, button_y + button_height)

    next_x = prev_x + button_width + button_spacing
    next_button = (next_x, button_y, next_x + button_width, button_y + button_height)

    button_color = UI_COLORS['bg_light']
    border_color = UI_COLORS['accent_blue']

    cv2.rectangle(nav_bar, (prev_button[0], prev_button[1]),
                  (prev_button[2], prev_button[3]), button_color, -1)
    cv2.rectangle(nav_bar, (prev_button[0], prev_button[1]),
                  (prev_button[2], prev_button[3]), border_color, 2)

    cv2.rectangle(nav_bar, (next_button[0], next_button[1]),
                  (next_button[2], next_button[3]), button_color, -1)
    cv2.rectangle(nav_bar, (next_button[0], next_button[1]),
                  (next_button[2], next_button[3]), border_color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_color = UI_COLORS['text_primary']

    prev_text = "<< PREV"
    prev_size = cv2.getTextSize(prev_text, font, font_scale, thickness)[0]
    prev_text_x = prev_button[0] + (button_width - prev_size[0]) // 2
    prev_text_y = prev_button[1] + (button_height + prev_size[1]) // 2
    cv2.putText(nav_bar, prev_text, (prev_text_x, prev_text_y),
                font, font_scale, text_color, thickness, cv2.LINE_AA)

    next_text = "NEXT >>"
    next_size = cv2.getTextSize(next_text, font, font_scale, thickness)[0]
    next_text_x = next_button[0] + (button_width - next_size[0]) // 2
    next_text_y = next_button[1] + (button_height + next_size[1]) // 2
    cv2.putText(nav_bar, next_text, (next_text_x, next_text_y),
                font, font_scale, text_color, thickness, cv2.LINE_AA)

    info_text = f"Image {current_index + 1} / {total_images}"
    info_size = cv2.getTextSize(info_text, font, 0.7, 2)[0]
    info_x = next_button[2] + 40
    info_y = bar_height // 2 + info_size[1] // 2
    cv2.putText(nav_bar, info_text, (info_x, info_y),
                font, 0.7, UI_COLORS['highlight'], 2, cv2.LINE_AA)

    save_x = width - 260
    reset_x = width - 130

    save_button = (save_x, button_y, save_x + 120, button_y + button_height)
    reset_button = (reset_x, button_y, reset_x + 120, button_y + button_height)

    save_color = UI_COLORS['accent_green']
    cv2.rectangle(nav_bar, (save_button[0], save_button[1]),
                  (save_button[2], save_button[3]), button_color, -1)
    cv2.rectangle(nav_bar, (save_button[0], save_button[1]),
                  (save_button[2], save_button[3]), save_color, 2)

    save_text = "SAVE (S)"
    save_size = cv2.getTextSize(save_text, font, 0.55, thickness)[0]
    save_text_x = save_button[0] + (120 - save_size[0]) // 2
    save_text_y = save_button[1] + (button_height + save_size[1]) // 2
    cv2.putText(nav_bar, save_text, (save_text_x, save_text_y),
                font, 0.55, save_color, thickness, cv2.LINE_AA)

    reset_color = UI_COLORS['accent_orange']
    cv2.rectangle(nav_bar, (reset_button[0], reset_button[1]),
                  (reset_button[2], reset_button[3]), button_color, -1)
    cv2.rectangle(nav_bar, (reset_button[0], reset_button[1]),
                  (reset_button[2], reset_button[3]), reset_color, 2)

    reset_text = "RESET (R)"
    reset_size = cv2.getTextSize(reset_text, font, 0.5, thickness)[0]
    reset_text_x = reset_button[0] + (120 - reset_size[0]) // 2
    reset_text_y = reset_button[1] + (button_height + reset_size[1]) // 2
    cv2.putText(nav_bar, reset_text, (reset_text_x, reset_text_y),
                font, 0.5, reset_color, thickness, cv2.LINE_AA)

    buttons = {
        'prev': prev_button,
        'next': next_button,
        'save': save_button,
        'reset': reset_button
    }

    return nav_bar, buttons


def create_stats_panel(stats_lines, width=450, bg_color=None, text_color=None):
    if bg_color is None:
        bg_color = UI_COLORS['bg_dark']
    if text_color is None:
        text_color = UI_COLORS['text_secondary']

    line_height = 28
    padding = 25
    height = len(stats_lines) * line_height + padding * 2

    panel = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        alpha = (i / height) * 0.1
        color = tuple(int(bg_color[j] * (1 + alpha)) for j in range(3))
        panel[i, :] = color

    header_color = UI_COLORS['accent_blue']
    cv2.rectangle(panel, (0, 0), (width, 5), header_color, -1)

    y = padding + 20
    for line in stats_lines:
        if line.strip() == "":
            y += line_height // 2
            continue

        indent_level = len(line) - len(line.lstrip())
        x_offset = padding + indent_level * 8

        if line.endswith(":") and not line.startswith("  "):
            font_scale = 0.65
            thickness = 2
            color = UI_COLORS['highlight']
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                        font_scale, thickness)[0]
            cv2.line(panel, (x_offset, y + 5),
                     (x_offset + text_size[0], y + 5), UI_COLORS['border'], 1)
        elif "Min:" in line or "Max:" in line or "Avg:" in line:
            font_scale = 0.55
            thickness = 1
            color = UI_COLORS['accent_green']
        elif "Keys:" in line:
            font_scale = 0.6
            thickness = 2
            color = UI_COLORS['accent_orange']
        else:
            font_scale = 0.52
            thickness = 1
            color = text_color

        cv2.putText(panel, line, (x_offset, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                    thickness, cv2.LINE_AA)
        y += line_height

    return panel


class InteractiveDehazingV6:
    """Interactive Dehazing Application - V6 Method 1"""

    def __init__(self, image_paths, start_index=0):
        if not image_paths:
            raise ValueError("No images provided for the interactive viewer.")

        self.image_files = image_paths
        self.total_images = len(image_paths)
        self.current_index = max(0, min(start_index, self.total_images - 1))
        self.clear_lookup = build_clear_lookup(CLEAR_DIR)

        self.psi_min = DEFAULT_PSI_MIN
        self.psi_max = DEFAULT_PSI_MAX
        self.t0 = DEFAULT_T0
        self.buffer_size = DEFAULT_BUFFER_SIZE
        self.window_size = DEFAULT_WINDOW_SIZE
        self.defogging_strength = DEFAULT_STRENGTH
        self.strength_curve = DEFAULT_STRENGTH_CURVE
        self.psi_threshold = DEFAULT_PSI_THRESHOLD

        self.image_path = None
        self.image_name = None
        self.image_original = None
        self.image = None
        self.preview_scale = 1.0
        self.clear_image_original = None
        self.clear_preview = None
        self.clear_path = None
        self.quality_metrics = None

        self.button_rects = {}

        self.load_image_at_index(self.current_index)
        self.update_atmospheric_light()

        self.window_name = "Interactive Dehazing V6 - Method 1 (Atmospheric Light Comparison)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1800, 1000)

        cv2.setMouseCallback(self.window_name, self.on_mouse_click)

        self.create_trackbars()

        self.needs_A_update = False
        self.update_display()

    def create_trackbars(self):
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

    def _prepare_preview(self, image_array):
        h, w = image_array.shape[:2]
        if h > PREVIEW_MAX_HEIGHT:
            scale = PREVIEW_MAX_HEIGHT / h
            new_h = PREVIEW_MAX_HEIGHT
            new_w = max(1, int(w * scale))
            resized = cv2.resize(image_array, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA).astype(np.float32)
            print(f"Resized for preview: {w}x{h} -> {new_w}x{new_h}")
            return resized, scale
        return image_array.copy(), 1.0

    def load_image_at_index(self, index):
        if not self.image_files:
            raise RuntimeError("Image file list is empty.")

        self.current_index = index % self.total_images
        image_path = self.image_files[self.current_index]
        self.image_path = image_path
        self.image_name = os.path.basename(image_path)

        with Image.open(image_path) as img:
            hazy = np.array(img.convert('RGB')).astype(np.float32)

        self.image_original = hazy
        self.image, self.preview_scale = self._prepare_preview(self.image_original)
        self.load_clear_reference()
        self.quality_metrics = None
        print(f"Loaded [{self.current_index + 1}/{self.total_images}]: {self.image_name}")

    def load_clear_reference(self):
        self.clear_preview = None
        self.clear_image_original = None
        self.clear_path = None

        if not self.clear_lookup:
            return

        key = normalize_filename_key(self.image_name)
        clear_path = self.clear_lookup.get(key)
        if not clear_path or not os.path.exists(clear_path):
            print(f"Ground truth not found for: {self.image_name}")
            return

        self.clear_path = clear_path
        with Image.open(clear_path) as img:
            clear = np.array(img.convert('RGB')).astype(np.float32)

        if clear.shape[:2] != self.image_original.shape[:2]:
            clear = cv2.resize(
                clear,
                (self.image_original.shape[1], self.image_original.shape[0]),
                interpolation=cv2.INTER_AREA
            ).astype(np.float32)

        self.clear_image_original = clear

        preview_width = self.image.shape[1]
        preview_height = self.image.shape[0]
        if clear.shape[:2] != (preview_height, preview_width):
            clear_preview = cv2.resize(clear, (preview_width, preview_height),
                                       interpolation=cv2.INTER_AREA)
        else:
            clear_preview = clear.copy()

        self.clear_preview = clear_preview.astype(np.float32)

    def show_next_image(self):
        if self.total_images <= 1:
            return
        next_index = (self.current_index + 1) % self.total_images
        self.load_image_at_index(next_index)
        self.update_atmospheric_light()
        self.needs_A_update = False
        self.update_display()

    def show_previous_image(self):
        if self.total_images <= 1:
            return
        prev_index = (self.current_index - 1) % self.total_images
        self.load_image_at_index(prev_index)
        self.update_atmospheric_light()
        self.needs_A_update = False
        self.update_display()

    def on_mouse_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if not hasattr(self, 'button_rects') or not self.button_rects:
            return

        if not hasattr(self, 'display_height'):
            return

        nav_bar_y_offset = self.display_height - 70
        relative_y = y - nav_bar_y_offset

        if relative_y < 0 or relative_y > 70:
            return

        for button_name, (bx1, by1, bx2, by2) in self.button_rects.items():
            if bx1 <= x <= bx2 and by1 <= relative_y <= by2:
                print(f"Button clicked: {button_name}")
                if button_name == 'prev':
                    self.show_previous_image()
                elif button_name == 'next':
                    self.show_next_image()
                elif button_name == 'save':
                    self.save_result()
                elif button_name == 'reset':
                    self.reset_parameters()
                break

    def on_trackbar_change(self, val):
        self.update_display()

    def on_window_size_change(self, val):
        self.needs_A_update = True
        self.update_display()

    def update_atmospheric_light(self):
        self.A = compute_atmospheric_light(self.image, window_size=self.window_size)

    def update_display(self):
        self.psi_min = cv2.getTrackbarPos('PSI Min (x0.01)', self.window_name)
        self.psi_max = cv2.getTrackbarPos('PSI Max (x0.01)', self.window_name)
        self.psi_threshold = cv2.getTrackbarPos('PSI Threshold (x0.01)', self.window_name)
        self.t0 = cv2.getTrackbarPos('t0 (x0.01)', self.window_name)
        self.defogging_strength = cv2.getTrackbarPos('Max Strength (x0.01)', self.window_name)
        self.strength_curve = cv2.getTrackbarPos('Strength Curve (x0.01)', self.window_name)
        self.buffer_size = cv2.getTrackbarPos('Buffer Size', self.window_name)
        self.window_size = cv2.getTrackbarPos('Window Size', self.window_name)

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

        if self.needs_A_update:
            self.update_atmospheric_light()
            self.needs_A_update = False

        psi_min_val = self.psi_min / 100.0
        psi_max_val = self.psi_max / 100.0
        psi_threshold_val = self.psi_threshold / 100.0
        t0_val = self.t0 / 100.0
        strength_val = self.defogging_strength / 100.0
        curve_val = self.strength_curve / 100.0

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
        self.quality_metrics = self.compute_quality_metrics()

        self.create_display(psi_min_val, psi_max_val, psi_threshold_val,
                            t0_val, strength_val, curve_val)

    def compute_quality_metrics(self):
        if self.clear_preview is None or self.current_result is None:
            return None

        result = np.clip(self.current_result, 0, 255).astype(np.uint8)
        clear = np.clip(self.clear_preview, 0, 255).astype(np.uint8)

        height = min(result.shape[0], clear.shape[0])
        width = min(result.shape[1], clear.shape[1])
        if height == 0 or width == 0:
            return None

        result = result[:height, :width]
        clear = clear[:height, :width]

        try:
            psnr_val = peak_signal_noise_ratio(clear, result, data_range=255)
        except Exception as exc:
            print(f"PSNR calculation failed: {exc}")
            psnr_val = None

        try:
            ssim_val = structural_similarity(clear, result, channel_axis=-1,
                                             data_range=255)
        except Exception as exc:
            print(f"SSIM calculation failed: {exc}")
            ssim_val = None

        try:
            clear_lab = rgb2lab(clear / 255.0)
            result_lab = rgb2lab(result / 255.0)
            delta_e = deltaE_ciede2000(clear_lab, result_lab)
            ciede_val = float(np.mean(delta_e))
        except Exception as exc:
            print(f"CIEDE2000 calculation failed: {exc}")
            ciede_val = None

        metrics = {
            'psnr': (float(psnr_val)
                     if psnr_val is not None and np.isfinite(psnr_val) else None),
            'ssim': (float(ssim_val)
                     if ssim_val is not None and np.isfinite(ssim_val) else None),
            'ciede': ciede_val
        }
        return metrics

    def create_display(self, psi_min_val, psi_max_val, psi_threshold_val,
                       t0_val, strength_val, curve_val):
        img_display = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        result_display = cv2.cvtColor(self.current_result.astype(np.uint8),
                                      cv2.COLOR_RGB2BGR)
        if self.clear_preview is not None:
            clear_display = cv2.cvtColor(self.clear_preview.astype(np.uint8),
                                         cv2.COLOR_RGB2BGR)
        else:
            clear_display = None

        psi_heatmap = create_heatmap(self.current_psi_map, cv2.COLORMAP_TURBO)
        strength_heatmap = create_heatmap(self.current_strength_map,
                                          cv2.COLORMAP_INFERNO)

        img1 = add_label(add_border(img_display),
                         "[HAZY]  Original Hazy Image", accent=True)
        img2 = add_label(add_border(psi_heatmap),
                         "[PSI]  PSI Map (Fog Density)", accent=False)
        img3 = add_label(add_border(strength_heatmap),
                         "[STRENGTH]  Adaptive Strength Map", accent=False)
        img4 = add_label(add_border(result_display),
                         "[RESULT]  Dehazed Result", accent=True)

        if clear_display is not None:
            img5 = add_label(add_border(clear_display),
                             "[GT]  Ground Truth (Clear)", accent=True)
            footer_lines = []
            if self.quality_metrics:
                psnr_text = (
                    f"PSNR: {self.quality_metrics['psnr']:.2f} dB"
                    if self.quality_metrics['psnr'] is not None
                    else "PSNR: N/A"
                )
                ssim_text = (
                    f"SSIM: {self.quality_metrics['ssim']:.4f}"
                    if self.quality_metrics['ssim'] is not None
                    else "SSIM: N/A"
                )
                ciede_text = (
                    f"CIEDE2000: {self.quality_metrics['ciede']:.2f}"
                    if self.quality_metrics['ciede'] is not None
                    else "CIEDE2000: N/A"
                )
                footer_lines = [psnr_text, ssim_text, ciede_text]
            else:
                footer_lines = ["PSNR: --", "SSIM: --", "CIEDE2000: --"]
            img5 = add_footer(img5, footer_lines)
        else:
            img5 = None

        spacing = 8
        spacing_color = UI_COLORS['bg_dark']

        h_spacer = np.full((img1.shape[0], spacing, 3), spacing_color, dtype=np.uint8)
        v_spacer = np.full((spacing, img1.shape[1] * 2 + spacing, 3),
                           spacing_color, dtype=np.uint8)

        top_row = np.hstack([img1, h_spacer, img2])
        bottom_row = np.hstack([img3, h_spacer, img4])
        image_grid = np.vstack([top_row, v_spacer, bottom_row])

        if img5 is not None:
            gt_panel = pad_panel_to_height(img5, image_grid.shape[0])
            gt_spacer = np.full((image_grid.shape[0], spacing, 3),
                                spacing_color, dtype=np.uint8)
            image_grid = np.hstack([image_grid, gt_spacer, gt_panel])

        stats = [
            f"FILE: {self.image_name}",
            f"IMAGE: {self.current_index + 1} of {self.total_images}",
            "",
            "[ALGORITHM PARAMETERS]",
            f"  PSI Min: {psi_min_val:.2f}",
            f"  PSI Max: {psi_max_val:.2f}",
            f"  PSI Threshold: {psi_threshold_val:.2f}",
            f"  t0: {t0_val:.2f}",
            f"  Max Strength: {strength_val:.2f}",
            f"  Curve Steepness: {curve_val:.2f}",
            f"  Buffer Size: {self.buffer_size}",
            f"  Window Size: {self.window_size}",
            "",
            "[STRENGTH MAP STATS]",
            f"  Min: {np.min(self.current_strength_map):.3f}",
            f"  Avg: {np.mean(self.current_strength_map):.3f}",
            f"  Max: {np.max(self.current_strength_map):.3f}",
            "",
            "[PSI MAP STATS]",
            f"  Min: {np.min(self.current_psi_map):.3f}",
            f"  Avg: {np.mean(self.current_psi_map):.3f}",
            f"  Max: {np.max(self.current_psi_map):.3f}",
            "",
            "[ATMOSPHERIC LIGHT]",
            f"  R: {self.A[0]:.1f}",
            f"  G: {self.A[1]:.1f}",
            f"  B: {self.A[2]:.1f}",
            "",
            "[HEAVY FOG SETTINGS]",
            "  Max Strength: 2.5~3.0",
            "  Curve: 4.0~8.0",
            "  PSI Threshold: 0.5~0.7",
            "  t0: 0.10~0.15",
            "",
            "[KEYBOARD CONTROLS]",
            "  Left/Right  Navigate images",
            "  S           Save result",
            "  R           Reset parameters",
            "  Q           Quit application",
        ]

        stats_panel = create_stats_panel(stats, width=520)

        stats_h = image_grid.shape[0]
        stats_w = stats_panel.shape[1]
        stats_panel_resized = cv2.resize(stats_panel, (stats_w, stats_h),
                                         interpolation=cv2.INTER_LINEAR)

        stats_spacer = np.full((image_grid.shape[0], spacing, 3),
                               spacing_color, dtype=np.uint8)
        display = np.hstack([image_grid, stats_spacer, stats_panel_resized])

        nav_bar, button_rects = create_navigation_bar(
            display.shape[1],
            self.current_index,
            self.total_images
        )

        self.button_rects = button_rects

        display = np.vstack([display, nav_bar])

        border_thickness = 5
        display = cv2.copyMakeBorder(display, border_thickness, border_thickness,
                                     border_thickness, border_thickness,
                                     cv2.BORDER_CONSTANT, value=UI_COLORS['bg_dark'])

        self.display_height = display.shape[0]

        cv2.imshow(self.window_name, display)

    def reset_parameters(self):
        cv2.setTrackbarPos('PSI Min (x0.01)', self.window_name, DEFAULT_PSI_MIN)
        cv2.setTrackbarPos('PSI Max (x0.01)', self.window_name, DEFAULT_PSI_MAX)
        cv2.setTrackbarPos('PSI Threshold (x0.01)', self.window_name,
                           DEFAULT_PSI_THRESHOLD)
        cv2.setTrackbarPos('t0 (x0.01)', self.window_name, DEFAULT_T0)
        cv2.setTrackbarPos('Max Strength (x0.01)', self.window_name, DEFAULT_STRENGTH)
        cv2.setTrackbarPos('Strength Curve (x0.01)', self.window_name,
                           DEFAULT_STRENGTH_CURVE)
        cv2.setTrackbarPos('Buffer Size', self.window_name, DEFAULT_BUFFER_SIZE)
        cv2.setTrackbarPos('Window Size', self.window_name, DEFAULT_WINDOW_SIZE)
        self.needs_A_update = True

    def save_result(self):
        """Save current result - 將預覽的 maps upscale 到原圖尺寸以確保一致性"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        base_name = os.path.splitext(self.image_name)[0]

        if self.preview_scale < 1.0:
            # 圖片有被縮放，不重跑演算法
            # 將預覽的 transmission map upscale 到原圖尺寸，直接做恢復
            print("Upscaling preview maps to full resolution...")
            H = self.image_original.astype(np.float32)
            orig_h, orig_w = H.shape[:2]

            t_full = cv2.resize(self.current_transmission, (orig_w, orig_h),
                                interpolation=cv2.INTER_LINEAR)
            psi_full = cv2.resize(self.current_psi_map, (orig_w, orig_h),
                                  interpolation=cv2.INTER_LINEAR)
            strength_full = cv2.resize(self.current_strength_map, (orig_w, orig_h),
                                       interpolation=cv2.INTER_LINEAR)

            t0_val = self.t0 / 100.0
            t_full = np.clip(t_full, t0_val, 1.0)
            t_expanded = t_full[:, :, np.newaxis]
            D = (H - self.A) / t_expanded + self.A
            D = np.clip(D, 0, 255)

            result_to_save = D
            psi_to_save = psi_full
            trans_to_save = t_full
            strength_to_save = strength_full
        else:
            # 沒有縮放，直接用 GUI 的計算結果
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
        strength_heatmap_path = os.path.join(OUTPUT_DIR,
                                              f"{base_name}_strength_heatmap.png")
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
            f.write(f"Atmospheric Light: R={self.A[0]:.2f}, G={self.A[1]:.2f}, "
                    f"B={self.A[2]:.2f}\n")
            f.write(f"Preview Scale: {self.preview_scale:.4f}\n")
            f.write(f"Save Method: "
                    f"{'upscaled maps' if self.preview_scale < 1.0 else 'direct'}\n")
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

        print(f"Saved: {output_path}")
        print(f"Saved: {psi_heatmap_path}")
        print(f"Saved: {strength_heatmap_path}")
        print(f"Saved: {params_path}")

    def run(self):
        print("\n" + "=" * 70)
        print("Interactive Dehazing V6 - STEEP CURVE Adaptive Method")
        print("=" * 70)
        print("Key Innovation: ULTRA STEEP curves for precise fog targeting")
        print("\nControls:")
        print("  - Adjust trackbars to see real-time effects")
        print("  - Press LEFT/RIGHT arrow to navigate dataset images")
        print("  - Press 'S' to save result")
        print("  - Press 'R' to reset to defaults")
        print("  - Press 'Q' or ESC to quit")
        print("=" * 70 + "\n")

        while True:
            key = cv2.waitKey(100) & 0xFF

            if key == ord('q') or key == ord('Q') or key == 27:
                break
            elif key == ord('s') or key == ord('S'):
                self.save_result()
            elif key == ord('r') or key == ord('R'):
                self.reset_parameters()
            elif key == 83 or key == 3 or key == 63235:
                self.show_next_image()
            elif key == 81 or key == 2 or key == 63234:
                self.show_previous_image()
            elif key != 255:
                print(f"Key pressed: {key}")

        cv2.destroyAllWindows()


def main():
    import sys

    dataset_images = list_image_files(INPUT_DIR)
    image_files = list(dataset_images)
    start_index = 0

    requested_path = None
    if len(sys.argv) > 1:
        requested_path = os.path.abspath(sys.argv[1])
        if not os.path.exists(requested_path):
            print(f"Image not found: {requested_path}")
            return

    if requested_path:
        if requested_path in image_files:
            start_index = image_files.index(requested_path)
        else:
            image_files = [requested_path]
            start_index = 0
            print("Provided image is outside the default dataset. "
                  "Navigation will be disabled.")
    else:
        if not image_files:
            print(f"No images found in {INPUT_DIR}")
            print(f"Usage: python interactive.py [optional_image_path]")
            return
        print(f"Using first dataset image: {image_files[0]}")

    app = InteractiveDehazingV6(image_files, start_index=start_index)
    app.run()


if __name__ == "__main__":
    main()