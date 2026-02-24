# psi_map_methods.py
# 三種 PSI Map 計算方式

import numpy as np


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
    diff = np.abs(H - A)
    relative_diff = diff / (A + epsilon)
    fog_density = np.mean(relative_diff, axis=2) * 100
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


def compute_psi_map_method2(H, A, psi_min=0.52, psi_max=1.38, buffer_size=8, epsilon=1e-6):
    """
    方法二：局部對比度法 (Local Contrast Method)

    原理：
    - 霧氣會降低圖像的局部對比度
    - 在 line buffer 內計算 max - min 作為對比度
    - 對比度低 = 霧濃 = PSI 高
    - 對比度高 = 無霧 = PSI 低

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

    # 取灰階 (R channel，與硬體一致)
    gray = H[:, :, 0]

    psi_map = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        line_buffer = np.zeros(buffer_size, dtype=np.float32)
        buffer_count = 0
        head = 0

        for j in range(width):
            current_gray = gray[i, j]

            if buffer_count < buffer_size:
                buffer_count += 1
            line_buffer[head] = current_gray
            head = (head + 1) % buffer_size

            # 計算局部對比度
            valid_buffer = line_buffer[:buffer_count] if buffer_count < buffer_size else line_buffer
            dynamic_range = np.max(valid_buffer) - np.min(valid_buffer)

            # 對比度 → 霧分數 (分段映射)
            if dynamic_range >= 240:
                fog_score = 0
            elif dynamic_range <= 30:
                fog_score = 100
            else:
                fog_score = int((240 - dynamic_range) / 2.1)
            fog_score = max(0, min(100, fog_score))

            # 映射到 PSI
            psi = psi_min + (fog_score / 100.0) * psi_range
            psi_map[i, j] = np.clip(psi, psi_min, psi_max)

    return psi_map


def compute_psi_map_method3(H, A, psi_min=0.52, psi_max=1.38, buffer_size=8, epsilon=1e-6,
                            alpha=0.6, contrast_threshold=50, brightness_threshold=200):
    """
    方法三：混合法 (Hybrid Method)

    原理：
    - 結合大氣光比較 + 局部對比度
    - 根據場景特性用 if-else 選擇策略

    策略：
    - 高亮度 + 低對比度 → 霧區 → PSI 高
    - 低亮度 + 高對比度 → 前景 → PSI 低
    - 其他 → 加權混合

    參數:
        H: 輸入圖像 (float32, shape: height x width x 3)
        A: 大氣光向量 (shape: 3,)
        psi_min: PSI 最小值
        psi_max: PSI 最大值
        buffer_size: line buffer 大小
        epsilon: 防除零
        alpha: 大氣光方法的權重
        contrast_threshold: 對比度閾值
        brightness_threshold: 亮度閾值

    返回:
        psi_map: PSI 分布圖 (shape: height x width)
    """
    height, width = H.shape[:2]
    psi_range = psi_max - psi_min

    # 預計算大氣光霧分數
    diff = np.abs(H - A)
    relative_diff = diff / (A + epsilon)
    fog_density_atm = np.mean(relative_diff, axis=2) * 100
    fog_density_atm = np.clip(fog_density_atm, 0, 100)
    fog_score_atm = 100 - fog_density_atm  # 高分 = 霧濃

    # 灰階
    gray = H[:, :, 0]

    psi_map = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        gray_buffer = np.zeros(buffer_size, dtype=np.float32)
        atm_buffer = np.zeros(buffer_size, dtype=np.float32)
        buffer_count = 0
        head = 0

        for j in range(width):
            current_gray = gray[i, j]
            current_atm_score = fog_score_atm[i, j]

            if buffer_count < buffer_size:
                buffer_count += 1
            gray_buffer[head] = current_gray
            atm_buffer[head] = current_atm_score
            head = (head + 1) % buffer_size

            # 局部對比度
            valid_gray = gray_buffer[:buffer_count] if buffer_count < buffer_size else gray_buffer
            dynamic_range = np.max(valid_gray) - np.min(valid_gray)

            # 對比度霧分數
            if dynamic_range >= 200:
                fog_score_contrast = 0
            elif dynamic_range <= 20:
                fog_score_contrast = 100
            else:
                fog_score_contrast = int((200 - dynamic_range) / 1.8)
            fog_score_contrast = max(0, min(100, fog_score_contrast))

            # 大氣光霧分數 (平滑)
            valid_atm = atm_buffer[:buffer_count] if buffer_count < buffer_size else atm_buffer
            fog_score_atm_smooth = np.mean(valid_atm)

            # 當前像素亮度
            pixel_brightness = np.mean(H[i, j, :])

            # 條件判斷
            if pixel_brightness > brightness_threshold and dynamic_range < contrast_threshold:
                # 高亮 + 低對比 = 霧濃區
                fog_score_final = max(fog_score_atm_smooth, fog_score_contrast) * 1.1
            elif pixel_brightness < brightness_threshold * 0.6 and dynamic_range > contrast_threshold:
                # 低亮 + 高對比 = 前景
                fog_score_final = min(fog_score_atm_smooth, fog_score_contrast) * 0.8
            else:
                # 加權混合
                fog_score_final = alpha * fog_score_atm_smooth + (1 - alpha) * fog_score_contrast

            fog_score_final = max(0, min(100, fog_score_final))

            # 映射到 PSI
            psi = psi_min + (fog_score_final / 100.0) * psi_range
            psi_map[i, j] = np.clip(psi, psi_min, psi_max)

    return psi_map
