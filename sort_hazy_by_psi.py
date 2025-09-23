#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根據 CSV 文件中的 BestPsi 值來分類 hazy 照片
分類規則：
- 1.7 資料夾：放置 PSI >= 1.7 的照片
- 1.6 資料夾：放置 PSI >= 1.6 且 < 1.7 的照片
- 1.5 資料夾：放置 PSI >= 1.5 且 < 1.6 的照片
- ...以此類推
- 0.5 資料夾：放置 PSI >= 0.5 且 < 0.6 的照片
"""

import os
import csv
import shutil
from pathlib import Path
import math

def get_psi_folder(psi_value):
    """
    根據 PSI 值決定應該放入哪個資料夾
    1.7 放的是 >= 1.7
    1.6 放的是 >= 1.6 且 < 1.7
    ...
    0.5 放的是 >= 0.5 且 < 0.6
    """
    # 定義所有可能的資料夾值（從大到小）
    folder_values = [1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    
    # 找到第一個小於等於 psi_value 的資料夾值
    for folder_val in folder_values:
        if psi_value >= folder_val:
            return folder_val
    
    # 如果都不符合，放入最小的資料夾
    return 0.5

def create_psi_folders(base_path):
    """
    創建所有可能的 PSI 值資料夾 (從 1.7 到 0.5，以 0.1 為間隔)
    """
    psi_values = []
    # 從 1.7 到 0.5，以 0.1 為間隔
    current_psi = 1.7
    while current_psi >= 0.5:
        psi_values.append(round(current_psi, 1))
        current_psi -= 0.1
    
    created_folders = []
    for psi in psi_values:
        folder_name = f"{psi:.1f}"
        folder_path = base_path / folder_name
        folder_path.mkdir(exist_ok=True)
        created_folders.append(folder_name)
        print(f"已創建資料夾: {folder_path}")
    
    return created_folders

def sort_hazy_images():
    """
    主要功能：根據 CSV 中的 BestPsi 值分類 hazy 照片
    """
    # 設定路徑
    base_dir = Path("/Users/hongweichen/Documents/實驗室/論文/defog/昱恩交接/img_soft/dataset/SOTS_inout")
    csv_file = base_dir / "report" / "score_optimize_psi_grid.csv"
    hazy_dir = base_dir / "hazy"
    sort_dir = base_dir / "sort_hazy"
    
    # 檢查必要的檔案和資料夾是否存在
    if not csv_file.exists():
        print(f"錯誤：CSV 檔案不存在 - {csv_file}")
        return
    
    if not hazy_dir.exists():
        print(f"錯誤：hazy 資料夾不存在 - {hazy_dir}")
        return
    
    # 創建 sort_hazy 資料夾
    sort_dir.mkdir(exist_ok=True)
    print(f"已創建主資料夾: {sort_dir}")
    
    # 創建所有 PSI 值的子資料夾
    created_folders = create_psi_folders(sort_dir)
    
    # 讀取 CSV 檔案並處理每一行
    processed_count = 0
    not_found_count = 0
    error_count = 0
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            for row in csv_reader:
                try:
                    # 跳過平均值行
                    if row['Image'] == 'AVERAGE':
                        continue
                    
                    image_id = row['Image']
                    best_psi = float(row['BestPsi'])
                    
                    # 根據 PSI 值決定應該放入哪個資料夾
                    folder_psi = get_psi_folder(best_psi)
                    
                    # 顯示超出範圍的 PSI 值（僅供參考）
                    if best_psi < 0.5:
                        print(f"警告：PSI 值過小 - Image: {image_id}, PSI: {best_psi} -> 放入 0.5 資料夾")
                    elif best_psi > 1.7:
                        print(f"警告：PSI 值過大 - Image: {image_id}, PSI: {best_psi} -> 放入 1.7 資料夾")
                    
                    # 構建原始檔案路徑和目標檔案路徑
                    source_file = hazy_dir / f"{image_id}_hazy.png"
                    target_folder = sort_dir / f"{folder_psi:.1f}"
                    target_file = target_folder / f"{image_id}_hazy.png"
                    
                    # 檢查原始檔案是否存在
                    if not source_file.exists():
                        print(f"警告：找不到檔案 - {source_file}")
                        not_found_count += 1
                        continue
                    
                    # 複製檔案
                    shutil.copy2(source_file, target_file)
                    processed_count += 1
                    
                    if processed_count % 50 == 0:  # 每處理50個檔案顯示一次進度
                        print(f"已處理 {processed_count} 個檔案...")
                    
                except ValueError as e:
                    print(f"錯誤：無法解析 PSI 值 - Image: {image_id}, BestPsi: {row['BestPsi']}, 錯誤: {e}")
                    error_count += 1
                    continue
                except Exception as e:
                    print(f"錯誤：處理檔案時發生錯誤 - Image: {image_id}, 錯誤: {e}")
                    error_count += 1
                    continue
    
    except Exception as e:
        print(f"錯誤：讀取 CSV 檔案時發生錯誤 - {e}")
        return
    
    # 顯示統計資訊
    print("\n" + "="*50)
    print("處理完成！統計資訊：")
    print(f"成功處理的檔案數量: {processed_count}")
    print(f"找不到的檔案數量: {not_found_count}")
    print(f"發生錯誤的檔案數量: {error_count}")
    print(f"已創建的資料夾數量: {len(created_folders)}")
    print("="*50)
    
    # 顯示每個資料夾的檔案數量
    print("\n各資料夾的檔案數量：")
    total_files = 0
    for folder_name in sorted(created_folders, reverse=True):  # 從大到小排序
        folder_path = sort_dir / folder_name
        file_count = len(list(folder_path.glob("*.png")))
        total_files += file_count
        print(f"  {folder_name}: {file_count} 個檔案")
    
    print(f"\n總計: {total_files} 個檔案")

if __name__ == "__main__":
    print("開始根據 BestPsi 值分類 hazy 照片...")
    sort_hazy_images()
    print("分類完成！")
