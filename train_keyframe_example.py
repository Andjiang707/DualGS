#!/usr/bin/env python3
"""
使用 key frame 策略的訓練範例腳本

這個腳本展示了如何使用修改後的 DualGS 以第5幀為 key frame 進行訓練。

使用方式:
python train_keyframe_example.py
"""

import subprocess
import os

def run_keyframe_training():
    """執行以第5幀為 key frame 的訓練"""
    
    # 設定訓練參數
    source_path = r"C:\DualGS\DualGS\datasets\4K_Actor2_Dancing\image_white_undistortion"
    model_path = r"C:\DualGS\DualGS\output\4K_Actor2_Dancing_keyframe_test"
    
    # 訓練指令
    cmd = [
        "python", "train.py",
        "-s", source_path,
        "-m", model_path,
        "--frame_st", "1",
        "--frame_ed", "199", 
        "--densify_until_iter", "1800",
        "--iterations", "5000",
        "--subseq_iters", "2000",
        "--training_mode", "0",
        "--parallel_load",
        "-r", "2",
        "--seq",
        "--key_frame", "5"  # 新增的 key frame 參數
    ]
    
    print("開始 key frame 訓練...")
    print(f"Key frame: 5")
    print(f"訓練順序將是: 5 -> 6,7,8,...,199 -> 4,3,2,1")
    print("執行指令:", " ".join(cmd))
    
    # 執行訓練
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("訓練完成!")
        print("輸出:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("訓練失敗!")
        print("錯誤:", e.stderr)
        return False
    
    return True

if __name__ == "__main__":
    print("=== DualGS Key Frame 訓練範例 ===")
    print()
    print("這個範例將使用第5幀作為 key frame:")
    print("1. 首先訓練第5幀 (densify + canonical)")
    print("2. 然後訓練正向幀 (6,7,8,...,199)")
    print("3. 最後訓練反向幀 (4,3,2,1)")
    print()
    
    # 確認是否繼續
    response = input("是否要開始訓練? (y/n): ")
    if response.lower() in ['y', 'yes', '是']:
        run_keyframe_training()
    else:
        print("取消訓練")