#!/usr/bin/env python3
"""
修正後的 key frame 訓練測試腳本

這個腳本測試修正後的 key frame 功能，包含對點雲數量變化的處理。
"""

import subprocess
import os
import sys

def test_keyframe_training():
    """測試修正後的 key frame 訓練"""
    
    print("=== 測試修正後的 Key Frame 訓練 ===")
    print()
    
    # 設定訓練參數 - 使用較少的幀進行測試
    source_path = r"C:\Users\User\DualGS\_DB\image_white_undistortion2"
    model_path = r"C:\Users\User\DualGS\_OUT\4K_Actor2_Dancing_keyframe_fix_test"
    
    # 使用較小的範圍進行測試
    frame_st = 1
    frame_ed = 10  # 只測試前10幀
    key_frame = 5
    
    # 訓練指令
    cmd = [
        "python", "train.py",
        "-s", source_path,
        "-m", model_path,
        "--frame_st", str(frame_st),
        "--frame_ed", str(frame_ed), 
        "--densify_until_iter", "1000",  # 減少迭代次數用於測試
        "--iterations", "2000",
        "--subseq_iters", "1000",
        "--training_mode", "0",
        "--parallel_load",
        "-r", "2",
        "--seq",
        "--key_frame", str(key_frame)
    ]
    
    print(f"測試設定:")
    print(f"  Key frame: {key_frame}")
    print(f"  Frame 範圍: {frame_st} - {frame_ed}")
    print(f"  訓練順序: {key_frame} -> {list(range(key_frame+1, frame_ed))} -> {list(range(key_frame-1, frame_st-1, -1))}")
    print()
    print("執行指令:", " ".join(cmd))
    print()
    
    # 確保輸出目錄存在
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # 執行訓練
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1,
            universal_newlines=True
        )
        
        # 即時顯示輸出
        for line in process.stdout:
            print(line.strip())
            sys.stdout.flush()
        
        # 等待完成
        return_code = process.wait()
        
        if return_code == 0:
            print("\n✅ 訓練成功完成!")
            
            # 檢查生成的檔案
            ckt_dir = os.path.join(model_path, "ckt")
            if os.path.exists(ckt_dir):
                ply_files = [f for f in os.listdir(ckt_dir) if f.endswith('.ply')]
                print(f"生成的 PLY 檔案: {len(ply_files)} 個")
                for f in sorted(ply_files):
                    print(f"  - {f}")
            
            return True
        else:
            print(f"\n❌ 訓練失敗，返回碼: {return_code}")
            return False
            
    except Exception as e:
        print(f"\n❌ 執行錯誤: {e}")
        return False

def main():
    print("這個腳本將測試修正後的 key frame 訓練功能")
    print("主要修正:")
    print("1. 處理點雲數量變化導致的張量尺寸不匹配")
    print("2. 自動重新初始化 graph 結構")
    print("3. 增強錯誤處理和日誌輸出")
    print()
    
    response = input("是否開始測試? (y/n): ")
    if response.lower() in ['y', 'yes', '是']:
        success = test_keyframe_training()
        if success:
            print("\n🎉 測試完成！Key frame 功能運作正常。")
        else:
            print("\n⚠️  測試遇到問題，請檢查錯誤輸出。")
    else:
        print("取消測試")

if __name__ == "__main__":
    main()