import gradio as gr
import subprocess
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import json
from datetime import datetime

def build_command(
    source_path,
    output_path,
    frame_st,
    frame_ed,
    key_frame,
    training_mode,
    motion_folder,
    parallel_load,
    resolution,
    use_seq,
    densify_until_iter,
    iterations,
    subseq_iters,
    force_retrain
):
    """構建訓練指令"""
    
    # 基礎指令
    cmd = [
        "python", "train.py",
        "-s", source_path,
        "-m", output_path,
        "--frame_st", str(frame_st),
        "--frame_ed", str(frame_ed),
        "--training_mode", str(training_mode)
    ]
    
    # 添加可選參數
    if key_frame is not None and key_frame >= 0:
        cmd.extend(["--key_frame", str(key_frame)])
    
    if training_mode == 2 and motion_folder:
        cmd.extend(["--motion_folder", motion_folder])
    
    if parallel_load:
        cmd.append("--parallel_load")
    
    if resolution > 0:
        cmd.extend(["-r", str(resolution)])
    
    if use_seq:
        cmd.append("--seq")
    
    if densify_until_iter:
        cmd.extend(["--densify_until_iter", str(densify_until_iter)])
    
    if iterations:
        cmd.extend(["--iterations", str(iterations)])
    
    if subseq_iters:
        cmd.extend(["--subseq_iters", str(subseq_iters)])
    
    if force_retrain:
        cmd.append("--force")
    
    return cmd

def run_training(
    source_path,
    output_path,
    frame_st,
    frame_ed,
    key_frame,
    training_mode,
    motion_folder,
    parallel_load,
    resolution,
    use_seq,
    densify_until_iter,
    iterations,
    subseq_iters,
    force_retrain
):
    """執行訓練"""
    try:
        # 驗證路徑
        if not os.path.exists(source_path):
            return f"❌ 錯誤：資料路徑不存在: {source_path}"
        
        if training_mode == 2 and motion_folder and not os.path.exists(motion_folder):
            return f"❌ 錯誤：Motion 資料夾不存在: {motion_folder}"
        
        # 構建指令
        cmd = build_command(
            source_path, output_path, frame_st, frame_ed, key_frame,
            training_mode, motion_folder, parallel_load, resolution,
            use_seq, densify_until_iter, iterations, subseq_iters, force_retrain
        )
        
        # 顯示指令
        cmd_str = " ".join(cmd)
        output = f"🚀 執行指令：\n{cmd_str}\n\n"
        output += "=" * 80 + "\n"
        
        # 執行指令
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 即時輸出
        for line in process.stdout:
            output += line
            yield output
        
        process.wait()
        
        if process.returncode == 0:
            output += "\n" + "=" * 80 + "\n"
            output += "✅ 訓練完成！\n"
        else:
            output += "\n" + "=" * 80 + "\n"
            output += f"❌ 訓練失敗，返回碼: {process.returncode}\n"
        
        yield output
        
    except Exception as e:
        yield f"❌ 執行錯誤：{str(e)}"

def get_training_mode_description(mode):
    """取得訓練模式說明"""
    descriptions = {
        0: """
### 📘 完整訓練模式 (Joint + Skin)
同時訓練 Joint 和 Skin，完整的端到端訓練流程。

**適用情境：**
- 初次訓練新資料集
- 需要完整的動作和外觀模型
- 一次性完成所有訓練

**輸出：**
- Joint motion 資料：`{output_path}/track/ckt/`
- Skin 模型：`{output_path}/ckt/`
""",
        1: """
### 🦴 Joint Only 模式
只訓練 Joint 動作資料，不產生 Skin。

**適用情境：**
- 只需要動作資料
- 後續想用不同參數訓練多個 Skin
- 先建立動作基礎，節省重複訓練時間

**輸出：**
- Joint motion 資料：`{output_path}/track/ckt/`

**下一步：**
使用 Skin Only 模式，指定此輸出路徑的 track 資料夾
""",
        2: """
### 🎨 Skin Only 模式
使用已訓練的 Joint 資料產生 Skin。

**適用情境：**
- 已有 Joint motion 資料
- 想用不同參數訓練 Skin
- 快速測試不同 Skin 設定

**必需設定：**
⚠️ 必須填寫「Motion 資料夾路徑」，指向之前訓練的 Joint 路徑
例如：`C:\\Users\\User\\DualGS\\_OUT\\joint_only\\track`

**輸出：**
- Skin 模型：`{output_path}/ckt/`
"""
    }
    return descriptions.get(mode, "")

def update_motion_folder_visibility(training_mode):
    """根據訓練模式更新 motion_folder 輸入框的可見性"""
    return gr.update(visible=(training_mode == 2))

def browse_folder(current_path=""):
    """開啟資料夾選擇對話框"""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder_path = filedialog.askdirectory(
        title="選擇資料夾",
        initialdir=current_path if current_path and os.path.exists(current_path) else os.path.expanduser("~")
    )
    root.destroy()
    return folder_path if folder_path else current_path

def save_config(source_path, output_path, frame_st, frame_ed, key_frame,
                training_mode, motion_folder, parallel_load, resolution,
                use_seq, densify_until_iter, iterations, subseq_iters, force_retrain):
    """保存配置到 JSON 檔案"""
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        
        # 預設檔名使用時間戳
        default_filename = f"train_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        file_path = filedialog.asksaveasfilename(
            title="保存訓練配置",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=default_filename
        )
        root.destroy()
        
        if not file_path:
            return "❌ 取消保存"
        
        config = {
            "source_path": source_path,
            "output_path": output_path,
            "frame_st": frame_st,
            "frame_ed": frame_ed,
            "key_frame": key_frame,
            "training_mode": training_mode,
            "motion_folder": motion_folder,
            "parallel_load": parallel_load,
            "resolution": resolution,
            "use_seq": use_seq,
            "densify_until_iter": densify_until_iter,
            "iterations": iterations,
            "subseq_iters": subseq_iters,
            "force_retrain": force_retrain,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return f"✅ 配置已保存至：{file_path}"
        
    except Exception as e:
        return f"❌ 保存失敗：{str(e)}"

def load_config():
    """從 JSON 檔案載入配置"""
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        
        file_path = filedialog.askopenfilename(
            title="載入訓練配置",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        root.destroy()
        
        if not file_path:
            return [None] * 14 + ["❌ 取消載入"]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return [
            config.get("source_path", ""),
            config.get("output_path", ""),
            config.get("frame_st", 0),
            config.get("frame_ed", 11),
            config.get("key_frame", 5),
            config.get("training_mode", 0),
            config.get("motion_folder", ""),
            config.get("parallel_load", True),
            config.get("resolution", 2),
            config.get("use_seq", True),
            config.get("densify_until_iter", 1800),
            config.get("iterations", 5000),
            config.get("subseq_iters", 2000),
            config.get("force_retrain", False),
            f"✅ 配置已載入：{file_path}\n保存時間：{config.get('saved_at', '未知')}"
        ]
        
    except Exception as e:
        return [None] * 14 + [f"❌ 載入失敗：{str(e)}"]

def get_preset_command(preset):
    """取得預設指令的參數"""
    presets = {
        "完整訓練": {
            "training_mode": 0,
            "parallel_load": True,
            "resolution": 2,
            "use_seq": True,
            "densify_until_iter": 1800,
            "iterations": 5000,
            "subseq_iters": 2000,
        },
        "只訓練 Joint": {
            "training_mode": 1,
            "parallel_load": True,
            "resolution": 2,
            "use_seq": True,
            "densify_until_iter": 1800,
            "iterations": 5000,
            "subseq_iters": 2000,
        },
        "只訓練 Skin": {
            "training_mode": 2,
            "parallel_load": True,
            "resolution": 2,
            "use_seq": True,
            "iterations": 5000,
            "subseq_iters": 2000,
        },
    }
    
    params = presets.get(preset, presets["完整訓練"])
    
    return (
        params["training_mode"],
        params.get("parallel_load", True),
        params.get("resolution", 2),
        params.get("use_seq", True),
        params.get("densify_until_iter", 0),
        params.get("iterations", 5000),
        params.get("subseq_iters", 2000),
    )

# 創建 Gradio 介面
with gr.Blocks(title="Funique DualGS 訓練介面", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 Funique DualGS 訓練控制介面")
    gr.Markdown("簡化訓練流程，一鍵執行訓練指令")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 路徑設定")
            
            with gr.Row():
                source_path = gr.Textbox(
                    label="資料來源路徑 (-s)",
                    placeholder=r"C:\Users\User\DualGS\_DB\image_white_undistortion2",
                    value=r"C:\Users\User\DualGS\_DB\image_white_undistortion2",
                    scale=4
                )
                browse_source_btn = gr.Button("📁 瀏覽", scale=1, size="sm")
            
            with gr.Row():
                output_path = gr.Textbox(
                    label="輸出路徑 (-m)",
                    placeholder=r"C:\Users\User\DualGS\_OUT\output_name",
                    value=r"C:\Users\User\DualGS\_OUT\4K_Actor2_Dancing_test",
                    scale=4
                )
                browse_output_btn = gr.Button("📁 瀏覽", scale=1, size="sm")
            
            with gr.Row():
                motion_folder = gr.Textbox(
                    label="Motion 資料夾路徑（僅 Skin Only 模式需要）",
                    placeholder=r"C:\Users\User\DualGS\_OUT\joint_only\track",
                    visible=False,
                    scale=4
                )
                browse_motion_btn = gr.Button("📁 瀏覽", scale=1, size="sm", visible=False)
            
            gr.Markdown("### ⚙️ 訓練參數")
            
            preset = gr.Radio(
                choices=["完整訓練", "只訓練 Joint", "只訓練 Skin"],
                value="完整訓練",
                label="預設模式"
            )
            
            training_mode = gr.Radio(
                choices=[(0, "完整訓練 (Joint + Skin)"), (1, "只訓練 Joint"), (2, "只訓練 Skin")],
                value=0,
                label="訓練模式 (--training_mode)"
            )
            
            mode_description = gr.Markdown(
                value=get_training_mode_description(0),
                elem_classes="mode-description"
            )
            
            with gr.Row():
                frame_st = gr.Number(label="起始幀 (--frame_st)", value=0, precision=0)
                frame_ed = gr.Number(label="結束幀 (--frame_ed)", value=11, precision=0)
                key_frame = gr.Number(label="關鍵幀 (--key_frame)", value=5, precision=0)
            
            with gr.Row():
                densify_until_iter = gr.Number(label="Densify 迭代數", value=1800, precision=0)
                iterations = gr.Number(label="總迭代數 (--iterations)", value=5000, precision=0)
                subseq_iters = gr.Number(label="子序列迭代數 (--subseq_iters)", value=2000, precision=0)
            
            gr.Markdown("### 🔧 進階選項")
            with gr.Row():
                parallel_load = gr.Checkbox(label="並行載入 (--parallel_load)", value=True)
                use_seq = gr.Checkbox(label="使用序列模式 (--seq)", value=True)
                force_retrain = gr.Checkbox(label="強制重新訓練 (--force)", value=False)
            
            resolution = gr.Slider(
                minimum=1,
                maximum=4,
                step=1,
                value=2,
                label="解析度縮放 (-r)"
            )
            
            gr.Markdown("### 💾 配置管理")
            with gr.Row():
                save_config_btn = gr.Button("💾 保存配置", size="sm")
                load_config_btn = gr.Button("📂 載入配置", size="sm")
            
            config_status = gr.Textbox(
                label="配置狀態",
                lines=2,
                interactive=False
            )
            
            run_btn = gr.Button("🚀 開始訓練", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 訓練輸出")
            output = gr.Textbox(
                label="執行狀態",
                lines=30,
                max_lines=30,
                show_copy_button=True
            )
    
    # 配置管理事件
    save_config_btn.click(
        fn=save_config,
        inputs=[
            source_path, output_path, frame_st, frame_ed, key_frame,
            training_mode, motion_folder, parallel_load, resolution, use_seq,
            densify_until_iter, iterations, subseq_iters, force_retrain
        ],
        outputs=[config_status]
    )
    
    load_config_btn.click(
        fn=load_config,
        inputs=[],
        outputs=[
            source_path, output_path, frame_st, frame_ed, key_frame,
            training_mode, motion_folder, parallel_load, resolution, use_seq,
            densify_until_iter, iterations, subseq_iters, force_retrain,
            config_status
        ]
    )
    
    # 瀏覽按鈕事件
    browse_source_btn.click(
        fn=browse_folder,
        inputs=[source_path],
        outputs=[source_path]
    )
    
    browse_output_btn.click(
        fn=browse_folder,
        inputs=[output_path],
        outputs=[output_path]
    )
    
    browse_motion_btn.click(
        fn=browse_folder,
        inputs=[motion_folder],
        outputs=[motion_folder]
    )
    
    # 事件處理
    preset.change(
        fn=get_preset_command,
        inputs=[preset],
        outputs=[training_mode, parallel_load, resolution, use_seq, 
                densify_until_iter, iterations, subseq_iters]
    )
    
    def update_motion_visibility(mode):
        return (
            gr.update(visible=(mode == 2)),
            gr.update(visible=(mode == 2)),
            get_training_mode_description(mode)
        )
    
    training_mode.change(
        fn=update_motion_visibility,
        inputs=[training_mode],
        outputs=[motion_folder, browse_motion_btn, mode_description]
    )
    
    run_btn.click(
        fn=run_training,
        inputs=[
            source_path, output_path, frame_st, frame_ed, key_frame,
            training_mode, motion_folder, parallel_load, resolution, use_seq,
            densify_until_iter, iterations, subseq_iters, force_retrain
        ],
        outputs=[output]
    )
    
    gr.Markdown(r"""
    ---
    ### 💡 使用說明
    
    **完整訓練模式 (0)**：同時訓練 Joint 和 Skin
    
    **只訓練 Joint (1)**：
    1. 只產生 Joint 動作資料
    2. 輸出會儲存到 `{output_path}/track/ckt/`
    
    **只訓練 Skin (2)**：
    1. 使用已訓練好的 Joint 產生 Skin
    2. 需要在「Motion 資料夾路徑」填入之前訓練的 Joint 路徑
    3. 例如：`C:\Users\User\DualGS\_OUT\joint_only\track`
    
    **Key Frame 策略**：
    - 設定 key_frame 後，會先訓練該幀，再向前後擴展
    - 可提高訓練品質和一致性
    """)
    
    gr.Markdown("---")
    gr.Markdown("<div style='text-align: right; color: #666; font-size: 0.9em;'>Developed by Harry</div>")

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)