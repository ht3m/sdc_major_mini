import subprocess
import os
import sys
import time

def run_pipeline():
    # 获取当前 Python解释器路径 (确保用的是同一个虚拟环境)
    python_exe = sys.executable
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # --- 1. 定义要静默运行的脚本 (计算步骤) ---
    # 这些脚本里有 plt.show()，但我们要抑制它弹出窗口
    calculation_steps = [
        "step01_preprocess.py",
        "step02_skeleton.py",
        "step03_graph.py",
        "step04_trajectories.py"
    ]

    # --- 2. 设置静默环境 ---
    # 复制当前环境变量
    silent_env = os.environ.copy()
    # 【核心魔法】强制 Matplotlib 使用 'Agg' 后端
    # 'Agg' 是非交互式后端，plt.show() 在此模式下不会弹出窗口，也不会阻塞程序
    silent_env["MPLBACKEND"] = "Agg"

    print("🚀 开始执行全自动流水线...")
    print("=" * 50)

    # --- 3. 依次执行计算步骤 ---
    for script_name in calculation_steps:
        script_path = os.path.join(current_dir, script_name)
        
        if not os.path.exists(script_path):
            print(f"❌ 找不到文件: {script_name}")
            return

        print(f"▶️  正在运行: {script_name} ...")
        start_time = time.time()
        
        # 使用 subprocess 启动子进程
        # 传入 silent_env 环境变量，抑制 plt.show()
        try:
            result = subprocess.run(
                [python_exe, script_path], 
                env=silent_env, 
                check=True # 如果脚本报错(返回码非0)，这里会抛出异常
            )
            elapsed = time.time() - start_time
            print(f"✅ {script_name} 完成 (耗时 {elapsed:.2f}s)\n")
            
        except subprocess.CalledProcessError:
            print(f"\n❌ {script_name} 执行失败！流水线终止。")
            print("请检查上方的报错信息。")
            return

    # --- 4. 执行最后的可视化步骤 (正常显示) ---
    show_script = "show.py"
    show_path = os.path.join(current_dir, show_script)
    
    if os.path.exists(show_path):
        print("=" * 50)
        print(f"👀 所有计算完成，正在打开最终结果: {show_script}")
        
        # 这里使用默认环境变量 (不加 MPLBACKEND=Agg)，所以 plt.show() 会正常弹窗
        subprocess.run([python_exe, show_path])
    else:
        print(f"❌ 找不到展示脚本: {show_script}")

if __name__ == "__main__":
    run_pipeline()