import os
import sys
import subprocess
import time

def run_script_headless(script_name):
    """
    读取脚本内容，注入屏蔽 GUI 的代码，保存为临时文件并运行。
    """
    if not os.path.exists(script_name):
        print(f"❌ 错误: 找不到文件 {script_name}")
        return False

    print(f"\n{'='*60}")
    print(f"🚀 正在执行: {script_name} (无窗口模式)")
    print(f"{'='*60}")

    # 读取源代码
    with open(script_name, 'r', encoding='utf-8') as f:
        original_code = f.read()

    # --- 注入屏蔽 GUI 的代码头 ---
    # 1. 强制 Matplotlib 使用 Agg 后端 (不显示 UI)
    # 2. 将 plt.show, cv2.imshow, cv2.waitKey 替换为空函数
    headless_header = """
import matplotlib
matplotlib.use('Agg') # 强制使用非交互式后端
import matplotlib.pyplot as plt
import cv2

# 定义空函数
def dummy_func(*args, **kwargs):
    pass

# 覆盖阻塞函数
plt.show = dummy_func
cv2.imshow = dummy_func
cv2.waitKey = dummy_func
print("   [System] GUI 显示函数已被 pipeline 屏蔽，程序将自动继续...")

# ================= 原程序代码开始 =================
"""
    
    # 合并代码
    full_code = headless_header + original_code
    
    # 创建临时文件 (例如: _temp_step01.py)
    temp_filename = f"_temp_{script_name}"
    
    try:
        with open(temp_filename, 'w', encoding='utf-8') as f:
            f.write(full_code)
            
        # 使用当前 Python 解释器运行临时文件
        # check=True 会在脚本报错时抛出异常，停止流水线
        subprocess.run([sys.executable, temp_filename], check=True)
        
        print(f"✅ {script_name} 执行完毕。")
        return True
        
    except subprocess.CalledProcessError:
        print(f"❌ {script_name} 执行出错，流水线终止。")
        return False
    except Exception as e:
        print(f"❌ 发生异常: {e}")
        return False
    finally:
        # 清理临时文件
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def main():
    # 记录总开始时间
    start_time = time.time()

    # === 定义执行顺序 ===
    # 注意：
    # 1. Step 07 使用 multiprocessing 版本
    # 2. Step 07 后增加了 Step 07_5
    scripts_to_run = [
        "step01_preprocess.py",
        "step02_skeleton.py",
        "step03_graph.py",
        "step04_trajectories.py",
        "step05_workspace_check.py",
        "step06_full_path.py",
        "step07_reverse_multicore.py", # 指定运行多核版本
        "step07_5.py",                # 插入的中间步骤
        "step08_smooth.py",
        "step09_check.py"
    ]

    # 获取当前脚本所在目录，确保路径正确
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir) # 切换工作目录到脚本所在位置

    print(f"📂 工作目录: {current_dir}")
    print(f"📋 计划执行 {len(scripts_to_run)} 个步骤...")

    for script in scripts_to_run:
        success = run_script_headless(script)
        if not success:
            print("\n⛔ 流水线因错误而中断。")
            sys.exit(1)
        
        # 稍微暂停一下，避免文件系统IO冲突
        time.sleep(0.5)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🎉🎉🎉 所有步骤执行完成！")
    print(f"⏱️ 总耗时: {total_time:.2f} 秒")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()