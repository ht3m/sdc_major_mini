import json
import numpy as np
import os
import sys

# ==========================================
# 配置区域
# ==========================================
# 浮点数对比的容差 (Tolerance)
# ATOL = 1e-5  # 绝对容差 (Absolute Tolerance)
# RTOL = 1e-3  # 相对容差 (Relative Tolerance)

def load_json(path):
    if not os.path.exists(path):
        print(f"❌ 错误: 找不到文件 {path}")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 错误: 读取文件 {path} 失败 - {e}")
        return None

def compare_structured_joints(file_a, file_b):
    print(f"🔍 正在对比文件:")
    print(f"   A: {file_a}")
    print(f"   B: {file_b}")
    print("-" * 60)

    data_a = load_json(file_a)
    data_b = load_json(file_b)

    if data_a is None or data_b is None:
        return

    # 1. 提取 structured_joints
    joints_a = data_a.get("structured_joints")
    joints_b = data_b.get("structured_joints")

    if joints_a is None or joints_b is None:
        print("❌ 错误: 某个文件中缺少 'structured_joints' 字段")
        return

    # 2. 结构层级对比 (笔画数 -> 段数 -> 点数)
    print("📋 正在检查数据结构一致性...")
    
    if len(joints_a) != len(joints_b):
        print(f"❌ 笔画数量不一致! A: {len(joints_a)}, B: {len(joints_b)}")
        return

    total_diff_sum = 0.0
    max_diff_val = 0.0
    max_diff_loc = "None"
    
    mismatch_count = 0
    total_points = 0

    # 遍历对比
    for i, (stroke_a, stroke_b) in enumerate(zip(joints_a, joints_b)):
        if len(stroke_a) != len(stroke_b):
            print(f"❌ 第 {i} 笔画的段数(Segment)不一致! A: {len(stroke_a)}, B: {len(stroke_b)}")
            return

        for j, (seg_a, seg_b) in enumerate(zip(stroke_a, stroke_b)):
            # seg_a 是一个点列表 [[j1, ... j6], [j1...j6]]
            np_a = np.array(seg_a)
            np_b = np.array(seg_b)

            if np_a.shape != np_b.shape:
                print(f"❌ 第 {i} 笔 - 第 {j} 段 点数或维度不一致!")
                print(f"   A shape: {np_a.shape}")
                print(f"   B shape: {np_b.shape}")
                return
            
            if np_a.size == 0: continue

            # 计算差值
            diff = np.abs(np_a - np_b)
            local_max = np.max(diff)
            
            # 累积统计
            total_points += len(np_a)
            
            if local_max > max_diff_val:
                max_diff_val = local_max
                max_diff_loc = f"Stroke {i}, Seg {j}"

            # 如果差异超过一个显眼的阈值 (比如 0.001 弧度)，打印出来
            if local_max > 1e-3:
                print(f"⚠️  差异发现: 第 {i} 笔, 第 {j} 段 | 最大偏差: {local_max:.6f} rad")
                mismatch_count += 1
                
            total_diff_sum += np.sum(diff)

    print("-" * 60)
    print("📊 对比结果摘要:")
    print(f"   - 对比总点数: {total_points} 点")
    
    # 3. 结论判定
    # 如果最大误差小于 1e-4 (0.005度)，通常认为是计算精度误差，实质是一样的
    threshold_safe = 1e-4 
    
    print(f"   - 全局最大偏差: {max_diff_val:.8f} radians")
    print(f"   - 偏差发生位置: {max_diff_loc}")

    if max_diff_val == 0.0:
        print("\n✅ 完美一致！两个文件数据完全相同。")
    elif max_diff_val < threshold_safe:
        print(f"\n✅ 基本一致 (偏差 < {threshold_safe})。")
        print("   微小的差异通常源于浮点数计算精度或并行计算顺序，可忽略。")
    else:
        print(f"\n❌ 数据不一致！存在显著差异 (> {threshold_safe})。")
        print("   这可能意味着算法逻辑不同，或者求解器收敛到了不同的逆解。")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    # 在这里修改你要对比的文件名
    FILE_1 = os.path.join(img_dir, "step07_joint_trajectory.json") 
    # 比如对比并行版和串行版的结果
    FILE_2 = os.path.join(img_dir, "step07_joint_trajectory_multicore.json") 
    
    # 如果你想对比同一个文件（自测），可以写一样的
    # FILE_2 = FILE_1 

    # 如果文件不存在，手动创建一个假的用于测试
    if not os.path.exists(FILE_2) and os.path.exists(FILE_1):
        print("⚠️ 提示: FILE_2 不存在，正在将 FILE_1 复制为 FILE_2 以进行自测演示...")
        import shutil
        shutil.copy(FILE_1, FILE_2)

    compare_structured_joints(FILE_1, FILE_2)