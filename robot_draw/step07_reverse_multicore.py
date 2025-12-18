import json
import numpy as np
import time
import os
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ==================================================================================
# 1. 机器人运动学定义 (独立函数，方便序列化)
# ==================================================================================
# DH 参数 [alpha, a, d, theta_offset]
DH_PARAMS = [
    [0,            0,      187.0,  0],          # J1
    [np.pi/2,      0,      6.0,    np.pi/2],    # J2
    [0,            210.0,  0,      -np.pi/2],   # J3
    [-np.pi/2,     0,      210.5,  0],          # J4
    [np.pi/2,      0,      0,      0],          # J5
    [-np.pi/2,     0,      159.3,  0]           # J6
]

# 关节限位
LARGE = 2 * np.pi
SMALL = (2 * np.pi) * (120/360) 
BOUNDS = [
    (-LARGE, LARGE), (-SMALL, SMALL), (-SMALL, SMALL),
    (-LARGE, LARGE), (-SMALL, SMALL), (-LARGE, LARGE)
]

def dh_matrix(alpha, a, d, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([
        [c,      -s,      0,      a],
        [s*ca,   c*ca,   -sa,    -d*sa],
        [s*sa,   c*sa,    ca,     d*ca],
        [0,       0,      0,      1]
    ])

def forward_kinematics(joints):
    T = np.eye(4)
    for i, (alpha, a, d, offset) in enumerate(DH_PARAMS):
        theta = joints[i] + offset
        Ti = dh_matrix(alpha, a, d, theta)
        T = T @ Ti
    return T

def ik_error_func(q, target_pos, target_rot_matrix):
    T = forward_kinematics(q)
    pos_err = np.linalg.norm(T[:3, 3] - target_pos)
    rot_err = np.linalg.norm(T[:3, :3] - target_rot_matrix)
    penalty = 0
    for i, val in enumerate(q):
        if val < BOUNDS[i][0] or val > BOUNDS[i][1]:
            penalty += 1000 * (abs(val) - abs(BOUNDS[i][1]))**2
    return pos_err * 1.0 + rot_err * 5.0 + penalty

def solve_single_point(args):
    """
    单点解算函数 (Worker Process)
    args: (target_pos_list, target_quat_list, last_joints_list)
    """
    pt, quat, seed_guess = args
    
    target_pos = np.array(pt)
    target_rot = R.from_quat(quat).as_matrix()
    
    # 策略：优先用上一个解作为种子 (Tracking)，如果失败则全局搜索
    if seed_guess is not None:
        res = minimize(ik_error_func, seed_guess, args=(target_pos, target_rot),
                       method='SLSQP', bounds=BOUNDS, tol=1e-4)
        if res.fun < 2.0:
            return res.x.tolist()

    # 全局搜索种子
    seeds = [
        np.array([0, 0, np.pi/2, 0, np.pi/2, 0]), 
        np.array([0, 0.5, 1.5, 0, 1.0, 0]), 
        np.array([0, -0.5, -1.5, 0, -1.0, 0])
    ]
    
    best_sol = None
    min_error = float('inf')
    
    for s in seeds:
        res = minimize(ik_error_func, s, args=(target_pos, target_rot),
                       method='SLSQP', bounds=BOUNDS, tol=1e-4)
        if res.fun < min_error and res.fun < 1.0:
            min_error = res.fun
            best_sol = res.x
            
    if best_sol is not None:
        return best_sol.tolist()
    return None

# ==================================================================================
# 2. 辅助工具
# ==================================================================================
def get_vertical_quat(pitch_deg=90):
    if abs(pitch_deg - 90) < 0.1:
        return [0.0, 1.0, 0.0, 0.0]
    else:
        rot_y = R.from_euler('y', 180 - (90 - pitch_deg), degrees=True)
        return rot_y.as_quat()

def save_json_indented(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, separators=(',', ':'))
    print(f"📁 数据已保存: {filename}")

def process_stroke_wrapper(args):
    """
    处理单个笔画的完整流程
    args: (stroke_segments, target_quat)
    return: {'joints': nested_list, 'fails': int, 'count': int}
    """
    segments, quat = args
    
    stroke_joints = []
    fails = 0
    count = 0
    last_sol = None # 笔画内部保持连续性
    
    for points in segments: # 遍历 Move, Drop, Draw, Lift
        segment_joints = []
        for pt in points:
            # 构造参数传给单点求解器，或者直接在这里调用 minimize
            # 为了减少进程间通信开销，直接在这里解算最快
            
            target_pos = np.array(pt)
            target_rot = R.from_quat(quat).as_matrix()
            
            sol = None
            
            # 1. 尝试 Tracking (极快)
            if last_sol is not None:
                res = minimize(ik_error_func, last_sol, args=(target_pos, target_rot),
                               method='SLSQP', bounds=BOUNDS, tol=1e-4)
                if res.fun < 2.0:
                    sol = res.x
            
            # 2. 如果失败或无前值，尝试全局搜索 (慢)
            if sol is None:
                seeds = [
                    np.array([0, 0, np.pi/2, 0, np.pi/2, 0]), 
                    np.array([0, 0.5, 1.5, 0, 1.0, 0]), 
                    np.array([0, -0.5, -1.5, 0, -1.0, 0])
                ]
                best_err = float('inf')
                for s in seeds:
                    res = minimize(ik_error_func, s, args=(target_pos, target_rot),
                                   method='SLSQP', bounds=BOUNDS, tol=1e-4)
                    if res.fun < best_err and res.fun < 1.0:
                        best_err = res.fun
                        sol = res.x

            # 3. 保存结果
            if sol is not None:
                segment_joints.append(sol.tolist())
                last_sol = sol
            else:
                fails += 1
                # 补救：沿用上一个点防止空缺
                if last_sol is not None:
                    segment_joints.append(last_sol.tolist())
            
            count += 1
            
        stroke_joints.append(segment_joints)
        
    return {'joints': stroke_joints, 'fails': fails, 'count': count}

# ==================================================================================
# 3. 主程序
# ==================================================================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    INPUT_FILE = os.path.join(img_dir, "step06_full_path.json")
    OUTPUT_FILE = os.path.join(img_dir, "step07_joint_trajectory_multicore.json")

    # 垂直向下姿态
    TARGET_PITCH = 90.0 
    target_quat = get_vertical_quat(TARGET_PITCH)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到文件 {INPUT_FILE}")
        exit()

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
        structured_path = data.get("structured_path", []) 
        meta = data.get("meta", {})

    print(f"🚀 开始并行计算逆解 (利用多核性能)...")
    print(f"   - CPU 核心数: {multiprocessing.cpu_count()}")
    
    # --- 1. 数据扁平化 (Flattening) ---
    # 为了并行计算，我们需要把复杂的嵌套结构打平成一个大列表
    # 同时记录索引以便还原
    # tasks = [(stroke_idx, seg_idx, point_idx, point_coord)]
    
    flat_tasks = []
    
    # 预计算：为了保证连续性，我们采用“分块并行”策略不太容易实现严格的连续追踪
    # 但由于我们的点非常密集，且优化器鲁棒，我们可以尝试让每个点都带上一个“推测的种子”
    # 或者简单粗暴地并行解每一个点（虽然牺牲了 Tracking 速度优势，但多核算力能弥补回来）
    
    # 更好的策略：按【笔画】(Stroke) 并行
    # 一个笔画内的点是连续的，交给一个核心去跑，能利用 Tracking 加速
    # 不同的笔画交给不同的核心
    
    stroke_tasks = []
    for stroke in structured_path:
        stroke_tasks.append((stroke, target_quat))

    start_time = time.time()
    
    # --- 2. 并行处理 (ProcessPool) ---
    # 核心逻辑：定义一个处理单笔画的函数
    
    final_structured_joints = [None] * len(structured_path)
    total_processed = 0
    fail_count = 0

    # 我们需要把 process_stroke 函数移到这里或作为独立函数
    # 由于 Python multiprocessing 的 pickle 限制，最好定义在 top-level
    
    print(f"   - 正在分发任务 (共 {len(stroke_tasks)} 笔)...")
    
    with ProcessPoolExecutor() as executor:
        # map 会按顺序返回结果
        results = executor.map(process_stroke_wrapper, stroke_tasks)
        
        for i, res in enumerate(results):
            final_structured_joints[i] = res['joints']
            fail_count += res['fails']
            total_processed += res['count']
            if i % 10 == 0:
                print(f"\r   - 已完成笔画: {i+1}/{len(stroke_tasks)}", end="")

    total_time = time.time() - start_time
    print(f"\n\n✅ 计算完成!")
    print(f"   - 耗时: {total_time:.2f} 秒")
    print(f"   - 平均速度: {total_processed / total_time:.1f} points/sec")
    
    if fail_count > 0:
        print(f"⚠️ 警告: {fail_count} 个点解算失败")

    output_data = {
        "meta": {
            "source": "step07_inverse_kinematics_parallel",
            "unit": "radians",
            "total_count_flat": total_processed,
            "original_meta": meta
        },
        "structured_joints": final_structured_joints
    }

    save_json_indented(OUTPUT_FILE, output_data)


