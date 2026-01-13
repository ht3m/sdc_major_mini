import json
import numpy as np
import time
import os
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ==================================================================================
# 1. 机器人运动学定义
# ==================================================================================
DH_PARAMS = [
    [0,            0,      187.0,  0],          # J1
    [np.pi/2,      0,      6.0,    np.pi/2],    # J2
    [0,            210.0,  0,      -np.pi/2],   # J3
    [-np.pi/2,     0,      210.5,  0],          # J4
    [np.pi/2,      0,      0,      0],          # J5
    [-np.pi/2,     0,      159.3,  0]           # J6
]

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

# 🔥🔥🔥 关键修改 1：加入 last_joints 参数实现“连续性惩罚” 🔥🔥🔥
def ik_error_func(q, target_pos, target_rot_matrix, last_joints=None):
    T = forward_kinematics(q)
    pos_err = np.linalg.norm(T[:3, 3] - target_pos)
    rot_err = np.linalg.norm(T[:3, :3] - target_rot_matrix)
    
    # 软限位惩罚
    limit_penalty = 0
    for i, val in enumerate(q):
        if val < BOUNDS[i][0] or val > BOUNDS[i][1]:
            limit_penalty += 1000 * (abs(val) - abs(BOUNDS[i][1]))**2

    # 连续性惩罚 (防止 180度 跳变)
    consistency_penalty = 0
    if last_joints is not None:
        diff = q - last_joints
        # 全局差异惩罚
        consistency_penalty = 0.5 * np.linalg.norm(diff)
        # J6 额外惩罚 (防止末端乱转)
        consistency_penalty += 0.5 * abs(diff[5]) 

    return pos_err * 1.0 + rot_err * 5.0 + limit_penalty + consistency_penalty

# ==================================================================================
# 2. Worker 函数 (修改了种子和调用方式)
# ==================================================================================
def process_stroke_wrapper(args):
    segments, quat = args
    stroke_joints = []
    fails = 0
    count = 0
    last_sol = None 
    
    target_rot = R.from_quat(quat).as_matrix()
    
    # 🔥🔥🔥 关键修改 2：全向种子 (Omni-directional Seeds) 🔥🔥🔥
    # 无论你的图纸旋转到了哪里 (前/后/左/右)，这里都有适合的初始姿态
    seeds = [
        # 前方 (X+)
        np.array([0, 0, np.pi/2, 0, np.pi/2, 0]), 
        # 左侧 (Y+) - 对应旋转 90度
        np.array([np.pi/2, 0, np.pi/2, 0, np.pi/2, 0]),
        # 右侧 (Y-) - 对应旋转 -90度
        np.array([-np.pi/2, 0, np.pi/2, 0, np.pi/2, 0]),
        # 后方 (X-) - 对应旋转 180度
        np.array([np.pi, 0, np.pi/2, 0, np.pi/2, 0]),
        # 下探姿态 (通用)
        np.array([0, 0.5, 1.5, 0, 1.0, 0]), 
    ]

    for points in segments: 
        segment_joints = []
        for pt in points:
            target_pos = np.array(pt)
            sol = None
            
            # 1. Tracking 模式 (传入 last_sol 以利用连续性惩罚)
            if last_sol is not None:
                # 注意：这里把 last_sol 传给了 ik_error_func 的第3个参数
                res = minimize(ik_error_func, last_sol, args=(target_pos, target_rot, last_sol),
                               method='SLSQP', bounds=BOUNDS, tol=1e-4)
                if res.fun < 2.0:
                    sol = res.x
            
            # 2. Global Search 模式 (Tracking 失败时启用)
            if sol is None:
                best_err = float('inf')
                for s in seeds:
                    # 全局搜索时不限制 last_joints (传 None)，允许它跳到最优解
                    res = minimize(ik_error_func, s, args=(target_pos, target_rot, None),
                                   method='SLSQP', bounds=BOUNDS, tol=1e-4)
                    if res.fun < best_err and res.fun < 1.0:
                        best_err = res.fun
                        sol = res.x

            if sol is not None:
                # J6 翻转保护 (数学修正)
                if last_sol is not None:
                    diff_j6 = sol[5] - last_sol[5]
                    if diff_j6 > np.pi: sol[5] -= 2*np.pi
                    elif diff_j6 < -np.pi: sol[5] += 2*np.pi
                
                segment_joints.append(sol.tolist())
                last_sol = sol
            else:
                fails += 1
                if last_sol is not None:
                    segment_joints.append(last_sol.tolist())
                else:
                    segment_joints.append([0.0]*6)
            
            count += 1
        stroke_joints.append(segment_joints)
        
    return {'joints': stroke_joints, 'fails': fails, 'count': count}

# ==================================================================================
# 3. 辅助工具 & 主程序
# ==================================================================================
def get_target_quat(angle_deg):
    # 90 = 笔尖朝前 (X+)
    # 180 = 笔尖朝下 (Z-)
    print(f"📐 目标姿态: 绕 Y 轴旋转 {angle_deg}°")
    r = R.from_euler('y', angle_deg, degrees=True)
    return r.as_quat()

def save_json_indented(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, separators=(',', ':'))
    print(f"📁 数据已保存: {filename}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    INPUT_FILE = os.path.join(img_dir, "step06_full_path.json")
    OUTPUT_FILE = os.path.join(img_dir, "step07_joint_trajectory_multicore.json")

    # 🔥 设置：笔尖垂直向下 (最通用，不易撞限位)
    TARGET_PITCH = 180.0 
    target_quat = get_target_quat(TARGET_PITCH)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件 {INPUT_FILE}")
        exit()

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
        structured_path = data.get("structured_path", []) 
        meta = data.get("meta", {})

    print(f"🚀 开始并行计算 (核心数: {multiprocessing.cpu_count()})...")
    
    stroke_tasks = [(stroke, target_quat) for stroke in structured_path]
    
    start_time = time.time()
    final_structured_joints = [None] * len(structured_path)
    total_processed = 0
    fail_count = 0

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_stroke_wrapper, stroke_tasks)
        for i, res in enumerate(results):
            final_structured_joints[i] = res['joints']
            fail_count += res['fails']
            total_processed += res['count']
            if (i+1) % 10 == 0:
                print(f"\r   - 进度: {i+1}/{len(stroke_tasks)}", end="")

    total_time = time.time() - start_time
    print(f"\n✅ 完成! 耗时: {total_time:.2f}s | 速度: {total_processed/total_time:.1f} pts/s")
    
    if fail_count > 0:
        print(f"⚠️ 警告: {fail_count} 个点失败")

    output_data = {
        "meta": {
            "source": "step07_optimized_seeds",
            "unit": "radians",
            "target_pitch": TARGET_PITCH,
            "original_meta": meta
        },
        "structured_joints": final_structured_joints
    }
    save_json_indented(OUTPUT_FILE, output_data)