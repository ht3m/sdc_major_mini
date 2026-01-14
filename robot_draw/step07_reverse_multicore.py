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

def ik_error_func(q, target_pos, target_rot_matrix, last_joints=None):
    T = forward_kinematics(q)
    pos_err = np.linalg.norm(T[:3, 3] - target_pos)
    rot_err = np.linalg.norm(T[:3, :3] - target_rot_matrix)
    
    # 软限位惩罚
    limit_penalty = 0
    for i, val in enumerate(q):
        if val < BOUNDS[i][0] or val > BOUNDS[i][1]:
            limit_penalty += 1000 * (abs(val) - abs(BOUNDS[i][1]))**2

    # 连续性惩罚
    consistency_penalty = 0
    if last_joints is not None:
        diff = q - last_joints
        consistency_penalty = 0.5 * np.linalg.norm(diff)
        consistency_penalty += 0.5 * abs(diff[5]) 

    return pos_err * 1.0 + rot_err * 5.0 + limit_penalty + consistency_penalty

# ==================================================================================
# 2. Worker 函数 (修改：记录失败坐标)
# ==================================================================================
def process_stroke_wrapper(args):
    segments, quat = args
    stroke_joints = []
    fails = 0
    count = 0
    last_sol = None 
    
    # 🔥 新增：用于记录失败点的列表
    failed_coords = []
    
    target_rot = R.from_quat(quat).as_matrix()
    
    seeds = [
        np.array([0, 0, np.pi/2, 0, np.pi/2, 0]), 
        np.array([np.pi/2, 0, np.pi/2, 0, np.pi/2, 0]),
        np.array([-np.pi/2, 0, np.pi/2, 0, np.pi/2, 0]),
        np.array([np.pi, 0, np.pi/2, 0, np.pi/2, 0]),
        np.array([0, 0.5, 1.5, 0, 1.0, 0]), 
    ]

    for points in segments: 
        segment_joints = []
        for pt in points:
            target_pos = np.array(pt)
            sol = None
            
            # 1. Tracking
            if last_sol is not None:
                res = minimize(ik_error_func, last_sol, args=(target_pos, target_rot, last_sol),
                               method='SLSQP', bounds=BOUNDS, tol=1e-4)
                if res.fun < 2.0:
                    sol = res.x
            
            # 2. Global Search
            if sol is None:
                best_err = float('inf')
                for s in seeds:
                    res = minimize(ik_error_func, s, args=(target_pos, target_rot, None),
                                   method='SLSQP', bounds=BOUNDS, tol=1e-4)
                    if res.fun < best_err and res.fun < 1.0:
                        best_err = res.fun
                        sol = res.x

            if sol is not None:
                if last_sol is not None:
                    diff_j6 = sol[5] - last_sol[5]
                    if diff_j6 > np.pi: sol[5] -= 2*np.pi
                    elif diff_j6 < -np.pi: sol[5] += 2*np.pi
                
                segment_joints.append(sol.tolist())
                last_sol = sol
            else:
                # 🔥🔥🔥 记录失败坐标 🔥🔥🔥
                fails += 1
                failed_coords.append(pt) # 记录当前的 [x, y, z]
                
                # 补救措施
                if last_sol is not None:
                    segment_joints.append(last_sol.tolist())
                else:
                    segment_joints.append([0.0]*6)
            
            count += 1
        stroke_joints.append(segment_joints)
        
    # 返回字典中增加 failed_coords
    return {'joints': stroke_joints, 'fails': fails, 'count': count, 'failed_coords': failed_coords}

# ==================================================================================
# 3. 辅助工具 & 主程序
# ==================================================================================
def get_target_quat(angle_deg):
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

    # 垂直向下姿态
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
            
            # 🔥🔥🔥 打印具体的失败坐标 🔥🔥🔥
            if res['fails'] > 0:
                print(f"\n❌ [Error] 笔画 {i+1} 出现 {res['fails']} 个解算失败点:")
                for fail_pt in res['failed_coords']:
                    # 格式化打印坐标，方便阅读
                    print(f"   -> 坐标 (XYZ): [{fail_pt[0]:.2f}, {fail_pt[1]:.2f}, {fail_pt[2]:.2f}]")
            
            if (i+1) % 10 == 0:
                print(f"\r   - 进度: {i+1}/{len(stroke_tasks)}", end="")

    total_time = time.time() - start_time
    print(f"\n\n✅ 完成! 耗时: {total_time:.2f}s | 速度: {total_processed/total_time:.1f} pts/s")
    
    if fail_count > 0:
        print(f"⚠️ 总计警告: {fail_count} 个点解算失败，请检查上述坐标是否超出机械臂工作空间。")

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