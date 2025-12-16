import json
import numpy as np
import time
import os
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

# ==================================================================================
# 1. 机器人运动学定义 (保持不变)
# ==================================================================================
class JakaRobot:
    def __init__(self):
        # DH 参数 [alpha, a, d, theta_offset]
        self.DH_PARAMS = [
            [0,            0,      187.0,  0],          # J1
            [np.pi/2,      0,      6.0,    np.pi/2],    # J2
            [0,            210.0,  0,      -np.pi/2],   # J3
            [-np.pi/2,     0,      210.5,  0],          # J4
            [np.pi/2,      0,      0,      0],          # J5
            [-np.pi/2,     0,      159.3,  0]           # J6
        ]

        # 关节限位 (弧度)
        large = 2 * np.pi
        small = (2 * np.pi) * (120/360) 
        self.bounds = [
            (-large, large), (-small, small), (-small, small),
            (-large, large), (-small, small), (-large, large)
        ]

    def dh_matrix(self, alpha, a, d, theta):
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

    def forward_kinematics(self, joints):
        T = np.eye(4)
        for i, (alpha, a, d, offset) in enumerate(self.DH_PARAMS):
            theta = joints[i] + offset
            Ti = self.dh_matrix(alpha, a, d, theta)
            T = T @ Ti
        return T

    def _error_func(self, q, target_pos, target_rot_matrix):
        T = self.forward_kinematics(q)
        curr_pos = T[:3, 3]
        curr_rot = T[:3, :3]
        pos_err = np.linalg.norm(curr_pos - target_pos)
        rot_err = np.linalg.norm(curr_rot - target_rot_matrix)
        penalty = 0
        for i, val in enumerate(q):
            if val < self.bounds[i][0] or val > self.bounds[i][1]:
                penalty += 1000 * (abs(val) - abs(self.bounds[i][1]))**2
        return pos_err * 1.0 + rot_err * 5.0 + penalty

    def solve_ik_init(self, target_pos, target_quat):
        t_pos = np.array(target_pos)
        r = R.from_quat(target_quat)
        t_rot = r.as_matrix()
        seeds = [
            np.array([0, 0, np.pi/2, 0, np.pi/2, 0]), 
            np.array([0, 0.5, 1.5, 0, 1.0, 0]), 
            np.array([0, -0.5, -1.5, 0, -1.0, 0])
        ]
        best_sol = None
        min_error = float('inf')
        print("   🔍 正在为起点进行全局构型搜索...")
        for seed in seeds:
            res = minimize(
                self._error_func, seed, args=(t_pos, t_rot),
                method='SLSQP', bounds=self.bounds, tol=1e-4
            )
            if res.fun < min_error and res.fun < 1.0:
                min_error = res.fun
                best_sol = res.x
        return best_sol

    def solve_ik_tracking(self, target_pos, target_quat, last_joints):
        t_pos = np.array(target_pos)
        r = R.from_quat(target_quat)
        t_rot = r.as_matrix()
        res = minimize(
            self._error_func, last_joints, args=(t_pos, t_rot),
            method='SLSQP', bounds=self.bounds, tol=1e-4
        )
        if res.fun < 2.0:
            return res.x
        else:
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
        # 使用 indent=None 减小体积，但结构依然是分层的
        json.dump(data, f, indent=None, separators=(',', ':'))
    print(f"📁 数据已保存: {filename}")

# ==================================================================================
# 3. 主程序
# ==================================================================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    # 输入 Step 06 的结构化路径
    INPUT_FILE = os.path.join(img_dir, "step06_full_path.json")
    # 输出 Step 07 的结构化关节数据
    OUTPUT_FILE = os.path.join(img_dir, "step07_joint_trajectory.json")

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

    total_points_flat = meta.get("total_points_flat", 0)
    print(f"🚀 开始计算逆解 (保持结构化)...")
    print(f"   - 总笔画数: {len(structured_path)}")
    print(f"   - 结构模式: [Move, Drop, Draw, Lift]")
    
    robot = JakaRobot()
    
    # 🔥 核心数据结构：三层嵌套列表
    # structured_joints[stroke_idx][segment_idx][point_idx] = [j1...j6]
    structured_joints = [] 
    
    start_time = time.time()
    last_sol = None # 全局状态，保证跨段连续
    fail_count = 0
    processed_count = 0

    # 1. 遍历每一笔
    for stroke_idx, segments in enumerate(structured_path):
        current_stroke_joints = []
        
        # 2. 遍历该笔画的 4 个阶段
        for seg_idx, points in enumerate(segments):
            current_segment_joints = []
            
            # 3. 遍历阶段内的每个点
            for pt_idx, pt in enumerate(points):
                target_pos = np.array(pt)
                sol = None
                
                if last_sol is None:
                    # 全局第一个点
                    sol = robot.solve_ik_init(target_pos, target_quat)
                    if sol is None:
                        print("❌ 致命错误：起点无法到达！")
                        exit()
                else:
                    # 追踪模式
                    sol = robot.solve_ik_tracking(target_pos, target_quat, last_sol)
                    # 救急策略
                    if sol is None:
                        sol = robot.solve_ik_init(target_pos, target_quat)

                if sol is not None:
                    current_segment_joints.append(sol.tolist())
                    last_sol = sol # 更新全局种子
                else:
                    fail_count += 1
                    # 沿用上一点
                    if last_sol is not None:
                        current_segment_joints.append(last_sol.tolist())
                
                processed_count += 1
                if processed_count % 100 == 0:
                     print(f"\r处理进度: {processed_count}/{total_points_flat}", end="")
            
            # 将该段结果存入当前笔画
            current_stroke_joints.append(current_segment_joints)
        
        # 将该笔画存入总结果
        structured_joints.append(current_stroke_joints)

    total_time = time.time() - start_time
    print(f"\n\n✅ 计算完成!")
    print(f"   - 耗时: {total_time:.2f} 秒")
    
    if fail_count > 0:
        print(f"⚠️ 警告: {fail_count} 个点解算失败")

    # 构造输出数据
    output_data = {
        "meta": {
            "source": "step07_inverse_kinematics",
            "unit": "radians",
            "structure_format": "[N_strokes, 4_segments, M_points, 6_joints]",
            "segment_meaning": ["0:AirMove", "1:Drop", "2:Draw", "3:Lift"],
            "total_count_flat": processed_count,
            "original_meta": meta
        },
        # 🔥 输出结构化数据，而非扁平数据
        "structured_joints": structured_joints
    }

    save_json_indented(OUTPUT_FILE, output_data)