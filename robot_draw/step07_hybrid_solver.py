import json
import numpy as np
import time
import os
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline

# ==================================================================================
# 1. 笛卡尔空间规划器 (Cartesian Planner)
# ==================================================================================
class CartesianPlanner:
    def __init__(self, frequency=125.0):
        self.dt = 1.0 / frequency
        
        # 速度配置 (单位: mm/s)
        # 这里的速度是真实的笔尖线速度
        self.SPEED_DRAW = 20.0   # 写字: 20mm/s (慢速，高精度)
        self.SPEED_AIR  = 80.0   # 空移: 80mm/s (快速)
        
        # 停顿配置 (单位: 秒)
        self.WAIT_START = 1.0
        self.WAIT_CONN  = 1.0    # 动作连接处 (Move->Drop->Draw->Lift)
        self.WAIT_END   = 1.0

    def _get_cumulative_dist(self, points):
        """计算路径点的累积弧长"""
        diffs = np.diff(points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        # 累积距离: [0, d1, d1+d2, ...]
        cum_dist = np.concatenate(([0], np.cumsum(dists)))
        return cum_dist

    def interpolate_segment(self, points, speed_mm_s, kind='cubic'):
        """
        对一段 3D 路径进行时域重采样
        points: List[[x,y,z]] 关键点
        speed_mm_s: 期望线速度
        kind: 'linear' (直线) 或 'cubic' (平滑曲线)
        """
        points = np.array(points)
        if len(points) < 2:
            return [points[0]]

        # 1. 计算总路程 (mm)
        cum_dist = self._get_cumulative_dist(points)
        total_dist = cum_dist[-1]
        
        if total_dist < 1e-3:
            return [points[0]]

        # 2. 计算总耗时 (s)
        total_time = total_dist / speed_mm_s
        # 至少给 1 个周期，防止除零
        if total_time < self.dt: total_time = self.dt

        # 3. 生成目标时间轴 (125Hz)
        num_frames = int(np.ceil(total_time / self.dt))
        t_target = np.linspace(0, total_dist, num_frames)
        
        # 4. 空间插值 (XYZ)
        # 使用累积距离作为自变量 x，坐标作为因变量 y
        # 这样可以保证沿着路径匀速采样
        
        # 只有点数够多才能用 cubic，否则降级为 linear
        if kind == 'cubic' and len(points) >= 4:
            cs = CubicSpline(cum_dist, points, axis=0, bc_type='natural')
            dense_points = cs(t_target)
        else:
            # 线性插值 (空移或者短线段)
            # 手动实现简单的线性插值
            dense_points = np.zeros((num_frames, 3))
            for i in range(3): # x, y, z
                dense_points[:, i] = np.interp(t_target, cum_dist, points[:, i])
                
        return dense_points.tolist()

    def generate_wait_frames(self, pos, duration):
        """生成停顿帧 (XYZ保持不变)"""
        frames = int(duration / self.dt)
        return [pos] * frames

# ==================================================================================
# 2. 机器人运动学求解器 (IK Solver)
# ==================================================================================
class JakaRobotIK:
    def __init__(self):
        # DH 参数 [alpha, a, d, theta_offset]
        self.DH_PARAMS = [
            [0,            0,      187.0,  0],
            [np.pi/2,      0,      6.0,    np.pi/2],
            [0,            210.0,  0,      -np.pi/2],
            [-np.pi/2,     0,      210.5,  0],
            [np.pi/2,      0,      0,      0],
            [-np.pi/2,     0,      159.3,  0]
        ]
        large = 2 * np.pi
        small = (2 * np.pi) * (120/360)
        self.bounds = [
            (-large, large), (-small, small), (-small, small),
            (-large, large), (-small, small), (-large, large)
        ]

    def dh_matrix(self, alpha, a, d, theta):
        c, s = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [c,    -s,    0,   a],
            [s*ca, c*ca, -sa, -d*sa],
            [s*sa, c*sa,  ca,  d*ca],
            [0,     0,    0,   1]
        ])

    def forward_kinematics(self, joints):
        T = np.eye(4)
        for i, (alpha, a, d, offset) in enumerate(self.DH_PARAMS):
            theta = joints[i] + offset
            T = T @ self.dh_matrix(alpha, a, d, theta)
        return T

    def _error_func(self, q, target_pos, target_rot_matrix):
        T = self.forward_kinematics(q)
        pos_err = np.linalg.norm(T[:3, 3] - target_pos)
        rot_err = np.linalg.norm(T[:3, :3] - target_rot_matrix)
        
        # 惩罚项
        penalty = 0
        for i, val in enumerate(q):
            if val < self.bounds[i][0] or val > self.bounds[i][1]:
                penalty += 1000 * (abs(val) - abs(self.bounds[i][1]))**2
        
        return pos_err * 1.0 + rot_err * 2.0 + penalty

    def solve_ik_init(self, target_pos, target_quat):
        """全局搜索 (首点)"""
        t_pos = np.array(target_pos)
        t_rot = R.from_quat(target_quat).as_matrix()
        
        seeds = [
            np.array([0, 0, np.pi/2, 0, np.pi/2, 0]), 
            np.array([0, 0.5, 1.5, 0, 1.0, 0]), 
            np.array([0, -0.5, -1.5, 0, -1.0, 0])
        ]
        
        best_sol = None
        min_error = float('inf')
        
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
        """局部追踪 (极速)"""
        t_pos = np.array(target_pos)
        t_rot = R.from_quat(target_quat).as_matrix()
        
        res = minimize(
            self._error_func, last_joints, args=(t_pos, t_rot),
            method='SLSQP', bounds=self.bounds, tol=1e-4
        )
        if res.fun < 2.0: return res.x
        return None

# ==================================================================================
# 3. 主程序：混合求解 (Cartesian Opt -> IK)
# ==================================================================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    # 1. 读取 Step 06 结构化路径
    INPUT_FILE = os.path.join(img_dir, "step06_full_path.json")
    # 2. 直接输出最终结果 (不需要 Step 08 了)
    OUTPUT_FILE = os.path.join(img_dir, "step07_more_optimized_trajectory.json")

    # 垂直向下姿态
    TARGET_QUAT = [0.0, 1.0, 0.0, 0.0] 

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到 {INPUT_FILE}")
        exit()

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
        structured_path = data.get("structured_path", []) 
        meta = data.get("meta", {})

    print(f"🚀 开始混合求解 (笛卡尔优化 + 逆解)...")
    print(f"   - 笔画数: {len(structured_path)}")
    
    planner = CartesianPlanner()
    robot = JakaRobotIK()
    
    final_joint_trajectory = []
    
    # 状态变量
    last_ik_sol = None
    
    # --- 1. 全局开始停顿 ---
    # 获取第一个点
    start_pos = structured_path[0][0][0]
    
    # 计算第一个点的 IK
    print("📍 计算起点 IK...")
    start_joints = robot.solve_ik_init(start_pos, TARGET_QUAT)
    if start_joints is None:
        print("❌ 起点无解！")
        exit()
    last_ik_sol = start_joints
    
    # 插入 Start Wait 帧
    start_wait_frames = int(planner.WAIT_START / planner.dt)
    final_joint_trajectory.extend([start_joints.tolist()] * start_wait_frames)
    
    total_strokes = len(structured_path)
    
    # --- 2. 循环处理笔画 ---
    for s_idx, segments in enumerate(structured_path):
        print(f"\r处理笔画: {s_idx+1}/{total_strokes}", end="")
        
        # segments: 0=Move, 1=Drop, 2=Draw, 3=Lift
        for seg_idx, points in enumerate(segments):
            # A. 笛卡尔空间优化 (生成密集的 XYZ)
            # ------------------------------------------------
            # 根据阶段选择速度和插值方式
            if seg_idx == 2: 
                # Draw: 慢速, Cubic Spline (平滑拐角)
                speed = planner.SPEED_DRAW
                kind = 'cubic'
            else:
                # Move/Drop/Lift: 快速, Linear (效率)
                speed = planner.SPEED_AIR
                kind = 'linear' # 空移不需要圆角，直达即可
            
            # 生成这一段的密集点 (Cartesian Dense Path)
            dense_xyz_path = planner.interpolate_segment(points, speed, kind)
            
            # B. 逆解算 (IK)
            # ------------------------------------------------
            for xyz in dense_xyz_path:
                # 每一毫米都解一次，因为点很密，tracking 极快且准
                sol = robot.solve_ik_tracking(xyz, TARGET_QUAT, last_ik_sol)
                
                if sol is None:
                    # 极罕见情况：如果 tracking 丢了，用 init 找回
                    sol = robot.solve_ik_init(xyz, TARGET_QUAT)
                
                if sol is not None:
                    final_joint_trajectory.append(sol.tolist())
                    last_ik_sol = sol
                else:
                    # 容错：沿用上一帧，防止崩溃
                    final_joint_trajectory.append(last_ik_sol.tolist())
            
            # C. 插入停顿 (Wait)
            # ------------------------------------------------
            # 每一段结束后都插入停顿
            wait_frames = int(planner.WAIT_CONN / planner.dt)
            if wait_frames > 0 and last_ik_sol is not None:
                final_joint_trajectory.extend([last_ik_sol.tolist()] * wait_frames)

    # --- 3. 全局结束停顿 ---
    end_wait_frames = int(planner.WAIT_END / planner.dt)
    if last_ik_sol is not None:
        final_joint_trajectory.extend([last_ik_sol.tolist()] * end_wait_frames)

    print(f"\n✅ 计算完成!")
    
    # 保存结果
    output_data = {
        "meta": {
            "source": "step07_hybrid_solver",
            "unit": "radians",
            "count": len(final_joint_trajectory),
            "frequency": 125.0,
            "strategy": "cartesian_spline_then_ik",
            "speeds_mm_s": {"draw": planner.SPEED_DRAW, "air": planner.SPEED_AIR},
            "parent_meta": meta
        },
        "joints": final_joint_trajectory
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    print(f"💾 文件已保存: {OUTPUT_FILE}")