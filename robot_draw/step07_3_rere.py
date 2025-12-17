import json
import numpy as np
import os
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

# ==================================================================================
# 1. 笛卡尔空间规划器 (生成 125Hz 密集流)
# ==================================================================================
class DensePlanner:
    def __init__(self, frequency=125.0):
        self.dt = 1.0 / frequency # 8ms
        
        # 🚀 速度配置 (已加速 1.5 倍)
        self.SPEED_DRAW = 30.0   # 写字: 30.0 mm/s
        self.SPEED_AIR  = 60.0   # 空移: 90.0 mm/s
        
        # 停顿配置 (保持 1.0s 以便观察)
        self.WAIT_CONN  = 1.0    
        self.WAIT_START = 1.0

    def _get_cumulative_dist(self, points):
        """计算路径点的累积弧长"""
        diffs = np.diff(points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        return np.concatenate(([0], np.cumsum(dists)))

    def interpolate_segment(self, points, speed_mm_s, kind='cubic'):
        """对一段几何路径进行 125Hz 重采样"""
        points = np.array(points)
        if len(points) < 2: return [points[0]]

        # 1. 计算路程 (mm)
        cum_dist = self._get_cumulative_dist(points)
        total_dist = cum_dist[-1]
        
        if total_dist < 1e-3: return [points[0]]

        # 2. 计算耗时 (s)
        total_time = total_dist / speed_mm_s
        # 至少给 2 个周期，防止过短
        if total_time < self.dt * 2: total_time = self.dt * 2

        # 3. 生成 125Hz 时间轴
        num_frames = int(np.ceil(total_time / self.dt))
        t_target = np.linspace(0, total_dist, num_frames)
        
        # 4. 空间插值 (XYZ)
        if kind == 'cubic' and len(points) >= 4:
            try:
                cs = CubicSpline(cum_dist, points, axis=0, bc_type='natural')
                dense_points = cs(t_target)
            except:
                dense_points = np.zeros((num_frames, 3))
                for i in range(3): 
                    dense_points[:, i] = np.interp(t_target, cum_dist, points[:, i])
        else:
            dense_points = np.zeros((num_frames, 3))
            for i in range(3): 
                dense_points[:, i] = np.interp(t_target, cum_dist, points[:, i])
                
        return dense_points.tolist()

# ==================================================================================
# 2. IK Solver (保持不变)
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
        self.bounds = [(-2*np.pi, 2*np.pi)] * 6

    def forward_kinematics(self, joints):
        T = np.eye(4)
        for i, (alpha, a, d, offset) in enumerate(self.DH_PARAMS):
            theta = joints[i] + offset
            c, s = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)
            Ti = np.array([
                [c,    -s,    0,   a],
                [s*ca, c*ca, -sa, -d*sa],
                [s*sa, c*sa,  ca,  d*ca],
                [0,     0,    0,   1]
            ])
            T = T @ Ti
        return T

    def _error_func(self, q, target_pos, target_rot):
        T = self.forward_kinematics(q)
        pos_err = np.linalg.norm(T[:3, 3] - target_pos)
        rot_err = np.linalg.norm(T[:3, :3] - target_rot)
        return pos_err + rot_err * 2.0

    def solve_ik_init(self, target_pos, target_quat):
        t_pos = np.array(target_pos)
        t_rot = R.from_quat(target_quat).as_matrix()
        seed = np.array([0, 0, np.pi/2, 0, np.pi/2, 0])
        res = minimize(self._error_func, seed, args=(t_pos, t_rot), 
                       method='SLSQP', bounds=self.bounds, tol=1e-4)
        return res.x if res.fun < 1.0 else None

    def solve_ik_tracking(self, target_pos, target_quat, last_joints):
        t_pos = np.array(target_pos)
        t_rot = R.from_quat(target_quat).as_matrix()
        res = minimize(self._error_func, last_joints, args=(t_pos, t_rot), 
                       method='SLSQP', bounds=self.bounds, tol=1e-4)
        return res.x

# ==================================================================================
# 3. 主程序
# ==================================================================================
def apply_smoothing(joints, window=11):
    if len(joints) < window: return joints
    print(f"🧹 应用平滑处理 (Window={window})...")
    smoothed = np.zeros_like(joints)
    for i in range(6):
        smoothed[:, i] = savgol_filter(joints[:, i], window, 3, mode='nearest')
    return smoothed.tolist()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    INPUT_FILE = os.path.join(img_dir, "step06_full_path.json")
    OUTPUT_FILE = os.path.join(img_dir, "step07_re_optimized_trajectory.json")
    
    TARGET_QUAT = [0.0, 1.0, 0.0, 0.0]

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        exit()

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
        structured_path = data.get("structured_path", [])
        meta = data.get("meta", {})

    print(f"🚀 生成 125Hz 密集轨迹流 (加速 1.5x)...")
    
    planner = DensePlanner()
    robot = JakaRobotIK()
    final_traj = []
    
    # 1. 起点
    start_pos = structured_path[0][0][0]
    print("📍 计算起始点 IK...")
    last_ik = robot.solve_ik_init(start_pos, TARGET_QUAT)
    if last_ik is None:
        print("❌ 起始点无解")
        exit()
    
    # Start Wait
    start_wait_count = int(planner.WAIT_START / planner.dt)
    final_traj.extend([last_ik.tolist()] * start_wait_count)

    # 2. 循环生成
    total_strokes = len(structured_path)
    for s_idx, segments in enumerate(structured_path):
        print(f"\r处理笔画: {s_idx+1}/{total_strokes}", end="")
        for seg_idx, points in enumerate(segments):
            # A. 笛卡尔规划
            if seg_idx == 2: # Draw
                speed = planner.SPEED_DRAW
                kind = 'cubic'
            else: # Move/Drop/Lift
                speed = planner.SPEED_AIR
                kind = 'linear'
            
            dense_xyz = planner.interpolate_segment(points, speed, kind)
            
            # B. IK
            for xyz in dense_xyz:
                sol = robot.solve_ik_tracking(xyz, TARGET_QUAT, last_ik)
                final_traj.append(sol.tolist())
                last_ik = sol
            
            # C. 停顿
            wait_frames = int(planner.WAIT_CONN / planner.dt)
            final_traj.extend([last_ik.tolist()] * wait_frames)

    print(f"\n✅ 生成完毕! 总帧数: {len(final_traj)}")
    
    final_traj_np = np.array(final_traj)
    final_traj_smoothed = apply_smoothing(final_traj_np)
    
    output_data = {
        "meta": {
            "source": "step07_dense_solver_fast",
            "unit": "radians",
            "count": len(final_traj_smoothed),
            "frequency": 125.0,
            "speeds": {"draw": planner.SPEED_DRAW, "air": planner.SPEED_AIR},
            "parent_meta": meta
        },
        "joints": final_traj_smoothed
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=None)
    
    print(f"💾 文件已保存: {OUTPUT_FILE}")