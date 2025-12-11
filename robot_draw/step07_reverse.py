import json
import numpy as np
import time
import os
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

# ==================================================================================
# 1. 机器人运动学定义 (JAKA Zu/Mini 系列通用 DH)
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
        small = (2 * np.pi) * (120/360) # 限制 J2/J3/J5 范围，避免奇异
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
        """计算末端 T 矩阵"""
        T = np.eye(4)
        for i, (alpha, a, d, offset) in enumerate(self.DH_PARAMS):
            theta = joints[i] + offset
            Ti = self.dh_matrix(alpha, a, d, theta)
            T = T @ Ti
        return T

    def _error_func(self, q, target_pos, target_rot_matrix):
        """
        逆解优化的代价函数
        """
        # 1. 正解计算当前位姿
        T = self.forward_kinematics(q)
        curr_pos = T[:3, 3]
        curr_rot = T[:3, :3]
        
        # 2. 误差计算
        pos_err = np.linalg.norm(curr_pos - target_pos)
        rot_err = np.linalg.norm(curr_rot - target_rot_matrix)
        
        # 3. 关节越界惩罚 (软约束)
        penalty = 0
        for i, val in enumerate(q):
            if val < self.bounds[i][0] or val > self.bounds[i][1]:
                penalty += 1000 * (abs(val) - abs(self.bounds[i][1]))**2
        
        # 权重设置：位置最重要，姿态次之
        return pos_err * 1.0 + rot_err * 5.0 + penalty

    def solve_ik_init(self, target_pos, target_quat):
        """
        【全局搜索】用于路径的第一个点，寻找最佳初始姿态
        """
        t_pos = np.array(target_pos)
        r = R.from_quat(target_quat)
        t_rot = r.as_matrix()

        # 定义几个典型的种子姿态 (Elbow Up / Elbow Down / Neutral)
        seeds = [
            np.array([0, 0, np.pi/2, 0, np.pi/2, 0]),     # 经典肘部朝上
            np.array([0, 0.5, 1.5, 0, 1.0, 0]),           # 伸展态
            np.array([0, -0.5, -1.5, 0, -1.0, 0])         # 肘部朝下
        ]

        best_sol = None
        min_error = float('inf')

        print("   🔍 正在为起点进行全局构型搜索...")
        for seed in seeds:
            res = minimize(
                self._error_func, seed, args=(t_pos, t_rot),
                method='SLSQP', bounds=self.bounds, tol=1e-4
            )
            # 这里的阈值设为 1.0，只要位置准，姿态稍微歪一点点可以接受
            if res.fun < min_error and res.fun < 1.0:
                min_error = res.fun
                best_sol = res.x

        return best_sol

    def solve_ik_tracking(self, target_pos, target_quat, last_joints):
        """
        【路径追踪】利用上一个点的解作为种子，快速求解
        """
        t_pos = np.array(target_pos)
        r = R.from_quat(target_quat)
        t_rot = r.as_matrix()

        # 直接 minimize，以 last_joints 为起点
        res = minimize(
            self._error_func, last_joints, args=(t_pos, t_rot),
            method='SLSQP', bounds=self.bounds, tol=1e-4
        )
        
        if res.fun < 2.0: # 稍微放宽阈值，保证连续性
            return res.x
        else:
            return None

# ==================================================================================
# 2. 辅助工具
# ==================================================================================
def get_vertical_quat(pitch_deg=90):
    """
    获取笔尖姿态四元数
    pitch=90 -> 垂直向下 (标准写字)
    """
    if abs(pitch_deg - 90) < 0.1:
        return [0.0, 1.0, 0.0, 0.0] # 垂直向下
    else:
        # 如果需要倾斜，这里进行计算 (从垂直向下回退)
        rot_y = R.from_euler('y', 180 - (90 - pitch_deg), degrees=True)
        return rot_y.as_quat()

def save_json_indented(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))
    print(f"📁 数据已保存: {filename}")

# ==================================================================================
# 3. 主程序
# ==================================================================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    # 输入 Step 06 生成的连续路径
    INPUT_FILE = os.path.join(img_dir, "step06_full_path.json")
    # 输出 Step 07 的关节角度序列
    OUTPUT_FILE = os.path.join(img_dir, "step07_joint_trajectory.json")

    # 👉 【接口】设置握笔姿态
    # 90 = 垂直向下 (推荐)
    TARGET_PITCH = 90.0 
    target_quat = get_vertical_quat(TARGET_PITCH)

    # 1. 读取路径
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到文件 {INPUT_FILE}")
        exit()

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
        path_points = data.get("path_points", [])
        meta = data.get("meta", {})

    print(f"🚀 开始计算逆解...")
    print(f"   - 总点数: {len(path_points)}")
    print(f"   - 目标姿态 Pitch: {TARGET_PITCH}°")

    robot = JakaRobot()
    joint_trajectory = []
    
    # 计时开始
    start_time = time.time()
    
    # 状态变量
    last_sol = None
    fail_count = 0

    for i, pt in enumerate(path_points):
        # pt 是 [x, y, z]
        
        sol = None
        
        if i == 0:
            # --- 第一点：全局搜索 ---
            sol = robot.solve_ik_init(pt, target_quat)
            if sol is None:
                print("❌ 致命错误：起点无法到达！请检查 Step 05 的坐标范围。")
                break
        else:
            # --- 后续点：追踪上一点 ---
            sol = robot.solve_ik_tracking(pt, target_quat, last_sol)
            
            # 如果追踪失败 (极其罕见，除非点距太大)，尝试一次全局搜索救急
            if sol is None:
                print(f"⚠️ 警告: 第 {i} 点追踪失败，尝试全局重搜...")
                sol = robot.solve_ik_init(pt, target_quat)
        
        if sol is not None:
            joint_trajectory.append(sol.tolist()) # 转 list 存入
            last_sol = sol
        else:
            fail_count += 1
            print(f"❌ 第 {i} 点无解: {pt}")
            # 简单的错误处理：沿用上一个有效的解 (机械臂停顿一下)
            if last_sol is not None:
                joint_trajectory.append(last_sol.tolist())

        # 进度条
        if i % 100 == 0:
            print(f"\r处理进度: {i}/{len(path_points)} (Errors: {fail_count})", end="")

    total_time = time.time() - start_time
    print(f"\n\n✅ 计算完成!")
    print(f"   - 耗时: {total_time:.2f} 秒")
    print(f"   - 平均每点耗时: {(total_time/len(path_points)*1000):.2f} ms")
    
    if fail_count > 0:
        print(f"⚠️ 警告: 共有 {fail_count} 个点解算失败 (已用上一帧填充)")
    else:
        print("✨ 完美！所有点均解算成功。")

    # 构造输出数据
    output_data = {
        "meta": {
            "source": "step07_inverse_kinematics",
            "unit": "radians",
            "count": len(joint_trajectory),
            "original_meta": meta
        },
        "joints": joint_trajectory
    }

    save_json_indented(OUTPUT_FILE, output_data)