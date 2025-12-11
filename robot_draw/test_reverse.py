import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

class JakaRobot:
    def __init__(self):
        # ==========================================
        # 1. 用户指定的 DH 参数
        # 严格顺序: [alpha, a, d, theta_offset]
        # ==========================================
        self.DH_PARAMS = [
            # alpha        a       d       theta_offset
            [0,            0,      187.0,  0],          # J1
            [np.pi/2,      0,      6.0,    np.pi/2],    # J2 (User: Offset +90)
            [0,            210.0,  0,      -np.pi/2],   # J3 (User: Offset -90)
            [-np.pi/2,     0,      210.5,  0],          # J4
            [np.pi/2,      0,      0,      0],          # J5
            [-np.pi/2,     0,      159.3,  0]           # J6
        ]

        # 2. 关节物理限位 (弧度)
        large = 2 * np.pi          # +/- 360
        small = (2 * np.pi) * (120/360)  # +/- 120
        
        self.bounds = [
            (-large, large), # J1
            (-small, small), # J2
            (-small, small), # J3
            (-large, large), # J4
            (-small, small), # J5
            (-large, large)  # J6
        ]

    def dh_matrix_modified(self, alpha, a, d, theta):
        """改进型 DH 矩阵 (Craig)"""
        c = np.cos(theta)
        s = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        return np.array([
            [c,      -s,      0,      a],
            [s*ca,   c*ca,   -sa,    -d*sa],
            [s*sa,   c*sa,    ca,     d*ca],
            [0,       0,       0,      1]
        ])

    def forward_kinematics_path(self, joints):
        """
        计算 FK，同时返回【关节路径点】用于画图
        """
        T = np.eye(4)
        path = [T[:3, 3].copy()] # 记录基座原点
        
        for i, (alpha, a, d, offset) in enumerate(self.DH_PARAMS):
            theta = joints[i] + offset
            Ti = self.dh_matrix_modified(alpha, a, d, theta)
            T = T @ Ti
            path.append(T[:3, 3].copy()) # 记录每一级关节原点
            
        return T, np.array(path)

    def solve_ik_all(self, target_pos, target_quat):
        """
        尝试寻找所有可能的逆解 (Multi-Seed)
        """
        t_pos = np.array(target_pos)
        r = R.from_quat(target_quat)
        t_rot = r.as_matrix()

        # 定义误差函数
        def error_func(q):
            # 1. 计算当前位姿
            T, _ = self.forward_kinematics_path(q)
            curr_pos = T[:3, 3]
            curr_rot = T[:3, :3]
            
            # 2. 误差计算
            pos_err = np.linalg.norm(curr_pos - t_pos)
            rot_err = np.linalg.norm(curr_rot - t_rot)
            
            # 3. 越界惩罚 (Soft Constraint)
            penalty = 0
            for i, val in enumerate(q):
                if val < self.bounds[i][0] or val > self.bounds[i][1]:
                    penalty += 100 * (abs(val) - abs(self.bounds[i][1]))**2
            
            return pos_err * 1.0 + rot_err * 20.0 + penalty

        # 生成 8 个典型种子，覆盖不同象限
        seeds = []
        for j2 in [-0.5, 0.5]:     # 肩膀 前/后
            for j3 in [-1.5, 1.5]: # 肘部 上/下
                for j5 in [-1.0, 1.0]: # 手腕 翻/不翻
                    seeds.append(np.array([0, j2, j3, 0, j5, 0]))

        found_solutions = []
        
        print(f"🔍 正在计算逆解 (目标: {t_pos})...")
        
        for seed in seeds:
            res = minimize(
                error_func, seed, method='SLSQP', bounds=self.bounds, tol=1e-5
            )
            
            if res.fun < 0.1: # 误差够小才算找到
                # 查重
                is_duplicate = False
                for sol in found_solutions:
                    if np.allclose(res.x, sol, atol=0.1): # 约5.7度内视为相同
                        is_duplicate = True
                        break
                if not is_duplicate:
                    found_solutions.append(res.x)

        return found_solutions

def visualize_solution(robot, joints, target_pos, solution_index=1):
    """
    可视化：画出机械臂骨架 + 目标点
    """
    # 计算骨架路径
    T_final, path = robot.forward_kinematics_path(joints)
    final_pos = T_final[:3, 3]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 绘制机械臂连杆
    xs, ys, zs = path[:, 0], path[:, 1], path[:, 2]
    ax.plot(xs, ys, zs, linewidth=3, color='#1f77b4', label='Robot Link')
    
    # 2. 绘制关节球
    ax.scatter(xs, ys, zs, s=60, c='#ff7f0e', marker='o')
    
    # 3. 绘制基座
    ax.scatter(xs[0], ys[0], zs[0], s=100, c='black', marker='^', label='Base')
    
    # 4. 绘制目标点 (红色虚影球)
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
               s=150, c='red', alpha=0.3, label='Target Goal')
    
    # 5. 绘制实际到达点 (红色星号)
    ax.scatter(final_pos[0], final_pos[1], final_pos[2], 
               s=100, c='red', marker='*', label='Actual TCP')
    
    # 添加文字标签
    for i, (x, y, z) in enumerate(path):
        label = "TCP" if i == len(path)-1 else f"J{i}"
        ax.text(x, y, z+10, label, fontsize=8)

    # 坐标轴设置
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (xs.max()+xs.min())*0.5, (ys.max()+ys.min())*0.5, (zs.max()+zs.min())*0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f"Solution #{solution_index} Visualization\nJoints: {np.degrees(joints).round(1)}")
    ax.legend()
    
    plt.show()

# ==========================================
# Main Interface
# ==========================================
if __name__ == "__main__":
    robot = JakaRobot()
    
    # 👉 【接口】设置目标点
    TARGET_POS = [300.0, 0.0, 50.0]
    TARGET_QUAT = [0.0, 1.0, 0.0, 0.0]
    
    solutions = robot.solve_ik_all(TARGET_POS, TARGET_QUAT)
    
    if not solutions:
        print("❌ 未找到逆解！(目标可能超出工作空间)")
    else:
        print(f"✅ 成功找到 {len(solutions)} 组解：")
        print("=" * 60)
        
        for i, sol in enumerate(solutions):
            deg = np.degrees(sol)
            
            # 使用 join 方法生成逗号分隔的字符串
            # {:.2f} 保留两位小数, {:.4f} 保留四位小数
            deg_str = ", ".join([f"{d:.2f}" for d in deg])
            rad_str = ", ".join([f"{r:.4f}" for r in sol])
            
            print(f"[解 {i+1}]")
            print(f"  角度 (Deg): {deg_str}")
            print(f"  弧度 (Rad): {rad_str}")
            print("-" * 60)
            
        print(f"\n🎨 正在绘制第 1 组解...")
        visualize_solution(robot, solutions[0], TARGET_POS, solution_index=1)