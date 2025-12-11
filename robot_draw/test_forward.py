import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
# 必须显式导入 Axes3D 才能支持 3D 绘图
from mpl_toolkits.mplot3d import Axes3D 

class JakaFKSolver:
    def __init__(self):
        # ==========================================
        # 1. 最终确定的 DH 参数 (软件导出版)
        # 严格顺序: [alpha, a, d, theta_offset]
        # ==========================================
        self.DH_PARAMS = [
            # alpha        a       d       theta
            [0,            0,      187.0,        0],      # J1
            [np.pi/2,      0,      6.0,    np.pi/2],      # J2
            [0,            210.0,  0,     -np.pi/2],      # J3
            [-np.pi/2,     0,      210.5,        0],      # J4
            [np.pi/2,      0,      0,            0],      # J5
            [-np.pi/2,     0,      159.3,        0]       # J6
        ]

    def dh_matrix_modified(self, alpha, a, d, theta):
        """改进型 D-H 变换矩阵"""
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

    def solve_fk(self, joints_deg):
        """计算正运动学，并返回关键点路径用于绘图"""
        joints_rad = np.radians(joints_deg)
        T = np.eye(4)
        
        # 记录路径点：从基座(Frame0)开始
        joint_positions = [T[:3, 3].copy()]
        
        print(f"{'Frame':<6} | {'alpha':<8} | {'a':<8} | {'d':<8} | {'theta':<8} | {'当前原点 Z高':<12}")
        print("-" * 70)

        for i, (alpha, a, d, offset) in enumerate(self.DH_PARAMS):
            theta = joints_rad[i] + offset
            Ti = self.dh_matrix_modified(alpha, a, d, theta)
            T = T @ Ti
            
            # 记录当前 Frame 原点的位置
            current_pos = T[:3, 3].copy()
            joint_positions.append(current_pos)
            
            print(f"F{i+1:<5} | {alpha:<8.2f} | {a:<8.1f} | {d:<8.1f} | {np.degrees(theta):<8.1f} | {current_pos[2]:<12.1f}")

        pos = T[:3, 3]
        rot_mat = T[:3, :3]
        r = R.from_matrix(rot_mat)
        quat = r.as_quat()
        euler = r.as_euler('xyz', degrees=True)
        
        return pos, quat, euler, np.array(joint_positions)

def visualize_robot_with_labels(joint_path):
    """
    绘制机械臂骨架，并标注每个关节的 XYZ 坐标
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 提取坐标
    xs = joint_path[:, 0]
    ys = joint_path[:, 1]
    zs = joint_path[:, 2]

    # 1. 绘制连杆骨架线条
    ax.plot(xs, ys, zs, linewidth=2, color='gray', linestyle='--', label='Links')
    
    # 2. 绘制关键点并添加坐标标签
    num_points = len(joint_path)
    for i, (x, y, z) in enumerate(joint_path):
        # 判断点的类型并设置颜色和标签文本
        if i == 0:
            # 基座 Frame 0
            label_text = f"Base(F0)\n({x:.0f}, {y:.0f}, {z:.0f})"
            color = 'black'
            marker = '^'
            size = 100
            offset_z = -20 # 标签往下放一点
        elif i == num_points - 1:
            # 末端 TCP (Frame 6)
            label_text = f"TCP(F6)\n({x:.1f}, {y:.1f}, {z:.1f})"
            color = 'red'
            marker = '*'
            size = 150
            offset_z = 20 # 标签往上放一点
        else:
            # 中间关节 (Frame 1~5)
            # 注意：Frame i 的原点通常对应物理上的关节 i+1 附近
            label_text = f"F{i}\n({x:.0f}, {y:.0f}, {z:.0f})"
            color = '#ff7f0e' # 橙色
            marker = 'o'
            size = 60
            offset_z = 10 if i%2==0 else -15 # 错开标签防止重叠

        # 绘制点
        ax.scatter(x, y, z, s=size, c=color, marker=marker)
        
        # 添加文字标签 (ax.text)
        # 在坐标点附近添加文本，稍微偏移一点以免盖住点
        ax.text(x, y, z + offset_z, label_text, color=color, fontsize=9, 
                horizontalalignment='center')

    # 设置绘图环境
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('JAKA Robot Joint Coordinates Visualization')
    
    # 强制坐标轴比例一致，防止变形
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置初始视角 (俯视侧面，方便看清楚)
    ax.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    print("\n🎨 3D 窗口已生成，请查看弹窗。你可以用鼠标旋转和缩放视图。")
    plt.show()

if __name__ == "__main__":
    fk_solver = JakaFKSolver()

    # ==========================================
    # 👉 【接口】在此处修改你的关节角度 (单位: 度)
    # ==========================================
    
    # 测试姿态 1: 直立状态 (检查高度是否对)
    # USER_JOINTS = [0, 0, 0, 0, 0, 0]

    # 测试姿态 2: 典型的写字姿态 (检查坐标是否合理)
    # J2前倾，J3下折，J5把笔尖对准地面
    USER_JOINTS = [180, 90, 0, 0, 0, 0]
    
    # ==========================================

    print("="*60)
    print(f"📍 正运动学计算 + 坐标可视化")
    print(f"   输入关节角: {USER_JOINTS}")
    print("="*60)

    # 计算并获取路径
    pos, quat, euler, joint_path = fk_solver.solve_fk(USER_JOINTS)

    print("\n✅ 最终末端结果:")
    print(f"   位置 (XYZ): [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] mm")
    print("="*60)
    
    # 可视化
    visualize_robot_with_labels(joint_path)