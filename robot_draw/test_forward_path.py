import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time

# ==================================================================================
# 1. 简化的正运动学求解器 (用于批量计算)
# ==================================================================================
class FastFKSolver:
    def __init__(self):
        # 严格复用之前的 DH 参数
        # [alpha, a, d, theta_offset]
        self.DH_PARAMS = [
            [0,            0,      187.0,   0],          # J1
            [np.pi/2,      0,      6.0,     np.pi/2],    # J2
            [0,            210.0,  0,       -np.pi/2],   # J3
            [-np.pi/2,     0,      210.5,   0],          # J4
            [np.pi/2,      0,      0,       0],          # J5
            [-np.pi/2,     0,      159.3,   0]           # J6
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

    def get_tcp_pos(self, joints_rad):
        """
        输入: 6个关节角 (弧度)
        输出: 末端 [x, y, z] 坐标
        """
        T = np.eye(4)
        for i, (alpha, a, d, offset) in enumerate(self.DH_PARAMS):
            theta = joints_rad[i] + offset
            Ti = self.dh_matrix(alpha, a, d, theta)
            T = T @ Ti
        return T[:3, 3]

# ==================================================================================
# 2. 轨迹云图可视化器
# ==================================================================================
class TrajectoryCloudVisualizer:
    def __init__(self, json_path):
        self.json_path = json_path
        self.fk = FastFKSolver()

    def process_and_plot(self):
        if not os.path.exists(self.json_path):
            print(f"❌ 找不到文件: {self.json_path}")
            return

        print(f"📂 正在读取: {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取关节数据 (List of List)
        joints_list = data.get("joints", [])
        meta = data.get("meta", {})
        
        total_frames = len(joints_list)
        print(f"   - 包含帧数: {total_frames}")
        if total_frames == 0: return

        print("🔄 正在批量计算正运动学 (FK)...")
        start_time = time.time()
        
        # 批量计算所有点的 XYZ
        trajectory_xyz = []
        for j in joints_list:
            pos = self.fk.get_tcp_pos(j)
            trajectory_xyz.append(pos)
            
        trajectory_xyz = np.array(trajectory_xyz)
        print(f"✅ 计算完成! 耗时: {time.time() - start_time:.3f}s")

        # 开始绘图
        self.plot_cloud(trajectory_xyz, total_frames)

    def plot_cloud(self, xyz_data, total_frames):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        xs = xyz_data[:, 0]
        ys = xyz_data[:, 1]
        zs = xyz_data[:, 2]

        print("🎨 正在渲染云图...")
        
        # --- 核心：颜色映射 ---
        # 使用 'jet' 或 'viridis' 色谱，根据索引(时间)变化颜色
        # 蓝色/紫色 = 起点 (早)
        # 红色/黄色 = 终点 (晚)
        colors = np.arange(total_frames)
        
        # s=1: 点的大小，设置小一点以显示精细结构
        # alpha=0.3: 透明度，叠加起来的地方会变深，显示停顿/减速位置
        sc = ax.scatter(xs, ys, zs, c=colors, cmap='jet', s=2, alpha=0.3, label='Path Cloud')
        
        # 标记起点和终点
        ax.scatter(xs[0], ys[0], zs[0], c='green', s=100, marker='^', label='Start')
        ax.scatter(xs[-1], ys[-1], zs[-1], c='black', s=100, marker='*', label='End')

        # 添加颜色条 (Colorbar) 表示时间进度
        cbar = plt.colorbar(sc, ax=ax, pad=0.1, fraction=0.03)
        cbar.set_label('Time Progression (Frame Index)')

        # 设置坐标轴标签
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'Trajectory Cloud Visualization\nTotal Points: {total_frames}')
        
        # 自动调整比例，防止扁平化
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
        mid_x = (xs.max()+xs.min()) * 0.5
        mid_y = (ys.max()+ys.min()) * 0.5
        mid_z = (zs.max()+zs.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 调整初始视角
        ax.view_init(elev=30, azim=-45)
        ax.legend()
        
        plt.show()

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    # 指向 Step 08 生成的最终优化文件
    INPUT_FILE = os.path.join(img_dir, "step08_optimized_trajectory.json")
    
    vis = TrajectoryCloudVisualizer(INPUT_FILE)
    vis.process_and_plot()