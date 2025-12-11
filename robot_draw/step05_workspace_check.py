import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import differential_evolution
import os

# ==========================================
# 1. 机械臂运动学模型 (保持不变)
# ==========================================
class JakaRobotKinematics:
    def __init__(self):
        # [alpha, a, d, theta_offset]
        self.DH_PARAMS = [
            [0,        0,      187.0,  0],          # J1
            [np.pi/2,  0,      0,      -np.pi/2],   # J2
            [0,        210.0,  0,      0],          # J3
            [np.pi/2,  6.0,    210.5,  0],          # J4
            [-np.pi/2, 0,      0,      0],          # J5
            [np.pi/2,  0,      159.3,  0]           # J6
        ]

        # 关节限位
        limit_large = 2 * np.pi
        limit_small = (2 * np.pi) * (120/360) 
        self.bounds = [
            (-limit_large, limit_large),
            (-limit_small, limit_small),
            (-limit_small, limit_small),
            (-limit_large, limit_large),
            (-limit_small, limit_small),
            (-limit_large, limit_large)
        ]

    def dh_matrix(self, theta, alpha, a, d):
        c = np.cos(theta)
        s = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        return np.array([
            [c,    -s,    0,   a],
            [s*ca, c*ca, -sa, -d*sa],
            [s*sa, c*sa,  ca,  d*ca],
            [0,    0,     0,   1]
        ])

    def forward_kinematics(self, joints):
        T = np.eye(4)
        for i, (alpha, a, d, offset) in enumerate(self.DH_PARAMS):
            theta = joints[i] + offset
            Ti = self.dh_matrix(theta, alpha, a, d)
            T = T @ Ti
        return T

    def calculate_planar_boundaries(self, z_target, pitch_deg):
        print(f"🔄 正在计算平面边界 (Z={z_target}mm, Pitch={pitch_deg}°)...")
        rad = np.radians(pitch_deg)
        target_vec_x = np.cos(rad) 
        target_vec_z = -np.sin(rad)
        
        def objective_func(joints, sign=1.0):
            T = self.forward_kinematics(joints)
            pos = T[:3, 3]
            rot = T[:3, :3]
            z_axis_curr = rot[:, 2] 
            
            z_err = (pos[2] - z_target) ** 2
            align_err = (z_axis_curr[0] - target_vec_x)**2 + (z_axis_curr[2] - target_vec_z)**2
            
            radius = pos[0] 
            penalty = (z_err * 1000) + (align_err * 5000)
            return (sign * radius) + penalty

        bounds_reduced = self.bounds[1:] 
        def wrapper_min(x): return objective_func([0] + list(x), sign=1.0)
        def wrapper_max(x): return objective_func([0] + list(x), sign=-1.0)

        res_min = differential_evolution(wrapper_min, bounds_reduced, tol=0.1, maxiter=50)
        res_max = differential_evolution(wrapper_max, bounds_reduced, tol=0.1, maxiter=50)
        
        def get_real_r(joints_reduced):
            T = self.forward_kinematics([0] + list(joints_reduced))
            return T[0, 3]
            
        real_min_r = get_real_r(res_min.x)
        real_max_r = get_real_r(res_max.x)
        
        return real_min_r + 2.0, real_max_r - 2.0

# ==========================================
# 2. 轨迹处理器
# ==========================================
class PlanarPathProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.robot = JakaRobotKinematics()
        
    def process(self, offset_x, offset_y, draw_z, pitch_deg):
        # 1. 边界计算
        min_r, max_r = self.robot.calculate_planar_boundaries(draw_z, pitch_deg)
        print(f"✅ 边界计算完成: 最小半径 R_min = {min_r:.1f} mm, 最大半径 R_max = {max_r:.1f} mm")
        
        if min_r > max_r:
            print("❌ 错误: 该高度/姿态下无解 (Min > Max)。")
            return

        # 2. 加载轨迹
        if not os.path.exists(self.input_path):
            print(f"❌ 找不到文件: {self.input_path}")
            return
            
        with open(self.input_path, 'r') as f:
            data = json.load(f)
            
        trajectories = data.get('trajectories', [])
        meta = data.get('meta', {})
        
        print(f"📂 加载轨迹: {len(trajectories)} 条")
        
        # --- 预处理：计算图纸包围盒 ---
        all_draw_x = []
        all_draw_y = []
        for stroke in trajectories:
            for pt in stroke:
                all_draw_x.append(pt[0])
                all_draw_y.append(pt[1])
        
        if not all_draw_x:
            print("❌ 错误：轨迹数据为空")
            return

        # 获取图纸坐标系下的最大最小值
        draw_min_x, draw_max_x = min(all_draw_x), max(all_draw_x)
        draw_min_y, draw_max_y = min(all_draw_y), max(all_draw_y)

        # 🔥 [新增] 定义图纸的4个角点（按顺时针或逆时针顺序）
        # 左下 -> 右下 -> 右上 -> 左上
        draw_corners = [
            (draw_min_x, draw_min_y),
            (draw_max_x, draw_min_y),
            (draw_max_x, draw_max_y),
            (draw_min_x, draw_max_y)
        ]

        # 🔥 [新增] 将4个角点转换到机械臂物理坐标系
        paper_bbox_robot = []
        for dx, dy in draw_corners:
            rx = dy + offset_x
            ry = -dx + offset_y
            paper_bbox_robot.append([rx, ry])

        # 计算原点信息 (取转换后的第一个点，即左下角)
        origin_info = {
            "desc": "Bottom-Left corner of drawing content in Robot Frame",
            "x": round(paper_bbox_robot[0][0], 3),
            "y": round(paper_bbox_robot[0][1], 3),
            "z": round(draw_z, 3)
        }
        print(f"📍 图纸原点(左下角) 物理坐标: [{origin_info['x']}, {origin_info['y']}, {origin_info['z']}]")

        # 3. 转换轨迹并校验
        transformed_trajectories = []
        all_points_robot = [] 
        is_safe = True
        
        for stroke in trajectories:
            new_stroke = []
            for pt in stroke:
                draw_x, draw_y = pt[0], pt[1]
                # 坐标转换
                r_x = draw_y + offset_x
                r_y = -draw_x + offset_y
                
                # 校验
                radius = np.sqrt(r_x**2 + r_y**2)
                if not (min_r <= radius <= max_r):
                    is_safe = False
                    print(f"⚠️ 越界: ({r_x:.1f}, {r_y:.1f}) R={radius:.1f}")
                
                new_stroke.append([r_x, r_y, draw_z])
                all_points_robot.append([r_x, r_y])
                
            transformed_trajectories.append(new_stroke)

        # 4. 结果处理与可视化
        # 🔥 [新增] 将 paper_bbox_robot 传给可视化函数
        if is_safe:
            print("✅ [校验通过] 所有点均在有效工作空间内。")
            self.save_result(transformed_trajectories, meta, offset_x, offset_y, draw_z, pitch_deg, origin_info)
            self.visualize(all_points_robot, min_r, max_r, offset_x, offset_y, origin_info, paper_bbox_robot)
        else:
            print("❌ [校验失败] 存在越界点。")
            self.visualize(all_points_robot, min_r, max_r, offset_x, offset_y, origin_info, paper_bbox_robot)

    def save_result(self, trajectories, meta, off_x, off_y, z, pitch, origin_info):
        # ... (保存函数不变，元数据已在 process 中处理)
        output = {
            "meta": {
                "source": "step05_workspace_check_plane",
                "offset_applied": [off_x, off_y],
                "z_height_mm": z,
                "target_pitch_deg": pitch,
                "paper_origin_robot_frame": origin_info 
            },
            "trajectories": [np.round(s, 3).tolist() for s in trajectories]
        }
        
        with open(self.output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"💾 文件已保存: {self.output_path}")

    # 🔥 [修改] 增加 paper_bbox 参数
    def visualize(self, points, min_r, max_r, off_x, off_y, origin_info, paper_bbox):
        pts = np.array(points)
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 1. 画圆环 (工作空间)
        wedge = patches.Wedge((0, 0), max_r, 0, 360, width=max_r-min_r, 
                              color='green', alpha=0.1, label='Reach Limit')
        ax.add_patch(wedge)
        
        # 🔥 2. [新增] 画图纸范围框 (蓝色虚线矩形)
        # 使用 Polygon 来绘制转换后的四个角点
        paper_patch = patches.Polygon(paper_bbox, closed=True, 
                                      edgecolor='blue', facecolor='none', 
                                      linewidth=2, linestyle='--', label='Paper Border')
        ax.add_patch(paper_patch)
        
        # 3. 画轨迹点 (红色)
        ax.scatter(pts[:,0], pts[:,1], s=1, c='red', label='Trajectory')
        
        # 4. 画机械臂基座 (黑色原点)
        ax.plot(0, 0, 'ko', markersize=10, label='Robot Base (0,0)')
        
        # 5. 画图纸原点 (蓝色星星)
        ax.plot(origin_info['x'], origin_info['y'], 'b*', markersize=15, label='Paper Origin (Bottom-Left)')
        
        # 设置视图
        ax.set_aspect('equal')
        ax.set_title(f'Workspace & Paper Location Check\nOffset: X={off_x}, Y={off_y}')
        ax.legend(loc='upper right')
        ax.grid(True)
        
        # 自动缩放视野以显示所有内容
        all_x = pts[:,0].tolist() + [p[0] for p in paper_bbox] + [0]
        all_y = pts[:,1].tolist() + [p[1] for p in paper_bbox] + [0]
        margin = 50
        ax.set_xlim(min(all_x)-margin, max(all_x)+margin)
        ax.set_ylim(min(all_y)-margin, max(all_y)+margin)
        
        plt.show()

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    # ... (主程序入口保持不变)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    INPUT_FILE = os.path.join(img_dir, "step04_path.json")
    OUTPUT_FILE = os.path.join(img_dir, "step05_re_path.json")
    
    processor = PlanarPathProcessor(INPUT_FILE, OUTPUT_FILE)
    
    # 参数配置 (可以在这里调整 OFFSET 来看看图纸框在视野中怎么移动)
    TARGET_Z = 50.0  
    TARGET_PITCH = 0.0
    OFFSET_X = 300.0  # 加大一点X偏移，让它离基座远点
    OFFSET_Y = 125.0 
    
    processor.process(OFFSET_X, OFFSET_Y, TARGET_Z, TARGET_PITCH)