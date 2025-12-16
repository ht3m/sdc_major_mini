import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ==========================================
# 1. 纯几何计算器 (JAKA Mini 2 专用)
# ==========================================
class JakaGeometricSolver:
    def __init__(self):
        # 物理尺寸 (基于 DH 参数提取)
        # J1 高度
        self.d1 = 187.0
        # J2 肩部水平偏置 (d2)
        self.shoulder_offset = 6.0 
        # J3 大臂长度 (a2)
        self.L1 = 210.0
        # J4 小臂长度 (d4 in DH, acts as link length here)
        self.L2 = 210.5
        # J6 工具法兰长度 (d6)
        self.tool_len = 159.3

    def calculate_boundaries(self, target_z):
        """
        使用勾股定理计算垂直姿态下的工作半径
        """
        print(f"📐 正在进行几何计算 (Target Z={target_z}mm, Pitch=90°)...")

        # 1. 计算腕部关节点 (Wrist Center) 需要在的高度
        # 笔尖在 target_z, 笔垂直向下, 所以腕部在 target_z + tool_len
        wrist_z = target_z + self.tool_len
        
        # 2. 计算肩部关节点 (Shoulder Center) 的高度
        shoulder_z = self.d1
        
        # 3. 计算垂直高度差 (Vertical Gap)
        h_diff = abs(wrist_z - shoulder_z)
        
        # 4. 计算机械臂在垂直平面内的伸展能力
        # 大臂 L1 和 小臂 L2 组成的折叠臂
        
        # 最大伸展长度 (手臂伸直)
        arm_span_max = self.L1 + self.L2
        
        # 最小折叠长度 (手臂死弯)
        arm_span_min = abs(self.L1 - self.L2)
        
        # 5. 校验高度是否可达
        if h_diff > arm_span_max:
            print(f"❌ 错误: 目标高度太远! 高度差 {h_diff:.1f} > 手臂最大跨度 {arm_span_max:.1f}")
            return 0.0, 0.0
            
        # 6. 利用勾股定理计算水平投影半径 (r_horizontal)
        # r^2 + h^2 = span^2  =>  r = sqrt(span^2 - h^2)
        
        # 计算最大水平延伸
        r_flat_max = np.sqrt(arm_span_max**2 - h_diff**2)
        
        # 计算最小水平收缩
        # 如果高度差小于最小折叠长(非常罕见)，则存在物理死区
        # 如果高度差大于最小折叠长，说明手臂可以"穿过"中心轴(理论上)，
        # 但考虑到实体碰撞和 shoulder_offset，我们取几何极限
        if h_diff >= arm_span_min:
            r_flat_min = 0.0 # 理论上可以收到 0，实际上受 offset 限制
        else:
            r_flat_min = np.sqrt(arm_span_min**2 - h_diff**2)
            
        # 7. 考虑肩部水平偏置 (Shoulder Offset d2=6.0)
        # 实际半径 R = sqrt(r_flat^2 + d2^2)
        # 这个偏置总是让半径变大一点点
        
        final_R_max = np.sqrt(r_flat_max**2 + self.shoulder_offset**2)
        final_R_min = np.sqrt(r_flat_min**2 + self.shoulder_offset**2)
        
        # 增加工程安全余量 (防止奇异点和全伸直锁死)
        # 收缩 5mm
        safe_R_max = final_R_max - 5.0
        # 扩张 50mm (太近了容易撞到底座，给自己留点空间)
        safe_R_min = max(final_R_min + 5.0, 150.0) 
        
        return safe_R_min, safe_R_max

# ==========================================
# 2. 轨迹处理器
# ==========================================
class PlanarPathProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.solver = JakaGeometricSolver()
        
    def process(self, offset_x, offset_y, draw_z):
        # 1. 几何边界计算
        min_r, max_r = self.solver.calculate_boundaries(draw_z)
        
        if min_r == 0 and max_r == 0:
            print("❌ 无法计算有效边界。")
            return
            
        print(f"✅ 几何边界: R_min = {min_r:.1f} mm, R_max = {max_r:.1f} mm")
        
        if min_r >= max_r:
            print("❌ 错误: 有效空间过小或不存在。")
            return

        # 2. 加载轨迹
        if not os.path.exists(self.input_path):
            print(f"❌ 找不到文件: {self.input_path}")
            return
            
        with open(self.input_path, 'r') as f:
            data = json.load(f)
            
        trajectories = data.get('trajectories', [])
        
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

        draw_min_x, draw_max_x = min(all_draw_x), max(all_draw_x)
        draw_min_y, draw_max_y = min(all_draw_y), max(all_draw_y)

        # 定义图纸4角
        draw_corners = [
            (draw_min_x, draw_min_y),
            (draw_max_x, draw_min_y),
            (draw_max_x, draw_max_y),
            (draw_min_x, draw_max_y)
        ]

        # 转换到机械臂坐标系
        paper_bbox_robot = []
        for dx, dy in draw_corners:
            rx = dy + offset_x
            ry = -dx + offset_y
            paper_bbox_robot.append([rx, ry])

        # 原点信息
        origin_info = {
            "desc": "Bottom-Left corner of drawing content in Robot Frame",
            "x": round(paper_bbox_robot[0][0], 3),
            "y": round(paper_bbox_robot[0][1], 3),
            "z": round(draw_z, 3)
        }
        print(f"📍 图纸原点: [{origin_info['x']}, {origin_info['y']}, {origin_info['z']}]")

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
                    print(f"⚠️ 越界: ({r_x:.1f}, {r_y:.1f}) R={radius:.1f} (Limit: {min_r:.0f}~{max_r:.0f})")
                
                new_stroke.append([r_x, r_y, draw_z])
                all_points_robot.append([r_x, r_y])
                
            transformed_trajectories.append(new_stroke)

        # 4. 结果处理与可视化
        # 垂直写字模式下 pitch 固定为 90
        target_pitch = 90.0 
        
        if is_safe:
            print("✅ [校验通过] 所有点均在有效工作空间内。")
            self.save_result(transformed_trajectories, offset_x, offset_y, draw_z, target_pitch, origin_info)
            self.visualize(all_points_robot, min_r, max_r, offset_x, offset_y, origin_info, paper_bbox_robot)
        else:
            print("❌ [校验失败] 存在越界点。请调整 OFFSET 或 图纸大小。")
            self.visualize(all_points_robot, min_r, max_r, offset_x, offset_y, origin_info, paper_bbox_robot)

    def save_result(self, trajectories, off_x, off_y, z, pitch, origin_info):
        output = {
            "meta": {
                "source": "step05_geometric_check",
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

    def visualize(self, points, min_r, max_r, off_x, off_y, origin_info, paper_bbox):
        pts = np.array(points)
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 1. 画满360度圆环
        # 外圆
        theta = np.linspace(0, 2*np.pi, 200)
        x_out = max_r * np.cos(theta)
        y_out = max_r * np.sin(theta)
        # 内圆
        x_in = min_r * np.cos(theta)
        y_in = min_r * np.sin(theta)
        
        # 填充
        wedge = patches.Wedge((0, 0), max_r, 0, 360, width=max_r-min_r, 
                              color='#2ca02c', alpha=0.15, label=f'Reach {min_r:.0f}-{max_r:.0f}mm')
        ax.add_patch(wedge)
        
        # 画边界线
        ax.plot(x_out, y_out, color='#2ca02c', linestyle='--', linewidth=1)
        ax.plot(x_in, y_in, color='#2ca02c', linestyle='--', linewidth=1)
        
        # 2. 画图纸框
        paper_patch = patches.Polygon(paper_bbox, closed=True, 
                                      edgecolor='blue', facecolor='none', 
                                      linewidth=2, linestyle='-', label='Paper Area')
        ax.add_patch(paper_patch)
        
        # 3. 画轨迹点
        ax.scatter(pts[:,0], pts[:,1], s=1, c='red', label='Trajectory', zorder=10)
        
        # 4. 画机械臂基座
        ax.plot(0, 0, 'ko', markersize=12, label='Robot Base', zorder=10)
        
        # 5. 画图纸原点
        ax.plot(origin_info['x'], origin_info['y'], 'b*', markersize=15, label='Start Point', zorder=10)
        
        ax.set_aspect('equal')
        ax.set_title(f'Geometric Workspace Check (Pitch=90, Z={origin_info["z"]})\nOffset: X={off_x}, Y={off_y}')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 自动缩放
        limit = max_r + 100
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        plt.show()

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    INPUT_FILE = os.path.join(img_dir, "step04_path.json")
    OUTPUT_FILE = os.path.join(img_dir, "step05_re_path.json")
    
    processor = PlanarPathProcessor(INPUT_FILE, OUTPUT_FILE)
    
    # 调整这些参数
    TARGET_Z = 50.0  
    
    # 经验值：JAKA Mini 2 在垂直书写姿态下，最佳工作区通常在前方 200-450mm 之间
    # 建议将图纸放在 Y=0, X=300 附近
    OFFSET_X = 250.0 
    OFFSET_Y = 100.0 
    
    # Pitch 参数在此版本中不再作为变量输入计算，因为几何解算已默认其为 90度 (垂直)
    # 但我们仍保留变量传递以兼容后续步骤
    processor.process(OFFSET_X, OFFSET_Y, TARGET_Z)