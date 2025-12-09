import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

class WorkspaceValidator:
    def __init__(self, json_path):
        self.json_path = json_path
        # JAKA Mini 2 物理参数
        self.MAX_RADIUS = 579.8  # 最大臂展
        self.MIN_RADIUS = 180.0  # 最小安全半径
        
    def load_trajectory(self):
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"找不到文件: {self.json_path}")
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        return data

    def transform_and_check(self, offset_x, offset_y, z_height=0.0, rotate_90=True):
        """
        坐标转换并校验
        :return: (transformed_points, is_safe, meta_data)
        """
        data = self.load_trajectory()
        trajectories = data['trajectories']
        meta = data.get('meta', {})
        
        # 1. Z轴高度检查
        if abs(z_height) > self.MAX_RADIUS:
            print(f"❌ 严重错误: 目标高度 Z={z_height} 超过了机械臂物理极限!")
            return [], False, meta

        # 2. 计算当前高度下的有效最大半径
        effective_max_radius = np.sqrt(self.MAX_RADIUS**2 - z_height**2)
        
        print(f"[Check] 正在校验轨迹...")
        print(f"       图纸原点设为: (X={offset_x}, Y={offset_y})")
        print(f"       工作高度 Z: {z_height} mm")
        print(f"       有效最大臂展: {effective_max_radius:.1f} mm")

        robot_points = []
        is_safe = True

        for stroke in trajectories:
            stroke_transformed = []
            for pt in stroke:
                draw_x, draw_y = pt[0], pt[1]
                
                # --- 坐标转换核心逻辑 ---
                if rotate_90:
                    # 旋转90度 + 镜像修正
                    # 图纸 X+ (向右) -> 机械臂 Y- (向右)
                    # 图纸 Y+ (向上) -> 机械臂 X+ (向前)
                    r_x = draw_y + offset_x
                    r_y = -draw_x + offset_y 
                else:
                    r_x = draw_x + offset_x
                    r_y = draw_y + offset_y
                
                stroke_transformed.append([r_x, r_y])
                
                # --- 范围检查 ---
                dist = np.sqrt(r_x**2 + r_y**2)
                
                if dist > effective_max_radius:
                    print(f"❌ 越界: 点 ({r_x:.1f}, {r_y:.1f}) 距离 {dist:.1f} > 极限 {effective_max_radius:.1f}")
                    is_safe = False
                
                if dist < self.MIN_RADIUS:
                    print(f"❌ 过近: 点 ({r_x:.1f}, {r_y:.1f}) 距离 {dist:.1f} < 安全半径 {self.MIN_RADIUS}")
                    is_safe = False
                    
            robot_points.append(np.array(stroke_transformed))

        if is_safe:
            print("✅ 校验通过！所有轨迹均在安全工作空间内。")
        else:
            print("⚠️ 校验失败！存在越界风险。")
            
        return robot_points, is_safe, meta

    def save_transformed_json(self, robot_points, meta, output_path, z_height):
        """
        【新增】将转换后的机械臂坐标保存为新的 JSON 文件
        """
        print(f"[Save] 正在保存转换后的轨迹到: {output_path} ...")
        
        # 将 numpy 数组转回 list，保留 3 位小数
        final_trajectories = []
        for stroke in robot_points:
            # stroke 是 np.array, 需要转成 list
            stroke_list = np.round(stroke, 3).tolist()
            final_trajectories.append(stroke_list)
            
        # 更新 meta 信息
        new_meta = meta.copy()
        new_meta["coordinate_system"] = "robot_base_frame"
        new_meta["z_height_mm"] = z_height
        new_meta["description"] = "Coordinates are absolute (X, Y) relative to robot base."
        
        output_data = {
            "meta": new_meta,
            "trajectories": final_trajectories
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"✅ 文件已保存成功！可以直接用于逆解算或控制。")

    def visualize(self, robot_points, offset_x, offset_y):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(0, 0, 'ko', markersize=10, label='Robot Base (0,0)')
        
        circle_max = patches.Circle((0, 0), self.MAX_RADIUS, color='green', alpha=0.1, label='Max Reach')
        ax.add_patch(circle_max)
        circle_min = patches.Circle((0, 0), self.MIN_RADIUS, color='red', alpha=0.3, label='Min Safe Zone')
        ax.add_patch(circle_min)
        
        for stroke in robot_points:
            ax.plot(stroke[:, 0], stroke[:, 1], 'b-', linewidth=1)
            
        ax.set_aspect('equal')
        ax.set_title(f"Workspace Check & Export (Offset: {offset_x}, {offset_y})")
        ax.set_xlabel("Robot X (mm) - Forward")
        ax.set_ylabel("Robot Y (mm) - Left/Right")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        limit = self.MAX_RADIUS + 100
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        plt.show()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    # 输入文件 (Step 4 的输出)
    INPUT_JSON = os.path.join(img_dir, "step04_robot_paths.json")
    # 输出文件 (转换后的机械臂坐标轨迹)
    OUTPUT_JSON = os.path.join(img_dir, "step05_re_robot_path.json")
    
    validator = WorkspaceValidator(INPUT_JSON)
    
    # --- 参数设置 ---
    TEST_Z = 20.0       # 笔尖高度
    OFFSET_X = 350.0    # 前后偏移 (推荐 300-400)
    OFFSET_Y = 75.0      # 左右偏移 (0 代表正前方)
    
    # 1. 转换与检查
    points, safe, meta_data = validator.transform_and_check(
        OFFSET_X, OFFSET_Y, z_height=TEST_Z, rotate_90=True
    )
    
    # 2. 如果安全，则保存文件
    if safe:
        validator.save_transformed_json(points, meta_data, OUTPUT_JSON, TEST_Z)
        # 3. 可视化确认
        validator.visualize(points, OFFSET_X, OFFSET_Y)
    else:
        print("❌ 轨迹不安全，未保存文件。请调整 Offset 参数。")