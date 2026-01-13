import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class StructuredPathGenerator:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def linear_interpolate(self, p_start, p_end, step_size):
        """
        在两点之间生成均匀的线性插值点
        """
        p_start = np.array(p_start)
        p_end = np.array(p_end)
        
        dist = np.linalg.norm(p_end - p_start)
        
        if dist < 1e-3:
            return [p_end.tolist()]
        
        num_points = int(np.ceil(dist / step_size))
        t_values = np.linspace(0, 1, num_points + 1)
        
        interpolated_points = []
        for t in t_values:
            pt = p_start + (p_end - p_start) * t
            interpolated_points.append(pt.tolist())
            
        return interpolated_points

    def generate(self, draw_z, air_z, air_step_mm):
        print(f"🔄 开始生成结构化路径 (N笔画 x 4阶段)...")
        print(f"   - 落笔高度: {draw_z} mm")
        print(f"   - 抬笔高度: {air_z} mm")

        # 1. 读取 Step 05 数据
        if not os.path.exists(self.input_path):
            print(f"❌ 找不到文件: {self.input_path}")
            return

        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_strokes = data.get("trajectories", [])
        meta = data.get("meta", {})
        
        if not raw_strokes:
            print("❌ 错误：输入文件中没有轨迹数据")
            return

        # ==========================================
        # 🔥 修改点：应用 90 度旋转 (x -> -y, y -> x)
        # ==========================================
        print("🔄 正在应用坐标变换: 旋转 90° (x -> -y, y -> x)...")
        rotated_strokes = []
        for stroke in raw_strokes:
            # pt[0] 是旧x, pt[1] 是旧y
            # 新x = -旧y, 新y = 旧x
            new_stroke = [[-pt[1], pt[0]] for pt in stroke]
            rotated_strokes.append(new_stroke)
        
        # 用旋转后的数据替换原始数据
        raw_strokes = rotated_strokes

        # 同时也要处理 Meta 中的原点信息（如果有的话），保持逻辑一致
        origin_info = meta.get("paper_origin_robot_frame", None)
        if origin_info:
            old_x = origin_info['x']
            old_y = origin_info['y']
            
            # 更新原点坐标
            origin_info['x'] = -old_y
            origin_info['y'] = old_x
            
            current_air_pos = np.array([origin_info['x'], origin_info['y'], air_z])
            print(f"✅ (已旋转) 起点设置为图纸原点: {current_air_pos}")
        else:
            # 如果没有原点信息，取旋转后的第一笔起点
            first_pt = raw_strokes[0][0]
            current_air_pos = np.array([first_pt[0], first_pt[1], air_z])
            print(f"⚠️ 未找到原点信息，起点设置为第一笔上方: {current_air_pos}")

        # ==========================================
        # 后续逻辑保持不变
        # ==========================================

        structured_strokes = [] 
        total_points_count = 0

        for i, stroke in enumerate(raw_strokes):
            stroke_segments = [] 
            
            stroke_start_2d = stroke[0]
            stroke_end_2d = stroke[-1]

            # P_Hover_Start
            p_hover_start = np.array([stroke_start_2d[0], stroke_start_2d[1], air_z])
            # P_Start
            p_draw_start  = np.array([stroke_start_2d[0], stroke_start_2d[1], draw_z])
            # P_End
            p_draw_end    = np.array([stroke_end_2d[0], stroke_end_2d[1], draw_z])
            # P_Hover_End
            p_hover_end   = np.array([stroke_end_2d[0], stroke_end_2d[1], air_z])

            # 1. Move
            seg_move = self.linear_interpolate(current_air_pos, p_hover_start, air_step_mm)
            stroke_segments.append(seg_move)

            # 2. Drop
            seg_drop = self.linear_interpolate(p_hover_start, p_draw_start, air_step_mm)
            stroke_segments.append(seg_drop)

            # 3. Draw
            current_draw_points = []
            for pt in stroke:
                current_draw_points.append([pt[0], pt[1], draw_z])
            stroke_segments.append(current_draw_points)

            # 4. Lift
            seg_lift = self.linear_interpolate(p_draw_end, p_hover_end, air_step_mm)
            stroke_segments.append(seg_lift)

            structured_strokes.append(stroke_segments)
            current_air_pos = p_hover_end
            total_points_count += len(seg_move) + len(seg_drop) + len(current_draw_points) + len(seg_lift)

        # 保存
        output_data = {
            "meta": {
                "source": "step06_structured_path",
                "rotation": "90_degrees_counter_clockwise", # 更新描述
                "structure_format": "[N_strokes, 4_segments, M_points, 3_coords]",
                "segment_meaning": ["0:AirMove", "1:Drop", "2:Draw", "3:Lift"],
                "total_strokes": len(structured_strokes),
                "total_points_flat": total_points_count,
                "draw_z": draw_z,
                "air_z": air_z,
                "step_mm": air_step_mm,
                "parent_meta": meta
            },
            "structured_path": structured_strokes 
        }

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2) 
            
        print(f"✅ 结构化路径生成完毕!")
        print(f"   - 笔画数: {len(structured_strokes)}")
        print(f"   - 结构: [Move -> Drop -> Draw -> Lift]")
        print(f"💾 文件已保存: {self.output_path}")
        
        self.visualize_3d(structured_strokes, draw_z, air_z)

    def visualize_3d(self, structured_data, draw_z, air_z):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        print("🎨 正在绘制 3D 预览...")
        
        colors = ['green', 'orange', 'blue', 'red']
        styles = ['--', '--', '-', '--']
        labels = ['Move', 'Drop', 'Draw', 'Lift']
        added_labels = set()

        for stroke_segs in structured_data:
            for i, seg_points in enumerate(stroke_segs):
                pts = np.array(seg_points)
                if len(pts) < 1: continue
                
                label = labels[i] if labels[i] not in added_labels else None
                if label: added_labels.add(label)

                ax.plot(pts[:,0], pts[:,1], pts[:,2], 
                        c=colors[i], linestyle=styles[i], linewidth=1.5, alpha=0.7, label=label)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_zlim(draw_z - 10, air_z + 20)
        ax.view_init(elev=20, azim=-45)
        
        plt.title(f"Structured 4-Stage Path (Rotated 90°)")
        plt.legend()
        plt.show()

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    INPUT_FILE = os.path.join(img_dir, "step05_re_path.json")
    OUTPUT_FILE = os.path.join(img_dir, "step06_full_path.json")
    
    generator = StructuredPathGenerator(INPUT_FILE, OUTPUT_FILE)

    # 👉 接口参数
    DRAW_Z = 130.0  
    AIR_Z = 140.0   
    AIR_MOVE_STEP = 0.5 
    
    generator.generate(DRAW_Z, AIR_Z, AIR_MOVE_STEP)