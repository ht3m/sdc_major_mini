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
        
        # 如果距离极小，直接返回终点，保证至少有一个点
        if dist < 1e-3:
            return [p_end.tolist()]
        
        # 计算插值数量
        num_points = int(np.ceil(dist / step_size))
        
        # 生成 t 序列
        # 注意：包含终点。通常线性插值不包含起点以避免重复，
        # 但既然您允许点重合，为了保证每段轨迹的完整性，我们生成完整的段。
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

        # 获取图纸原点，如果没有则默认为第一笔的起点位置(Air Z)
        origin_info = meta.get("paper_origin_robot_frame", None)
        
        # 初始化“当前机械臂在空中的位置”
        if origin_info:
            current_air_pos = np.array([origin_info['x'], origin_info['y'], air_z])
            print(f"✅ 起点设置为图纸原点: {current_air_pos}")
        else:
            first_pt = raw_strokes[0][0]
            current_air_pos = np.array([first_pt[0], first_pt[1], air_z])
            print(f"⚠️ 未找到原点信息，起点设置为第一笔上方: {current_air_pos}")

        # 结果容器：[Stroke1[Move, Drop, Draw, Lift], Stroke2[...], ...]
        structured_strokes = [] 

        total_points_count = 0

        # ==========================================
        # 循环处理每一笔，生成标准的 4 段式结构
        # ==========================================
        for i, stroke in enumerate(raw_strokes):
            stroke_segments = [] # 存放当前笔画的 4 个阶段
            
            # --- 关键点定义 ---
            stroke_start_2d = stroke[0]
            stroke_end_2d = stroke[-1]

            # P_Hover_Start: 这一笔的开始上方
            p_hover_start = np.array([stroke_start_2d[0], stroke_start_2d[1], air_z])
            # P_Start: 这一笔的落笔点
            p_draw_start  = np.array([stroke_start_2d[0], stroke_start_2d[1], draw_z])
            # P_End: 这一笔的结束点
            p_draw_end    = np.array([stroke_end_2d[0], stroke_end_2d[1], draw_z])
            # P_Hover_End: 这一笔的结束上方
            p_hover_end   = np.array([stroke_end_2d[0], stroke_end_2d[1], air_z])

            # --- 阶段 1: Air Move (Move to Approach) ---
            # 从“上一次的空中位置”移动到“当前笔画的上方”
            seg_move = self.linear_interpolate(current_air_pos, p_hover_start, air_step_mm)
            stroke_segments.append(seg_move)

            # --- 阶段 2: Drop (Vertical Down) ---
            # 从“上方”垂直下降到“落笔点”
            seg_drop = self.linear_interpolate(p_hover_start, p_draw_start, air_step_mm)
            stroke_segments.append(seg_drop)

            # --- 阶段 3: Draw (The actual stroke) ---
            # 这一步比较特殊，因为 stroke 本身已经是点集了
            # 但我们需要确保 Z 轴正确，并且如果点太稀疏也需要插值（这里假设Step05已重采样，只修正Z）
            current_draw_points = []
            for pt in stroke:
                current_draw_points.append([pt[0], pt[1], draw_z])
            stroke_segments.append(current_draw_points)

            # --- 阶段 4: Lift (Vertical Up) ---
            # 从“结束点”垂直抬起到“上方”
            seg_lift = self.linear_interpolate(p_draw_end, p_hover_end, air_step_mm)
            stroke_segments.append(seg_lift)

            # 将这 4 段加入总列表
            structured_strokes.append(stroke_segments)
            
            # 更新状态：当前的空中位置变成了这一笔结束后的上方
            current_air_pos = p_hover_end

            # 统计点数
            total_points_count += len(seg_move) + len(seg_drop) + len(current_draw_points) + len(seg_lift)

        # 保存
        output_data = {
            "meta": {
                "source": "step06_structured_path",
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
        
        # 颜色映射：Move(绿), Drop(黄), Draw(蓝), Lift(红)
        colors = ['green', 'orange', 'blue', 'red']
        styles = ['--', '--', '-', '--']
        labels = ['Move', 'Drop', 'Draw', 'Lift']
        added_labels = set()

        for stroke_segs in structured_data:
            # stroke_segs 包含 4 个 list
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
        
        plt.title(f"Structured 4-Stage Path")
        plt.legend()
        plt.show()

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    # 注意：这里读取的是 Step 05 的输出
    INPUT_FILE = os.path.join(img_dir, "step05_re_path.json")
    OUTPUT_FILE = os.path.join(img_dir, "step06_full_path.json")
    
    generator = StructuredPathGenerator(INPUT_FILE, OUTPUT_FILE)

    # 👉 接口参数
    DRAW_Z = 120.0  
    AIR_Z = 140.0   
    AIR_MOVE_STEP = 0.5 # 采样更密，适应后续运动学要求
    
    generator.generate(DRAW_Z, AIR_Z, AIR_MOVE_STEP)