import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class FullPathGenerator:
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
        
        # 如果距离极小，直接返回终点
        if dist < 1e-3:
            return [p_end.tolist()]
        
        # 计算插值数量
        num_points = int(np.ceil(dist / step_size))
        
        # 生成 t 序列 (不包含起点，因为起点通常是上一段的终点)
        t_values = np.linspace(0, 1, num_points + 1)[1:]
        
        interpolated_points = []
        for t in t_values:
            pt = p_start + (p_end - p_start) * t
            interpolated_points.append(pt.tolist())
            
        return interpolated_points

    def generate(self, draw_z, air_z, air_step_mm):
        print(f"🔄 开始生成完整路径 (含入场轨迹)...")
        print(f"   - 落笔高度: {draw_z} mm")
        print(f"   - 抬笔高度: {air_z} mm")

        # 1. 读取 Step 05 数据
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_strokes = data.get("trajectories", [])
        meta = data.get("meta", {})
        
        if not raw_strokes:
            print("❌ 错误：输入文件中没有轨迹数据")
            return

        full_path_points = []
        vis_segments = [] 

        # ==========================================
        # 🔥 A. 生成入场轨迹 (Approach)
        # ==========================================
        # 从 Step 05 的 meta 中获取图纸原点
        origin_info = meta.get("paper_origin_robot_frame", None)
        
        if origin_info:
            print("✅ 检测到图纸原点信息，正在生成入场路径...")
            
            # 1. 定义关键坐标
            # 起点：图纸原点 (x, y)，高度设为 Air Z (安全高度)
            p_home_air = np.array([origin_info['x'], origin_info['y'], air_z])
            
            # 终点：第一笔画的第一个点
            first_stroke_start = raw_strokes[0][0]
            p_start_air  = np.array([first_stroke_start[0], first_stroke_start[1], air_z])  # 第一笔上方
            p_start_draw = np.array([first_stroke_start[0], first_stroke_start[1], draw_z]) # 第一笔落笔
            
            # 2. 生成插值路径
            # Segment 0-1: 从原点平移到第一笔上方 (Air Move)
            approach_move = self.linear_interpolate(p_home_air, p_start_air, air_step_mm)
            
            # Segment 0-2: 垂直下刀 (Drop) - 这一步必须加，否则机器人会斜着插进纸里
            approach_drop = self.linear_interpolate(p_start_air, p_start_draw, air_step_mm)
            
            # 3. 添加到总路径
            # 注意：我们需要先手动把 p_home_air 加进去作为全路径的绝对起点
            full_path_points.append(p_home_air.tolist()) 
            full_path_points.extend(approach_move)
            full_path_points.extend(approach_drop)
            
            vis_segments.append({'type': 'air', 'points': [p_home_air.tolist()] + approach_move + approach_drop})
        else:
            print("⚠️ 警告：JSON中未包含图纸原点信息，跳过入场轨迹生成。")

        # ==========================================
        # B. 循环处理每一笔 (Draw + Transition)
        # ==========================================
        for i, stroke in enumerate(raw_strokes):
            # --- 1. 画字 (Draw) ---
            current_stroke_points = []
            for pt in stroke:
                current_stroke_points.append([pt[0], pt[1], draw_z])
            
            full_path_points.extend(current_stroke_points)
            vis_segments.append({'type': 'draw', 'points': current_stroke_points})

            # --- 2. 笔画间过渡 (Air Move) ---
            if i < len(raw_strokes) - 1:
                last_pt = np.array(current_stroke_points[-1])
                next_stroke_start = raw_strokes[i+1][0]
                
                # 关键点
                p_lift = np.array([last_pt[0], last_pt[1], air_z])             # 原地抬笔
                p_hover = np.array([next_stroke_start[0], next_stroke_start[1], air_z]) # 下一笔上方
                p_next_draw = np.array([next_stroke_start[0], next_stroke_start[1], draw_z]) # 下一笔落笔
                
                # 生成三段
                seg_lift = self.linear_interpolate(last_pt, p_lift, air_step_mm)
                seg_move = self.linear_interpolate(p_lift, p_hover, air_step_mm)
                seg_drop = self.linear_interpolate(p_hover, p_next_draw, air_step_mm)
                
                full_path_points.extend(seg_lift + seg_move + seg_drop)
                vis_segments.append({'type': 'air', 'points': seg_lift + seg_move + seg_drop})
        
        # ==========================================
        # C. 结束动作 (Final Lift)
        # ==========================================
        # 画完最后一笔后，抬起笔，回到安全高度
        if full_path_points:
            last_pt = np.array(full_path_points[-1])
            if abs(last_pt[2] - draw_z) < 0.1: # 如果停在纸面上
                p_final_lift = np.array([last_pt[0], last_pt[1], air_z])
                seg_final = self.linear_interpolate(last_pt, p_final_lift, air_step_mm)
                full_path_points.extend(seg_final)
                vis_segments.append({'type': 'air', 'points': seg_final})

        # 保存
        output_data = {
            "meta": {
                "source": "step06_full_path",
                "total_points": len(full_path_points),
                "draw_z": draw_z,
                "air_z": air_z,
                "step_mm": air_step_mm,
                "parent_meta": meta
            },
            "path_points": full_path_points 
        }

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2) # 保持良好缩进
            
        print(f"✅ 全路径生成完毕 (含入场与结束)! 总点数: {len(full_path_points)}")
        print(f"💾 文件已保存: {self.output_path}")
        
        self.visualize_3d(vis_segments, draw_z, air_z, origin_info)

    def visualize_3d(self, segments, draw_z, air_z, origin_info):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        print("🎨 正在绘制 3D 预览...")
        
        for i, seg in enumerate(segments):
            pts = np.array(seg['points'])
            if len(pts) < 1: continue
            
            if seg['type'] == 'draw':
                ax.plot(pts[:,0], pts[:,1], pts[:,2], c='blue', linewidth=1.5, alpha=0.8)
            else:
                # 第一段 air 是入场，用绿色虚线区分
                color = 'green' if i == 0 else 'red'
                width = 1.0 if i == 0 else 0.5
                ax.plot(pts[:,0], pts[:,1], pts[:,2], c=color, linewidth=width, alpha=0.5, linestyle='--')

        # 标记原点
        if origin_info:
            ax.scatter(origin_info['x'], origin_info['y'], air_z, c='green', s=50, marker='^', label='Start (Air)')
            ax.scatter(origin_info['x'], origin_info['y'], draw_z, c='green', s=20, marker='x', label='Origin (Paper)')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_zlim(draw_z - 10, air_z + 20)
        ax.view_init(elev=20, azim=-45)
        
        plt.title(f"Full Path\nGreen=Approach, Blue=Draw, Red=Transit")
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
    
    generator = FullPathGenerator(INPUT_FILE, OUTPUT_FILE)

    # 👉 接口参数
    DRAW_Z = 50.0  
    AIR_Z = 70.0   
    AIR_MOVE_STEP = 2.0
    
    generator.generate(DRAW_Z, AIR_Z, AIR_MOVE_STEP)