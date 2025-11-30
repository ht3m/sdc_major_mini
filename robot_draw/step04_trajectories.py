import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pickle
from scipy.interpolate import splprep, splev

class TrajectoryPlanner:
    def __init__(self, graph_path):
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"找不到图数据: {graph_path}")
        
        print(f"[Init] 加载图数据: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
            
        self.strokes = [] 

    def generate_trajectories(self):
        """ 贪婪算法规划路径 """
        print("[Plan] 开始规划路径...")
        
        unread_edges = []
        # ---【核心修复】---
        # 之前这里有 if u != v，导致孤立环路（自环）被过滤了
        # 现在我们要允许 u == v 的情况
        for u, v, data in self.graph.edges(data=True):
            unread_edges.append((u, v, data.get('path', [])))
        # ------------------
        
        visited_indices = set()
        current_pos = None
        
        while len(visited_indices) < len(unread_edges):
            best_edge_idx = -1; best_direction = 1; min_dist = float('inf')
            
            for i in range(len(unread_edges)):
                if i in visited_indices: continue
                u, v, path = unread_edges[i]
                pos_u = np.array(self.graph.nodes[u]['pos'])
                pos_v = np.array(self.graph.nodes[v]['pos'])
                
                if current_pos is None:
                    best_edge_idx = i; best_direction = 1; break
                else:
                    dist_u = np.linalg.norm(current_pos - pos_u)
                    dist_v = np.linalg.norm(current_pos - pos_v)
                    if dist_u < min_dist: min_dist = dist_u; best_edge_idx = i; best_direction = 1
                    if dist_v < min_dist: min_dist = dist_v; best_edge_idx = i; best_direction = -1
            
            if best_edge_idx != -1:
                u, v, raw_path = unread_edges[best_edge_idx]
                visited_indices.add(best_edge_idx)
                
                stroke_pixels = raw_path if raw_path else [self.graph.nodes[u]['pos'], self.graph.nodes[v]['pos']]
                if best_direction == -1: stroke_pixels = stroke_pixels[::-1]
                
                is_connected = False
                if current_pos is not None and self.strokes:
                    last_stroke_end = self.strokes[-1][-1]
                    start_pt = stroke_pixels[0]
                    if np.linalg.norm(np.array(last_stroke_end) - np.array(start_pt)) < 2.0:
                        self.strokes[-1].extend(stroke_pixels[1:])
                        is_connected = True
                
                if not is_connected: self.strokes.append(stroke_pixels)
                current_pos = np.array(stroke_pixels[-1])
                
                while True:
                    curr_node_id = v if best_direction == 1 else u
                    found_next = False
                    for i in range(len(unread_edges)):
                        if i in visited_indices: continue
                        nu, nv, npath = unread_edges[i]
                        
                        # 处理普通连接
                        if nu == curr_node_id: 
                            visited_indices.add(i)
                            new = npath if npath else [self.graph.nodes[nu]['pos'], self.graph.nodes[nv]['pos']]
                            self.strokes[-1].extend(new[1:]) 
                            current_pos = np.array(new[-1])
                            u, v = nu, nv; best_direction = 1; found_next = True; break
                        elif nv == curr_node_id: 
                            visited_indices.add(i)
                            new = npath if npath else [self.graph.nodes[nu]['pos'], self.graph.nodes[nv]['pos']]
                            self.strokes[-1].extend(new[::-1][1:]) 
                            current_pos = np.array(new[::-1][-1])
                            u, v = nu, nv; best_direction = -1; found_next = True; break
                    if not found_next: break

        print(f"[Plan] 路径规划完成，共 {len(self.strokes)} 条笔画")

    def _split_at_sharp_corners(self, stroke, angle_threshold=120):
        """ 检测急转弯并断开 """
        if len(stroke) < 3: return [stroke]
        sub_strokes = []
        current_sub = [stroke[0]]
        pts = np.array(stroke)
        
        keep = [True] * len(pts)
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[i-1]) < 0.1: keep[i] = False
        pts = pts[keep]
        
        if len(pts) < 3: return [pts.tolist()]

        for i in range(1, len(pts) - 1):
            p_prev = pts[i-1]
            p_curr = pts[i]
            p_next = pts[i+1]
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 20.0 or norm2 > 20.0:
                sub_strokes.append(current_sub)
                current_sub = [p_curr.tolist()] 
                continue

            if norm1 < 0.1 or norm2 < 0.1: 
                current_sub.append(p_curr.tolist()); continue
                
            cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            if angle > angle_threshold:
                current_sub.append(p_curr.tolist())
                sub_strokes.append(current_sub)
                current_sub = [p_curr.tolist()] 
            else:
                current_sub.append(p_curr.tolist())
                
        current_sub.append(pts[-1].tolist())
        sub_strokes.append(current_sub)
        return sub_strokes

    def smooth_and_resample(self, smoothing=5.0, step_size=2.0):
        print(f"[Smooth] 锐角打断 -> B样条平滑 (s={smoothing}) -> 重采样...")
        final_strokes = []
        for stroke in self.strokes:
            sub_segments = self._split_at_sharp_corners(stroke, angle_threshold=90)
            for seg in sub_segments:
                if len(seg) < 3: final_strokes.append(seg); continue
                pts = np.array(seg).T
                try:
                    tck, u = splprep(pts, k=3, s=smoothing)
                    u_fine = np.linspace(0, 1, len(seg) * 5)
                    x_fine, y_fine = splev(u_fine, tck)
                    fine_points = np.vstack((x_fine, y_fine)).T
                    dists = np.sqrt(np.sum(np.diff(fine_points, axis=0)**2, axis=1))
                    cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
                    total_length = cumulative_dist[-1]
                    num_points = int(total_length / step_size)
                    if num_points < 2: num_points = 2
                    target_dists = np.linspace(0, total_length, num_points)
                    target_us = np.interp(target_dists, cumulative_dist, u_fine)
                    new_x, new_y = splev(target_us, tck)
                    final_strokes.append(list(zip(new_x, new_y)))
                except Exception as e:
                    final_strokes.append(seg)
        self.strokes = final_strokes

    def export_json(self, output_path, canvas_width_mm=200):
        print(f"[Export] 导出 JSON (宽: {canvas_width_mm}mm)...")
        all_points = [p for s in self.strokes for p in s]
        if not all_points: return
        all_points = np.array(all_points)
        min_x, max_x = np.min(all_points[:,0]), np.max(all_points[:,0])
        min_y, max_y = np.min(all_points[:,1]), np.max(all_points[:,1])
        pixel_width = max_x - min_x
        if pixel_width == 0: pixel_width = 1
        scale = canvas_width_mm / pixel_width
        
        json_data = {
            "meta": {"total_strokes": len(self.strokes), "scale_factor": scale, "canvas_width_mm": canvas_width_mm},
            "trajectories": []
        }
        for stroke in self.strokes:
            traj = []
            for x, y in stroke:
                world_x = (x - min_x) * scale
                world_y = (max_y - y) * scale 
                traj.append([round(world_x, 3), round(world_y, 3)])
            json_data["trajectories"].append(traj)
            
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"✅ JSON 已保存至: {output_path}")

    def visualize_order(self, save_path=None):
        """ 
        可视化：使用高对比度颜色区分笔画，并显示空走轨迹
        """
        plt.figure(figsize=(12, 12))
        plt.title(f"Robot Path | Solid: Draw | Dashed: Air Move | Strokes: {len(self.strokes)}")
        plt.axis('equal')
        
        # 高对比度颜色池 (循环使用)
        colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
                  '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
                  '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
                  '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']
        
        for i, stroke in enumerate(self.strokes):
            pts = np.array(stroke)
            if len(pts) == 0: continue
            
            # 1. 选颜色
            c = colors[i % len(colors)]
            
            # 2. 画实线 (机械臂在纸上画)
            plt.plot(pts[:, 0], pts[:, 1], color=c, linewidth=2, label=f"Stroke {i}")
            
            # 3. 标起点 (圆点)
            plt.scatter(pts[0,0], pts[0,1], color=c, s=25, zorder=5)
            
            # 4. 画空走轨迹 (灰色虚线) - 模拟机械臂抬笔移动
            if i > 0:
                prev_stroke = self.strokes[i-1]
                if len(prev_stroke) > 0:
                    prev_end = prev_stroke[-1]
                    curr_start = pts[0]
                    # 画一条灰色的虚线连接上一笔终点和这一笔起点
                    plt.plot([prev_end[0], curr_start[0]], [prev_end[1], curr_start[1]], 
                             color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        plt.gca().invert_yaxis()
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 路径预览图已保存至: {save_path}")
        plt.show()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    GRAPH_FILE = os.path.join(img_dir, "graph_data.pkl")
    JSON_FILE = os.path.join(img_dir, "robot_paths.json")
    OUTPUT_VIS_FILE = os.path.join(img_dir, "step04_path_preview.png")
    
    try:
        planner = TrajectoryPlanner(GRAPH_FILE)
        
        # 1. 规划
        planner.generate_trajectories()
        
        # 2. 智能平滑
        planner.smooth_and_resample(smoothing=5.0, step_size=2.0)
        
        # 3. 导出
        planner.export_json(JSON_FILE, canvas_width_mm=300)
        
        # 4. 可视化
        planner.visualize_order(save_path=OUTPUT_VIS_FILE)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()