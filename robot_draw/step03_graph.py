import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from scipy.spatial import cKDTree

class GraphBuilder:
    def __init__(self, skeleton_path):
        if not os.path.exists(skeleton_path):
            raise FileNotFoundError(f"找不到骨架文件: {skeleton_path}")
            
        self.img = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
        _, self.img = cv2.threshold(self.img, 127, 255, cv2.THRESH_BINARY)
        self.skel_bool = (self.img > 0).astype(np.uint8)
        self.h, self.w = self.skel_bool.shape
        
        self.graph = nx.Graph()
        self.visited_mask = np.zeros_like(self.skel_bool, dtype=np.uint8)

    def build_graph(self):
        print("[Info] 1. 正在识别关键节点...")
        nodes_idx = self._find_potential_nodes()
        
        for idx, (y, x) in enumerate(nodes_idx):
            node_id = f"raw_{idx}"
            self.graph.add_node(node_id, pos=(x, y), type="potential")
            self.visited_mask[y, x] = 1

        print(f"[Info]    找到 {len(nodes_idx)} 个常规节点，开始追踪...")
        self._trace_paths(nodes_idx)
        
        print("[Info] 3. 正在扫描漏网的孤立环路...")
        self._find_and_trace_loops()

        print(f"[Info]    图构建完成: {self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边")
        return self.graph

    def merge_close_nodes(self, distance_threshold=15.0):
        """
        [修正版] 延迟删除策略
        """
        print(f"[Info] 4. 正在合并距离 < {distance_threshold} 的节点...")
        
        nodes = list(self.graph.nodes(data=True))
        if not nodes: return

        node_ids = [n[0] for n in nodes]
        coords = [n[1]['pos'] for n in nodes]
        
        tree = cKDTree(coords)
        pairs = tree.query_pairs(distance_threshold)
        
        merge_graph = nx.Graph()
        merge_graph.add_nodes_from(node_ids)
        merge_graph.add_edges_from([(node_ids[i], node_ids[j]) for i, j in pairs])
        
        groups = list(nx.connected_components(merge_graph))
        nodes_to_remove = [] # 待删除列表
        
        for group in groups:
            if len(group) < 2: continue
            
            group_list = list(group)
            
            # 计算新中心
            coords_arr = np.array([self.graph.nodes[n]['pos'] for n in group_list])
            cx, cy = np.mean(coords_arr, axis=0)
            
            new_node_id = f"merged_{group_list[0]}"
            self.graph.add_node(new_node_id, pos=(cx, cy), type="junction")
            
            # 转移连线
            for old_node in group_list:
                nodes_to_remove.append(old_node) # 标记删除
                
                edges = list(self.graph.edges(old_node, data=True))
                for _, neighbor, data in edges:
                    if neighbor in group_list: continue # 忽略内部连线
                    
                    # 建立新连接，保留原 Path
                    # 防止覆盖：保留较长的路径
                    if self.graph.has_edge(new_node_id, neighbor):
                        old_len = self.graph[new_node_id][neighbor].get('weight', 0)
                        curr_len = data.get('weight', 0)
                        if curr_len > old_len:
                            self.graph.add_edge(new_node_id, neighbor, weight=curr_len, path=data.get('path'))
                    else:
                        self.graph.add_edge(new_node_id, neighbor, weight=data.get('weight', 0), path=data.get('path'))

        # 统一执行删除
        self.graph.remove_nodes_from(nodes_to_remove)
        print(f"[Info]    合并完成，当前剩余: {self.graph.number_of_nodes()} 节点")

    def salvage_missing_segments(self, min_length=15):
        """
        【兜底大招 - 像素级修复版】
        逻辑：
        1. 找出遗漏的像素连通域。
        2. 对乱序的像素点进行【几何排序】，整理成连贯路径。
        3. 将整理好的路径挂载到最近的节点。
        """
        print("[Salvage] 正在扫描并挽救遗失的路径 (保留完整像素)...")
        
        # 1. 绘制当前 Graph 的覆盖范围
        covered_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        for u, v, data in self.graph.edges(data=True):
            path = data.get('path', [])
            if len(path) > 1:
                pts = np.array(path, np.int32).reshape((-1, 1, 2))
                cv2.polylines(covered_mask, [pts], False, 255, thickness=2)
        
        for node, data in self.graph.nodes(data=True):
            nx, ny = int(data['pos'][0]), int(data['pos'][1])
            cv2.circle(covered_mask, (nx, ny), 3, 255, -1)

        # 2. 寻找遗漏像素
        missing_pixels = cv2.bitwise_and(self.skel_bool, self.skel_bool, mask=cv2.bitwise_not(covered_mask))
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(missing_pixels, connectivity=8)
        
        # 准备 KDTree
        all_nodes = list(self.graph.nodes(data=True))
        if not all_nodes: return
        node_coords = [n[1]['pos'] for n in all_nodes]
        node_ids = [n[0] for n in all_nodes]
        tree = cKDTree(node_coords)
        
        salvaged_count = 0
        
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < min_length: continue
            
            # 获取线段像素 (y, x) -> (x, y)
            ys, xs = np.where(labels == i)
            segment_xy = [ (x, y) for x, y in zip(xs, ys) ]
            
            # 找离现有图最近的挂载点
            # 这里只检查线段中的一个点来定大概位置即可
            sample_pt = segment_xy[0] 
            dists, idxs = tree.query(sample_pt)
            
            # 为了更准，可以检查所有点离哪个节点最近
            # 但通常一段遗失的线段只会附着在一个节点附近
            # 我们遍历一下 segment 里的点，找到离现有节点最近的那个点作为“接触点”
            
            min_global_dist = float('inf')
            nearest_node_idx = -1
            contact_pixel_idx = -1
            
            # 这是一个 N*1 的查询，很快
            segment_arr = np.array(segment_xy)
            dists, idxs = tree.query(segment_arr)
            
            best_idx_in_segment = np.argmin(dists)
            min_dist = dists[best_idx_in_segment]
            nearest_node_id = node_ids[idxs[best_idx_in_segment]]
            nearest_node_pos = node_coords[idxs[best_idx_in_segment]] # (x,y)
            
            # 阈值判断
            #TODO
            if min_dist < 5.0:
                # --- 情况 A: 这是一个分支/尾巴 ---
                # 1. 整理像素顺序：从“接触点”开始，理顺整条线
                # 接触点就是 segment 中离节点最近的那个点
                contact_pixel = segment_xy[best_idx_in_segment]
                
                # 调用排序算法，得到有序路径
                sorted_path = self._sort_pixels_to_path(segment_xy, start_point=contact_pixel)
                
                # 2. 确定新端点 (路径的最后一个点)
                end_pt = sorted_path[-1]
                
                new_id = f"salvaged_{salvaged_count}"
                self.graph.add_node(new_id, pos=end_pt, type="endpoint")
                
                # 3. 添加边
                # 路径 = [节点位置] + [整理好的像素点]
                # 这样保证了视觉上的连续性
                full_path = [tuple(nearest_node_pos)] + sorted_path
                
                self.graph.add_edge(nearest_node_id, new_id, weight=len(full_path), path=full_path)
                salvaged_count += 1
                
            else:
                # --- 情况 B: 孤立环路 ---
                # 1. 随便选个起点排序
                start_pixel = segment_xy[0]
                sorted_path = self._sort_pixels_to_path(segment_xy, start_point=start_pixel)
                
                # 2. 闭合
                sorted_path.append(sorted_path[0])
                
                loop_id = f"salvaged_loop_{salvaged_count}"
                self.graph.add_node(loop_id, pos=sorted_path[0], type="loop_start")
                self.graph.add_edge(loop_id, loop_id, weight=len(sorted_path), path=sorted_path)
                salvaged_count += 1
            
        print(f"[Salvage] 成功挽救并整理了 {salvaged_count} 条路径")

    def _sort_pixels_to_path(self, pixels, start_point):
        """
        【辅助函数】将一袋子乱序像素整理成一条连贯路径
        算法：最近邻贪婪搜索 (Nearest Neighbor)
        """
        # 转为 list 方便移除
        remaining = set(pixels) # 使用 set 加速查找
        
        # 结果列表
        path = []
        
        # 当前点
        current = start_point
        if current in remaining:
            remaining.remove(current)
        path.append(current)
        
        while remaining:
            # 在剩余点中找离 current 最近的点
            # 为了性能优化，我们先只搜 3x3 邻域 (绝大多数情况都在邻域内)
            cx, cy = current
            found = None
            
            # 8邻域快速查找
            neighbors = [
                (cx-1, cy-1), (cx, cy-1), (cx+1, cy-1),
                (cx-1, cy),               (cx+1, cy),
                (cx-1, cy+1), (cx, cy+1), (cx+1, cy+1)
            ]
            
            for nb in neighbors:
                if nb in remaining:
                    found = nb
                    break
            
            if not found:
                # 如果邻域没找到（可能有断裂），则暴力搜全局最近
                # 这种情况很少，但在 salvage 模式下可能发生
                rem_list = list(remaining)
                rem_arr = np.array(rem_list)
                curr_arr = np.array([current])
                # 计算距离
                dists = np.linalg.norm(rem_arr - curr_arr, axis=1)
                min_idx = np.argmin(dists)
                found = rem_list[min_idx]
            
            # 移动到下一个点
            path.append(found)
            remaining.remove(found)
            current = found
            
        return path

    def _find_potential_nodes(self):
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbors_count = cv2.filter2D(self.skel_bool, -1, kernel) * self.skel_bool
        node_mask = (neighbors_count == 1) | (neighbors_count >= 3)
        y_idxs, x_idxs = np.where(node_mask > 0)
        return list(zip(y_idxs, x_idxs))

    def _trace_paths(self, start_nodes_coords):
        node_set = set(start_nodes_coords)
        visited_edges = set()
        
        for start_idx, (y, x) in enumerate(start_nodes_coords):
            start_node_id = f"raw_{start_idx}"
            neighbors = [
                (y-1, x-1), (y-1, x), (y-1, x+1),
                (y, x-1),             (y, x+1),
                (y+1, x-1), (y+1, x), (y+1, x+1)
            ]
            for ny, nx in neighbors:
                if not (0 <= ny < self.h and 0 <= nx < self.w): continue
                if self.skel_bool[ny, nx] == 0: continue
                if (ny, nx) in node_set: continue # 稍后 merge 会合并
                if self.visited_mask[ny, nx] == 1: continue

                path_nodes = self._walk_path(y, x, ny, nx, node_set)
                
                if path_nodes:
                    end_y, end_x = path_nodes[-1]
                    end_node_id = None
                    for i, (ny_node, nx_node) in enumerate(start_nodes_coords):
                        if ny_node == end_y and nx_node == end_x:
                            end_node_id = f"raw_{i}"
                            break
                    if end_node_id:
                        edge_key = tuple(sorted((start_node_id, end_node_id)))
                        if edge_key not in visited_edges:
                            path_xy = [(c[1], c[0]) for c in path_nodes] 
                            full_path = [(x, y)] + path_xy
                            self.graph.add_edge(start_node_id, end_node_id, weight=len(full_path), path=full_path)
                            visited_edges.add(edge_key)

    def _find_and_trace_loops(self):
        unvisited = (self.skel_bool == 1) & (self.visited_mask == 0)
        y_idxs, x_idxs = np.where(unvisited)
        loop_id_counter = 0
        while len(y_idxs) > 0:
            curr_y, curr_x = y_idxs[0], x_idxs[0]
            loop_node_id = f"loop_{loop_id_counter}"
            self.graph.add_node(loop_node_id, pos=(curr_x, curr_y), type="loop_start")
            self.visited_mask[curr_y, curr_x] = 1
            neighbors = [
                (curr_y-1, curr_x-1), (curr_y-1, curr_x), (curr_y-1, curr_x+1),
                (curr_y, curr_x-1),                       (curr_y, curr_x+1),
                (curr_y+1, curr_x-1), (curr_y+1, curr_x), (curr_y+1, curr_x+1)
            ]
            for ny, nx in neighbors:
                if 0 <= ny < self.h and 0 <= nx < self.w:
                    if self.skel_bool[ny, nx] == 1 and self.visited_mask[ny, nx] == 0:
                        target_set = {(curr_y, curr_x)} 
                        path_nodes = self._walk_path(curr_y, curr_x, ny, nx, target_set)
                        if path_nodes:
                            path_xy = [(c[1], c[0]) for c in path_nodes]
                            full_path = [(curr_x, curr_y)] + path_xy
                            self.graph.add_edge(loop_node_id, loop_node_id, weight=len(full_path), path=full_path)
                            break 
            loop_id_counter += 1
            unvisited = (self.skel_bool == 1) & (self.visited_mask == 0)
            y_idxs, x_idxs = np.where(unvisited)

    def _walk_path(self, start_y, start_x, first_step_y, first_step_x, stop_set):
        path = [(first_step_y, first_step_x)]
        self.visited_mask[first_step_y, first_step_x] = 1
        prev_y, prev_x = start_y, start_x
        curr_y, curr_x = first_step_y, first_step_x
        while True:
            if (curr_y, curr_x) in stop_set: return path
            next_steps = []
            neighbors = [
                (curr_y-1, curr_x-1), (curr_y-1, curr_x), (curr_y-1, curr_x+1),
                (curr_y, curr_x-1),                       (curr_y, curr_x+1),
                (curr_y+1, curr_x-1), (curr_y+1, curr_x), (curr_y+1, curr_x+1)
            ]
            for ny, nx in neighbors:
                if 0 <= ny < self.h and 0 <= nx < self.w:
                    if self.skel_bool[ny, nx] == 1 and (ny, nx) != (prev_y, prev_x):
                        is_target = (ny, nx) in stop_set
                        if is_target or self.visited_mask[ny, nx] == 0:
                            next_steps.append((ny, nx))
            if not next_steps: return None 
            next_y, next_x = next_steps[0]
            for ny, nx in next_steps:
                if (ny, nx) in stop_set:
                    next_y, next_x = ny, nx; break
            prev_y, prev_x = curr_y, curr_x
            curr_y, curr_x = next_y, next_x
            path.append((curr_y, curr_x))
            if (curr_y, curr_x) not in stop_set: self.visited_mask[curr_y, curr_x] = 1
            else: return path

    def prune_spurs(self, min_length=20):
        """
        【剪毛刺】迭代删除短小的死胡同分支
        :param min_length: 小于这个长度的端点分支会被剪掉
        """
        print(f"[Pruning] 正在修剪长度 < {min_length} 的毛刺...")
        total_removed = 0
        
        while True:
            found_spur = False
            # 获取当前所有度为1的节点（端点）
            # 转为list因为我们在迭代中要修改图
            nodes = list(self.graph.nodes())
            
            for node in nodes:
                if node not in self.graph: continue # 可能已被删
                
                # 度为1 = 端点
                if self.graph.degree(node) == 1:
                    # 找到它的邻居
                    neighbor = list(self.graph.neighbors(node))[0]
                    
                    # 获取边长
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    weight = edge_data.get('weight', 0)
                    
                    # 只有当边太短，且邻居不是另一个端点（防止把孤立的一根短线删没了）时才删
                    # 也就是说，只删附着在主干上的小刺
                    if weight < min_length and self.graph.degree(neighbor) > 1:
                        self.graph.remove_node(node)
                        found_spur = True
                        total_removed += 1
            
            # 如果这一轮没剪掉任何东西，说明干净了
            if not found_spur:
                break
                
        print(f"   -> 共修剪了 {total_removed} 个毛刺节点")

    def remove_small_components(self, min_nodes=3, min_total_length=30.0):
        """
        【移除孤立噪点】
        Salvage 步骤可能会引入一些只有2个节点的微小线段（灰尘），
        如果它们没连在主图上，就删掉。
        """
        print(f"[Cleaning] 正在移除节点数 < {min_nodes} 的孤立连通分量...")
        
        # 获取所有连通分量
        components = list(nx.connected_components(self.graph))
        removed_count = 0
        
        for comp in components:
            group_list = list(comp)
            if len(group_list) < min_nodes:
                # 计算这个小分量的总线长
                total_length = 0.0
                # 构建一个子图来方便计算边
                subgraph = self.graph.subgraph(group_list)
                for u, v, data in subgraph.edges(data=True):
                    total_length += data.get('weight', 0)
                    
                # 只有节点少、且长度也短的，才是真噪点
                if total_length < min_total_length:
                    for node in group_list:
                        self.graph.remove_node(node)
                    removed_count += 1
                
        print(f"   -> 移除了 {removed_count} 个孤立小分量")
            
    def visualize(self, title="Graph", save_path=None):
        plt.figure(figsize=(12, 12))
        plt.title(title)
        plt.imshow(1 - self.skel_bool, cmap='gray')
        
        for u, v, data in self.graph.edges(data=True):
            path = data.get('path', [])
            if path:
                path = np.array(path)
                plt.plot(path[:, 0], path[:, 1], color='orange', linewidth=2, alpha=0.7)
        
        for node, data in self.graph.nodes(data=True):
            x, y = data['pos']
            ntype = data.get('type')
            color = 'red' if ntype == 'junction' else ('green' if ntype == 'loop_start' else 'blue')
            if ntype == 'endpoint' and node.startswith('salvaged'): color = 'cyan' # 区分挽救回来的端点
            plt.scatter(x, y, c=color, s=40, zorder=10, edgecolors='white')
            
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Info] 可视化结果已保存至: {save_path}")
        plt.show()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    INPUT_FILE = os.path.join(img_dir, "step02_skeleton_img.png") # 注意确认你的输入文件名
    OUTPUT_VIS_FILE = os.path.join(img_dir, "step03_final_graph.png")

    try:
        builder = GraphBuilder(INPUT_FILE)
        
        # 1. 构建基础图 (追踪+环路)
        builder.build_graph() 
        
        # 2. 合并节点 (使用延迟删除，避免断链)
        builder.merge_close_nodes(distance_threshold=15.0) 
        
        # 3. 兜底扫描 (挽救合并或追踪中遗失的路径)
        builder.salvage_missing_segments(min_length=15)

        # 4. 美容 (修剪毛刺 & 去噪) ---
        builder.prune_spurs(min_length=20) # 小于20像素的短枝剪掉
        builder.remove_small_components(min_nodes=3, min_total_length=20) # 极小的孤立噪点删掉
        
        # --- 5. 结果输出 ---
        builder.visualize("Final Optimized Graph", save_path=OUTPUT_VIS_FILE)
        
        import pickle
        pkl_path = os.path.join(img_dir, "graph_data.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(builder.graph, f)
        print(f"✅ 图数据已保存至: {pkl_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()