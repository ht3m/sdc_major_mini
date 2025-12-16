import json
import numpy as np
import os
from scipy.interpolate import interp1d

class TrajectoryOptimizer:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        
        # ==========================================
        # 1. Jaka Mini 2 物理极限
        # ==========================================
        self.N = 6
        self.PHYSICAL_MAX_VEL = np.pi 
        self.PHYSICAL_MAX_ACC = np.pi * 4.0 
        
        # 保持 20% 的极柔和加速度限制
        self.LIMIT_VEL = self.PHYSICAL_MAX_VEL * 0.5 
        self.LIMIT_ACC = self.PHYSICAL_MAX_ACC * 0.2 
        
        self.FREQ = 125.0
        self.DT = 1.0 / self.FREQ
        
        # ==========================================
        # 2. 停顿配置 (保持 1.0s)
        # ==========================================
        self.WAIT_START = 1.0   
        self.WAIT_MOVE  = 1.0   
        self.WAIT_DROP  = 1.0   
        self.WAIT_DRAW  = 1.0   
        self.WAIT_LIFT  = 1.0   
        self.WAIT_END   = 1.0   
        
        # ==========================================
        # 3. 速度配置 (极致慢速)
        # ==========================================
        # 🔥 修改点：运笔速度减慢一半 (0.3 -> 0.15)
        self.SCALE_AIR_VEL = 0.15 
        
        # 写字速度保持 0.1 rad/s (极慢)
        self.CONSTANT_DRAW_VEL = 0.1 

    def generate_bang_bang_profile_smoothed(self, path_points, vel_scale=1.0, label=""):
        """
        升级版梯形规划 (Move/Drop/Lift)
        包含：累积路程计算 + Cubic 平滑
        """
        path_points_np = np.array(path_points)
        count = len(path_points_np)
        if count < 2: return path_points_np.tolist()
            
        # 1. 累积路程 (Cumulative Distance)
        diffs = np.abs(np.diff(path_points_np, axis=0))
        total_dist_per_joint = np.sum(diffs, axis=0)
        max_dist = np.max(total_dist_per_joint)
        
        if max_dist < 1e-6: return [path_points_np[0].tolist()]
            
        # 2. 规划时间
        acc_limit = self.LIMIT_ACC
        vel_limit = self.LIMIT_VEL * vel_scale
        
        d_acc = (vel_limit ** 2) / (2.0 * acc_limit)
        
        if max_dist < 2 * d_acc:
            t_acc = np.sqrt(max_dist / acc_limit)
            t_flat = 0.0
            total_time = 2 * t_acc
        else:
            t_acc = vel_limit / acc_limit
            d_flat = max_dist - 2 * d_acc
            t_flat = d_flat / vel_limit
            total_time = 2 * t_acc + t_flat
            
        # 3. 重采样
        num_steps = int(np.ceil(total_time / self.DT))
        if num_steps < 2: num_steps = 2
        
        # 调试打印
        # if label: print(f"   [{label}] Dist={max_dist:.4f}, Time={total_time:.2f}s")

        time_grid = np.linspace(0, total_time, num_steps)
        s_t = np.zeros_like(time_grid)
        
        for i, t in enumerate(time_grid):
            if t <= t_acc:
                s_t[i] = 0.5 * acc_limit * t**2
            elif t <= t_acc + t_flat:
                v_peak = acc_limit * t_acc
                d_done = 0.5 * acc_limit * t_acc**2
                s_t[i] = d_done + v_peak * (t - t_acc)
            else:
                t_dec = total_time - t
                s_t[i] = max_dist - 0.5 * acc_limit * t_dec**2
                
        # 4. 插值 (Cubic 平滑拐弯)
        original_indices = np.linspace(0, 1, count)
        target_progress = s_t / max_dist
        kind_type = 'cubic' if count >= 4 else 'linear'
        
        try:
            interpolator = interp1d(original_indices, path_points_np, axis=0, kind=kind_type)
            result = interpolator(target_progress)
        except:
            interpolator = interp1d(original_indices, path_points_np, axis=0, kind='linear')
            result = interpolator(target_progress)
            
        return result.tolist()

    def generate_constant_velocity_profile_smoothed(self, path_points, const_vel, label=""):
        """
        恒定速度 + 平滑 (Draw)
        """
        path_points_np = np.array(path_points)
        count = len(path_points_np)
        if count < 2: return path_points_np.tolist()

        # 1. 累积路程
        diffs = np.abs(np.diff(path_points_np, axis=0))
        total_dist_per_joint = np.sum(diffs, axis=0)
        max_dist = np.max(total_dist_per_joint)
        
        if max_dist < 1e-6: return [path_points_np[0].tolist()]

        # 2. 计算时间
        total_time = max_dist / const_vel
        if total_time < 0.5: total_time = 0.5 
        
        num_steps = int(np.ceil(total_time / self.DT))
        if num_steps < 20: num_steps = 20
        
        if label:
            print(f"   [{label}] MaxJointTravel={max_dist:.4f} rad, PlanTime={total_time:.2f} s")
        
        # 3. 插值
        original_indices = np.linspace(0, 1, count)
        target_indices = np.linspace(0, 1, num_steps)
        kind_type = 'cubic' if count >= 4 else 'linear'
        
        try:
            interpolator = interp1d(original_indices, path_points_np, axis=0, kind=kind_type)
            result = interpolator(target_indices)
        except Exception as e:
            print(f"⚠️ 插值降级: {e}")
            interpolator = interp1d(original_indices, path_points_np, axis=0, kind='linear')
            result = interpolator(target_indices)
            
        return result.tolist()

    def create_wait_frames(self, joint_pose, duration):
        if isinstance(joint_pose, np.ndarray): joint_pose = joint_pose.tolist()
        count = int(duration / self.DT)
        if count < 1: return []
        return [joint_pose] * count

    def optimize(self):
        print(f"🚀 开始优化 (运笔减速版)...")
        print(f"   - 运笔倍率: {self.SCALE_AIR_VEL} (极慢)")
        print(f"   - 画画速度: {self.CONSTANT_DRAW_VEL} rad/s")
        print(f"   - 停顿时间: 1.0s")
        
        if not os.path.exists(self.input_path):
            print(f"❌ 找不到文件: {self.input_path}")
            return

        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        structured_joints = data.get("structured_joints", [])
        original_meta = data.get("meta", {})
        
        final_trajectory = []
        
        # 1. Start Wait
        start_pose = structured_joints[0][0][0]
        final_trajectory.extend(self.create_wait_frames(start_pose, self.WAIT_START))
        
        # 2. Process Strokes
        for s_idx, stroke_segments in enumerate(structured_joints):
            # A. Move
            traj_move = self.generate_bang_bang_profile_smoothed(stroke_segments[0], self.SCALE_AIR_VEL, f"Stroke {s_idx} Move")
            final_trajectory.extend(traj_move)
            final_trajectory.extend(self.create_wait_frames(traj_move[-1], self.WAIT_MOVE))

            # B. Drop
            traj_drop = self.generate_bang_bang_profile_smoothed(stroke_segments[1], self.SCALE_AIR_VEL, f"Stroke {s_idx} Drop")
            final_trajectory.extend(traj_drop)
            final_trajectory.extend(self.create_wait_frames(traj_drop[-1], self.WAIT_DROP))
            
            # C. Draw
            traj_draw = self.generate_constant_velocity_profile_smoothed(stroke_segments[2], self.CONSTANT_DRAW_VEL, f"Stroke {s_idx} Draw")
            final_trajectory.extend(traj_draw)
            final_trajectory.extend(self.create_wait_frames(traj_draw[-1], self.WAIT_DRAW))
            
            # D. Lift
            traj_lift = self.generate_bang_bang_profile_smoothed(stroke_segments[3], self.SCALE_AIR_VEL, f"Stroke {s_idx} Lift")
            final_trajectory.extend(traj_lift)
            final_trajectory.extend(self.create_wait_frames(traj_lift[-1], self.WAIT_LIFT))

        # 3. End Wait
        end_pose = final_trajectory[-1]
        final_trajectory.extend(self.create_wait_frames(end_pose, self.WAIT_END))
        
        # Save
        output_data = {
            "meta": {
                "source": "step08_slow_motion",
                "unit": "radians",
                "count": len(final_trajectory),
                "frequency": self.FREQ,
                "strategy": "super_slow_air_move",
                "parent_meta": original_meta
            },
            "joints": final_trajectory
        }
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"✅ 优化完成！文件已保存至: {self.output_path}")
        print(f"   - 总点数: {len(final_trajectory)}")
        print(f"   - 总时长: {len(final_trajectory) * self.DT:.2f} s")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    INPUT_FILE = os.path.join(img_dir, "step07_joint_trajectory.json")
    OUTPUT_FILE = os.path.join(img_dir, "step08_optimized_trajectory.json")
    
    optimizer = TrajectoryOptimizer(INPUT_FILE, OUTPUT_FILE)
    optimizer.optimize()