import json
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

class JointTrajectoryOptimizer:
    def __init__(self, frequency=125.0):
        self.dt = 1.0 / frequency  # 8ms
        
        # === 速度配置 (关节角速度 rad/s) ===
        self.SPEED_SCALE_DRAW = 0.06
        self.SPEED_SCALE_AIR  = 0.15
        
        # === 停顿配置 (秒) ===
        self.WAIT_START = 0.5
        self.WAIT_CONN  = 0.5 

    def _get_cumulative_dist(self, joints):
        diffs = np.abs(np.diff(joints, axis=0))
        dists = np.max(diffs, axis=1) 
        return np.concatenate(([0], np.cumsum(dists)))

    def resample_segment(self, joints, speed_rad_s, kind='cubic'):
        joints = np.array(joints)
        if len(joints) < 2: return [joints[0]]

        cum_dist = self._get_cumulative_dist(joints)
        total_dist = cum_dist[-1]
        
        if total_dist < 1e-4: return [joints[0]]

        total_time = total_dist / speed_rad_s
        if total_time < self.dt * 2: total_time = self.dt * 2

        num_frames = int(np.ceil(total_time / self.dt))
        t_target = np.linspace(0, total_dist, num_frames)
        
        actual_kind = 'cubic' if (kind == 'cubic' and len(joints) >= 4) else 'linear'
        
        try:
            interpolator = interp1d(cum_dist, joints, axis=0, kind=actual_kind)
            dense_joints = interpolator(t_target)
        except:
            interpolator = interp1d(cum_dist, joints, axis=0, kind='linear')
            dense_joints = interpolator(t_target)
            
        return dense_joints.tolist()

def apply_smoothing(joints, window=11):
    if len(joints) < window: return joints
    print(f"🧹 应用 Savitzky-Golay 平滑 (Window={window})...")
    smoothed = np.zeros_like(joints)
    for i in range(6):
        smoothed[:, i] = savgol_filter(joints[:, i], window, 3, mode='nearest')
    return smoothed.tolist()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    INPUT_FILE = os.path.join(img_dir, "step07_joint_trajectory_fixed_j6.json")
    
    # === 输出文件定义 ===
    # 1. 包含元数据的完整文件 (供 Python/Debug 查看)
    OUTPUT_FILE_FULL = os.path.join(img_dir, "step08_optimized_trajectory.json")
    # 2. ⚠️ 专门给 Rust move_traj_from_file 用的文件
    OUTPUT_FILE_RUST = os.path.join(img_dir, "step08_optimized_trajectory_rust.json")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        exit()

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
        structured_joints = data.get("structured_joints", [])
        meta = data.get("meta", {})

    print(f"🚀 开始轨迹优化与重采样...")
    optimizer = JointTrajectoryOptimizer()
    
    final_traj = []
    
    # 1. 初始停顿
    if structured_joints:
        start_pose = structured_joints[0][0][0]
        start_wait_frames = int(optimizer.WAIT_START / optimizer.dt)
        final_traj.extend([start_pose] * start_wait_frames)

    # 2. 遍历处理
    total_strokes = len(structured_joints)
    for s_idx, segments in enumerate(structured_joints):
        print(f"\r处理笔画: {s_idx+1}/{total_strokes}", end="")
        for seg_idx, points in enumerate(segments):
            if seg_idx == 2: # Draw
                speed = optimizer.SPEED_SCALE_DRAW
                kind = 'cubic'
            else: # Air Move
                speed = optimizer.SPEED_SCALE_AIR
                kind = 'linear'
            
            dense_segment = optimizer.resample_segment(points, speed, kind)
            final_traj.extend(dense_segment)
            
            last_pose = dense_segment[-1]
            wait_frames = int(optimizer.WAIT_CONN / optimizer.dt)
            final_traj.extend([last_pose] * wait_frames)

    print(f"\n✅ 优化完成! 总帧数: {len(final_traj)}")
    
    # 3. 后处理平滑
    final_traj_np = np.array(final_traj)
    final_traj_smoothed = apply_smoothing(final_traj_np)
    
    # ==========================================================
    # 4. 保存文件 (重点修改部分)
    # ==========================================================
    
    # (A) 保存标准格式 (带 Meta 信息)
    output_data_full = {
        "meta": {
            "source": "step08_trajectory_optimization",
            "unit": "radians",
            "count": len(final_traj_smoothed),
            "frequency": 125.0,
            "speeds": {"draw": optimizer.SPEED_SCALE_DRAW, "air": optimizer.SPEED_SCALE_AIR},
            "parent_meta": meta
        },
        "joints": final_traj_smoothed
    }
    with open(OUTPUT_FILE_FULL, 'w') as f:
        json.dump(output_data_full, f, indent=2)
    print(f"💾 通用格式已保存: {OUTPUT_FILE_FULL}")

    # (B) 🔥 保存 Rust Enum 专用格式 🔥
    # 格式要求: [{"Joint": [0.1, ...]}, {"Joint": [0.2, ...]}, ...]
    # 只有这种格式才能被 move_traj_from_file 直接读取
    rust_data = [{"Joint": frame} for frame in final_traj_smoothed]
    
    with open(OUTPUT_FILE_RUST, 'w') as f:
        json.dump(rust_data, f, indent=2)
        
    print(f"🦀 Rust专用格式已保存: {OUTPUT_FILE_RUST}")
    print(f"   (请在 Rust 中加载此文件)")