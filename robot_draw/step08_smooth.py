import json
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

class JointTrajectoryOptimizer:
    def __init__(self, frequency=125.0):
        self.dt = 1.0 / frequency  # 8ms
        
        # === 速度配置 (关节角速度 rad/s) ===
        # 注意：这里不是 mm/s，而是关节转动的快慢
        # Draw: 慢速 (保证精度)
        self.SPEED_SCALE_DRAW = 0.15 
        # Air: 快速 (提高效率)
        self.SPEED_SCALE_AIR  = 0.40 
        
        # === 停顿配置 (秒) ===
        self.WAIT_START = 1.0
        self.WAIT_CONN  = 1.0  # 动作连接处停顿 1 秒

    def _get_cumulative_dist(self, joints):
        """计算关节空间的累积路程 (L1范数或L2范数均可)"""
        diffs = np.abs(np.diff(joints, axis=0))
        # 找出变动最大的那个关节作为基准
        dists = np.max(diffs, axis=1) 
        return np.concatenate(([0], np.cumsum(dists)))

    def resample_segment(self, joints, speed_rad_s, kind='cubic'):
        """对一段关节轨迹进行重采样"""
        joints = np.array(joints)
        if len(joints) < 2: return [joints[0]]

        # 1. 计算总行程 (弧度)
        cum_dist = self._get_cumulative_dist(joints)
        total_dist = cum_dist[-1]
        
        if total_dist < 1e-4: return [joints[0]]

        # 2. 计算耗时
        total_time = total_dist / speed_rad_s
        # 至少给 2 帧
        if total_time < self.dt * 2: total_time = self.dt * 2

        # 3. 生成时间轴
        num_frames = int(np.ceil(total_time / self.dt))
        t_target = np.linspace(0, total_dist, num_frames)
        
        # 4. 插值
        # 只有点数够多才能用 cubic
        actual_kind = 'cubic' if (kind == 'cubic' and len(joints) >= 4) else 'linear'
        
        try:
            # axis=0 对每一列(每个关节)分别插值
            interpolator = interp1d(cum_dist, joints, axis=0, kind=actual_kind)
            dense_joints = interpolator(t_target)
        except:
            # 降级
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
    
    INPUT_FILE = os.path.join(img_dir, "step07_joint_trajectory.json")
    OUTPUT_FILE = os.path.join(img_dir, "step08_optimized_trajectory.json")
    
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
    
    # 1. 初始停顿 (获取第一个点)
    if structured_joints:
        start_pose = structured_joints[0][0][0]
        start_wait_frames = int(optimizer.WAIT_START / optimizer.dt)
        final_traj.extend([start_pose] * start_wait_frames)

    # 2. 遍历处理
    total_strokes = len(structured_joints)
    for s_idx, segments in enumerate(structured_joints):
        print(f"\r处理笔画: {s_idx+1}/{total_strokes}", end="")
        
        # segments: 0=Move, 1=Drop, 2=Draw, 3=Lift
        for seg_idx, points in enumerate(segments):
            
            # --- 策略选择 ---
            if seg_idx == 2: # Draw
                speed = optimizer.SPEED_SCALE_DRAW
                kind = 'cubic'  # 笔画要平滑
            else: # Air Move
                speed = optimizer.SPEED_SCALE_AIR
                kind = 'linear' # 空移走直线即可，防抖
            
            # --- 重采样 ---
            dense_segment = optimizer.resample_segment(points, speed, kind)
            final_traj.extend(dense_segment)
            
            # --- 插入停顿 ---
            # 获取这段的最后一个姿态
            last_pose = dense_segment[-1]
            wait_frames = int(optimizer.WAIT_CONN / optimizer.dt)
            final_traj.extend([last_pose] * wait_frames)

    print(f"\n✅ 优化完成! 总帧数: {len(final_traj)}")
    
    # 3. 后处理平滑 (消除 IK 带来的微小锯齿)
    final_traj_np = np.array(final_traj)
    final_traj_smoothed = apply_smoothing(final_traj_np)
    
    # 4. 保存
    output_data = {
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
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=None)
    
    print(f"💾 文件已保存: {OUTPUT_FILE}")