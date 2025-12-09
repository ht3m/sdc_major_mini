import json
import numpy as np
import os
from scipy.optimize import minimize

class JakaRobotIK:
    def __init__(self):
        # --- JAKA Mini 2 D-H 参数 (基于之前的图纸) ---
        # 单位: mm
        # alpha, a, d, theta_offset
        # 注意：数值解法中，d4+d5+d6 总长匹配即可，内部拆分主要影响中间关节位置，不影响末端到达
        self.DH_PARAMS = [
            # alpha(i-1), a(i-1), d(i), theta_offset(i)
            [np.pi/2,  0,      187.0,  0],          # Joint 1
            [0,        210.0,  0,      0],          # Joint 2
            [0,        210.5,  0,      0],          # Joint 3
            [np.pi/2,  0,      80.0,   0],          # Joint 4 (估算)
            [-np.pi/2, 0,      40.0,   0],          # Joint 5 (估算)
            [0,        0,      39.3,   0]           # Joint 6 (估算, 总 wrist=159.3)
        ]
        
    def dh_matrix(self, theta, alpha, a, d):
        """ 标准 D-H 变换矩阵 """
        c = np.cos(theta)
        s = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [c, -s*ca, s*sa, a*c],
            [s, c*ca, -c*sa, a*s],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, joints_rad):
        """
        正运动学：计算给定关节角下的末端位姿矩阵 T (4x4)
        """
        T = np.eye(4)
        
        for i, (alpha, a, d, offset) in enumerate(self.DH_PARAMS):
            theta = joints_rad[i] + offset
            Ti = self.dh_matrix(theta, alpha, a, d)
            T = T @ Ti
            
        return T

    def inverse_kinematics(self, target_pos, target_rpy_rad, seed_joints):
        """
        数值逆解 (Optimization based IK)
        :param target_pos: [x, y, z] 目标位置
        :param target_rpy_rad: [roll, pitch, yaw] 目标欧拉角 (弧度)
        :param seed_joints: 初始猜测关节角 (通常用上一个点的解)
        """
        
        # 1. 构建目标旋转矩阵 (R_target)
        rx, ry, rz = target_rpy_rad
        # ZYX 欧拉角旋转顺序 (这里的定义需与实际需求匹配，通常 Robotic 常用 ZYZ 或 ZYX)
        # 这里使用简单的固定轴 XYZ 旋转矩阵构建
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        
        # 注意：这里假设 target_rpy 是想让法兰盘转到特定的空间姿态
        # 如果是 "正对X正方向(Pitch 90)"，意味着法兰Z轴指向世界坐标X轴
        R_target = Rz @ Ry @ Rx

        # 2. 定义优化误差函数
        def error_function(current_joints):
            T_current = self.forward_kinematics(current_joints)
            P_current = T_current[:3, 3]
            R_current = T_current[:3, :3]
            
            # 位置误差 (欧氏距离)
            pos_error = np.linalg.norm(P_current - np.array(target_pos))
            
            # 姿态误差 (旋转矩阵差值的范数)
            rot_error = np.linalg.norm(R_current - R_target)
            
            # 权重：位置通常比姿态更重要一点，或者设为 1:1
            return pos_error * 1.0 + rot_error * 0.5

        # 3. 求解
        # 关节限位 (设为 -2pi 到 2pi)
        bounds = [(-2*np.pi, 2*np.pi)] * 6
        
        res = minimize(
            error_function, 
            seed_joints, 
            method='SLSQP', 
            bounds=bounds, 
            tol=1e-5,
            options={'maxiter': 50} # 实时性要求不高，可以多迭代几次
        )
        
        return res.x, res.fun # 返回角度和误差值

def process_and_solve(input_file, output_file, draw_z_mm, pitch_angle_deg):
    if not os.path.exists(input_file):
        print(f"❌ 找不到输入文件: {input_file}")
        return

    print(f"[IK] 启动逆解算...")
    print(f"     输入文件: {input_file}")
    print(f"     目标平面高度 (Z): {draw_z_mm} mm")
    print(f"     目标法兰俯仰角 (Pitch): {pitch_angle_deg}° (正对 X 正方向)")

    # 1. 加载轨迹
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    trajectories = data.get('trajectories', [])
    if not trajectories:
        print("❌ JSON 中没有轨迹数据")
        return

    robot = JakaRobotIK()
    
    # 2. 初始姿态 (Seed)
    # 给一个合理的初始猜测，防止解算器一开始就跑偏
    # 这个姿态大约是机械臂弯曲在前方
    current_joints = np.radians([0, -45, 90, 0, 45, 0]) 
    
    # 目标姿态转换
    # Roll=0, Pitch=用户设定, Yaw=0 (假设画笔不自转，始终朝前)
    target_rpy = np.radians([0, pitch_angle_deg, 0])
    
    final_output = []
    total_points = sum(len(s) for s in trajectories)
    processed_count = 0
    
    print(f"     共发现 {len(trajectories)} 条笔画，准备计算...")

    # 3. 逐点逆解
    for stroke_idx, stroke in enumerate(trajectories):
        stroke_joints = []
        
        for point in stroke:
            # 读取绝对物理坐标 (Step 5 生成的)
            x, y = point[0], point[1]
            z = draw_z_mm
            
            target_pos = [x, y, z]
            
            # 计算逆解
            joints_rad, error = robot.inverse_kinematics(target_pos, target_rpy, current_joints)
            
            # 简单检查误差
            if error > 1.0: # 如果误差大于 1mm/rad 混合值，可能有点问题
                print(f"⚠️ 警告: 点 ({x:.1f}, {y:.1f}) 逆解误差较大: {error:.4f}")

            # 转为角度保存
            joints_deg = np.degrees(joints_rad).tolist()
            # 保留2位小数
            joints_deg = [round(j, 2) for j in joints_deg]
            
            stroke_joints.append(joints_deg)
            
            # 【关键】更新 Seed
            # 下一个点从当前点的解开始算，保证轨迹连续，不会突变
            current_joints = joints_rad
            
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"     进度: {processed_count}/{total_points} ...", end='\r')

        final_output.append(stroke_joints)

    # 4. 保存
    output_data = {
        "meta": {
            "source": "step06_reverse",
            "z_height_mm": draw_z_mm,
            "pitch_angle_deg": pitch_angle_deg,
            "description": "Joint angles (deg) for Jaka Mini 2"
        },
        "joint_trajectories": final_output
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\n✅ 逆解完成！关节角度数据已保存至: {output_file}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    # 输入：Step 05 生成的绝对坐标文件
    INPUT_PATH = os.path.join(img_dir, "step05_re_robot_path.json") # 注意你刚才要求的文件名
    # 输出：关节角度文件
    OUTPUT_PATH = os.path.join(img_dir, "step06_joint_angles.json")
    
    # --- 这里是你要的可变参数 ---
    
    # 1. 图纸的 Z 轴坐标 (单位 mm)
    # 如果笔尖刚好接触桌面，设为 0；如果笔架比较长，可能是一个正值或负值，视基座安装高度而定
    DRAW_Z = 20.0 
    
    # 2. 末端法兰盘的角度 (单位 度)
    # 你要求的：正对 X 正方向 -> 90度
    # (如果是垂直向下写字通常是 180度)
    TARGET_PITCH = 90.0
    
    process_and_solve(INPUT_PATH, OUTPUT_PATH, DRAW_Z, TARGET_PITCH)