import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

class JakaPoseChecker:
    def __init__(self):
        # 1. JAKA Mini 2 D-H 参数 (alpha, a, d, offset)
        self.DH_PARAMS = [
            [0,        0,      0,      0],          # J1
            [np.pi/2,  0,      187.0,  0],          # J2
            [0,        210.0,  0,      0],          # J3
            [-np.pi/2, 210.5,  6.0,    0],          # J4 
            [np.pi/2,  0,      0,      0],          # J5
            [np.pi/2,  159.3,  0,      0]           # J6
        ]

        # 2. 严格的关节物理限位 (对应 Rust 代码)
        # J1, J4, J6: ±360° | J2, J3, J5: ±120°
        limit_large = 2 * np.pi
        limit_small = (2 * np.pi) / 3.0
        
        self.bounds = [
            (-limit_large, limit_large), # J1
            (-limit_small, limit_small), # J2 (受限)
            (-limit_small, limit_small), # J3 (受限)
            (-limit_large, limit_large), # J4
            (-limit_small, limit_small), # J5 (受限)
            (-limit_large, limit_large)  # J6
        ]

    def dh_matrix(self, theta, alpha, a, d):
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
        T = np.eye(4)
        for i, (alpha, a, d, offset) in enumerate(self.DH_PARAMS):
            theta = joints_rad[i] + offset
            Ti = self.dh_matrix(theta, alpha, a, d)
            T = T @ Ti
        return T

    def solve_ik(self, target_pos, target_quat):
        """
        :param target_pos: [x, y, z] (mm)
        :param target_quat: [x, y, z, w]
        """
        t_pos = np.array(target_pos)
        
        # 目标姿态处理
        try:
            r = R.from_quat(target_quat) # 格式: [x, y, z, w]
            t_rot = r.as_matrix()
        except Exception as e:
            print(f"❌ 四元数无效: {e}")
            return None, 999.0

        # 初始猜测 (Seed) - 避免全0陷入奇异点
        seed_joints = np.radians([0, -30, 90, 0, 60, 0])

        # 误差函数
        def error_function(current_joints):
            T_curr = self.forward_kinematics(current_joints)
            P_curr = T_curr[:3, 3]
            R_curr = T_curr[:3, :3]
            
            # 这里的误差权重：位置(mm) : 姿态(norm) = 1 : 50
            pos_err = np.linalg.norm(P_curr - t_pos)
            rot_err = np.linalg.norm(R_curr - t_rot)
            
            return pos_err * 1.0 + rot_err * 50.0

        # 优化求解
        res = minimize(
            error_function,
            seed_joints,
            method='SLSQP',
            bounds=self.bounds,
            tol=1e-6,
            options={'maxiter': 200}
        )

        return res.x, res.fun

def run_check():
    # ==========================================
    #在此处修改您的目标参数 (无需交互)
    # ==========================================
    
    # 1. 目标位置 XYZ (毫米)
    # 示例: 前方 350mm, 高度 20mm
    TARGET_POS = [300.0, 0.0, 0.0] 

    # 2. 目标姿态四元数 [x, y, z, w]
    # 常用姿态参考:
    # 笔尖垂直向下 (Pitch 180): [0, 1, 0, 0]  <-- 最推荐
    # 笔尖水平向前 (Pitch 90) : [0, 0.7071, 0, 0.7071]
    # 笔尖水平向左 (Roll 90)  : [0.7071, 0, 0, 0.7071]
    TARGET_QUAT = [0.0, 1, 0.0, 0.0]

    # ==========================================
    
    print("-" * 60)
    print(f"📍 检测目标:")
    print(f"   Pos (XYZ):  {TARGET_POS}")
    print(f"   Quat(XYZW): {TARGET_QUAT}")
    print("-" * 60)

    checker = JakaPoseChecker()
    joints_rad, error = checker.solve_ik(TARGET_POS, TARGET_QUAT)
    joints_deg = np.degrees(joints_rad)

    # 判定阈值 (误差通常应小于 1.0)
    if error < 1.0:
        print("✅ 求解成功 (Reachable)")
        print(f"   综合残差: {error:.5f}")
        print("\n🤖 关节角度 (Degrees):")
        
        labels = ["J1", "J2", "J3", "J4", "J5", "J6"]
        limits = [360, 120, 120, 360, 120, 360] # 显示用的参考限位
        
        for i, val in enumerate(joints_deg):
            limit_str = f"±{limits[i]}"
            warn_mark = "⚠️ 极限!" if abs(val) > (limits[i] - 0.5) else ""
            print(f"   {labels[i]}: {val:8.2f}°  (Limit: {limit_str}) {warn_mark}")
            
    else:
        print("❌ 求解失败 (Unreachable)")
        print(f"   综合残差: {error:.4f} (太大，说明够不着或姿态别扭)")
        print("\n💡 建议:")
        print("   1. 检查目标点是否太远 (>580mm) 或太近 (<200mm)。")
        print("   2. 检查 J2/J3/J5 是否受限于 ±120°。")
        print("   3. 尝试将姿态改为垂直向下 [0, 1, 0, 0]。")

if __name__ == "__main__":
    run_check()