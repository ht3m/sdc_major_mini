import json
import numpy as np
import os
import copy

def override_j6_angle(input_file, output_file, target_deg=10.0):
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件: {input_file}")
        return

    print(f"📂 读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    structured_joints = data.get("structured_joints", [])
    meta = data.get("meta", {})
    
    # 目标角度转弧度
    target_rad = np.radians(target_deg)
    print(f"🔒 正在将所有点的 J6 强制锁定为: {target_deg}° ({target_rad:.4f} rad)")

    # 统计修改点数
    total_points = 0
    modified_strokes = []

    # === 核心处理循环 ===
    # 遍历结构: [笔画] -> [段(Move/Drop...)] -> [点] -> [6个关节]
    for s_idx, stroke in enumerate(structured_joints):
        new_stroke = []
        for seg_idx, segment in enumerate(stroke):
            new_segment = []
            for p_idx, point in enumerate(segment):
                # 复制原有关节数据 (防止引用修改)
                new_joints = list(point)
                
                # 🔥 强制覆盖 J6 (索引 5)
                # 无论之前 IK 算出是多少，这里统统改成 10度
                if len(new_joints) >= 6:
                    new_joints[5] = target_rad
                
                new_segment.append(new_joints)
                total_points += 1
            new_stroke.append(new_segment)
        modified_strokes.append(new_stroke)

        if (s_idx + 1) % 10 == 0:
            print(f"\r   - 处理进度: {s_idx + 1}/{len(structured_joints)}", end="")

    print(f"\n✅ 处理完成! 共修改了 {total_points} 个点。")

    # 更新 Meta 信息
    meta["post_processing"] = f"Forced J6 to {target_deg} degrees"
    meta["source"] = "step07b_force_j6"

    output_data = {
        "meta": meta,
        "structured_joints": modified_strokes
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 结果已保存至: {output_file}")
    print(f"👉 下一步: 请修改 step08，让它读取这个新文件！")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    # 输入：原始 IK 结果
    INPUT_FILE = os.path.join(img_dir, "step07_joint_trajectory_multicore.json")
    
    # 输出：修正后的结果 (给 Step 08 用)
    OUTPUT_FILE = os.path.join(img_dir, "step07_joint_trajectory_fixed_j6.json")
    
    # 执行
    override_j6_angle(INPUT_FILE, OUTPUT_FILE, target_deg=10.0)