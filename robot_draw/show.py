import matplotlib.pyplot as plt
import cv2
import os
import math

def show_all_steps():
    # 1. 基础配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    
    # 2. 定义要展示的图片列表 (标题, 文件名)
    # 请确保这些文件都在 img 文件夹里
    IMAGE_LIST = [
        ("Original Input", "test.jpg"),                      # 原图
        ("Step 1: Preprocess", "step01_preprocess.png"),      # 预处理 (二值化/去噪)
        ("Step 2: Skeleton", "step02_skeleton.png"),     # 骨架化
        ("Step 3: Graph Topology", "step03_graph.png"),# 图构建 (节点/边)
        ("Step 4: Path Planning", "step04_path.png") # 最终路径规划 (顺序/平滑)
    ]

    # 计算布局：2行3列 (为了容纳5张图)
    num_images = len(IMAGE_LIST)
    cols = 3
    rows = math.ceil(num_images / cols)

    # 设置画布大小
    plt.figure(figsize=(18, 10))
    plt.suptitle("Robot Drawing Project - Full Pipeline", fontsize=20, weight='bold')

    for i, (title, filename) in enumerate(IMAGE_LIST):
        file_path = os.path.join(img_dir, filename)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"[Warn] 找不到文件: {filename}，跳过展示。")
            continue
            
        # 读取图片
        # 注意：matplotlib 显示是用 RGB，OpenCV 读取是 BGR
        img = cv2.imread(file_path)
        if img is None:
            print(f"[Error] 无法读取文件: {filename}")
            continue
            
        # 颜色转换 BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 添加子图
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_rgb)
        plt.title(title, fontsize=14)
        plt.axis('off') # 隐藏坐标轴刻度，只看图

    # 自动调整布局，防止重叠
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        show_all_steps()
    except Exception as e:
        print(f"发生错误: {e}")