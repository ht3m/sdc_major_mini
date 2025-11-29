import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class Skeletonizer:
    def __init__(self, input_path):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"找不到输入文件: {input_path}\n请先运行 step01_preprocess.py 生成该文件！")
            
        # 读取 Step 1 的结果
        # 注意：一定要以【灰度模式】读取
        self.binary_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        # 保险起见，再次强制二值化，确保只有 0 和 255
        # 防止保存图片时产生了一些非纯黑纯白的边缘像素
        _, self.binary_img = cv2.threshold(self.binary_img, 127, 255, cv2.THRESH_BINARY)
        
        self.skeleton = None

    def run_guo_hall(self):
        """
        执行 Guo-Hall 细化算法
        """
        print("[Info] 正在执行 Guo-Hall 骨架化...")
        
        # cv2.ximgproc.thinning 需要 opencv-contrib-python
        # thinningType=1 对应 Guo-Hall 算法
        self.skeleton = cv2.ximgproc.thinning(
            self.binary_img, 
            thinningType=cv2.ximgproc.THINNING_GUOHALL
        )
        return self.skeleton

    def show_comparison(self):
        """
        显示对比图：原二值图 vs 骨架图 vs 叠加图
        """
        if self.skeleton is None:
            return

        plt.figure(figsize=(15, 6))

        # 1. 输入的粗线条
        plt.subplot(1, 3, 1)
        plt.title("Input: Clean Binary")
        plt.imshow(self.binary_img, cmap='gray')
        plt.axis('off')

        # 2. 输出的细骨架
        plt.subplot(1, 3, 2)
        plt.title("Output: Guo-Hall Skeleton")
        plt.imshow(self.skeleton, cmap='gray')
        plt.axis('off')

        # 3. 叠加检查 (Overlay)
        # 这一步最重要：检查骨架是不是真的在正中间
        plt.subplot(1, 3, 3)
        plt.title("Overlay Check (Red=Skeleton)")
        
        # 制作背景：把灰度转成 RGB，这样可以是彩色的
        background = cv2.cvtColor(self.binary_img, cv2.COLOR_GRAY2BGR)
        
        # 制作前景：创建一个纯红色的图
        # 逻辑：在 skeleton 为白色的地方，把 background 染成红色
        background[self.skeleton == 255] = [255, 0, 0]  # BGR格式，红色是 (0,0,255)，但在matplotlib里对应 RGB
        
        # 因为 matplotlib 用 RGB，OpenCV 用 BGR，这里为了显示红色，我们需要留意
        # 简单的做法：直接显示，如果红色变成了蓝色，就是通道反了，不影响逻辑检查
        plt.imshow(background) 
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    INPUT_FILE = os.path.join(img_dir, "step01_final_img.png")
    OUTPUT_FILE = os.path.join(img_dir, "step02_skeleton_img.png")

    try:
        # 1. 初始化
        skel_tool = Skeletonizer(INPUT_FILE)
        
        # 2. 运行算法
        result = skel_tool.run_guo_hall()
        
        # 3. 保存结果
        cv2.imwrite(OUTPUT_FILE, result)
        print(f"✅ 骨架化完成，结果已保存至: {OUTPUT_FILE}")
        
        # 4. 显示
        skel_tool.show_comparison()
        
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")