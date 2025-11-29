import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class ImageProcessor:
    """
    包含的img：
    original_img_raw
    original_img
    gray_img
    binary_img
    processed_img
    """
    def __init__(self, image_path, target_max_size=1024):
        """
        初始化图像处理器
        :param image_path: 图片路径
        :param target_max_size: 标准化处理的最大边长（像素），默认1024
        """
        if not os.path.exists(image_path):
             raise FileNotFoundError(f"找不到图片文件: {image_path}")
             
        self.original_img_raw = cv2.imread(image_path)
        if self.original_img_raw is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # --- 1. 标准化缩放 ---
        # 这一步保证了无论输入是多大分辨率，后续处理的尺度基本一致
        self.original_img = self.resize_maintaining_aspect_ratio(self.original_img_raw, target_max_size)
        
        # 转换为灰度图准备处理
        self.gray_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        
        # 初始化中间变量
        self.binary_img = None
        self.processed_img = None

        h, w = self.gray_img.shape
        print(f"[Info] 图片已加载并标准化为: {w}x{h}")


    def resize_maintaining_aspect_ratio(self, image, target_max_size):
        """
        辅助函数：保持纵横比缩放图片
        """
        h, w = image.shape[:2]
        scale = 1.0
        if h > w:
            scale = target_max_size / h
        else:
            scale = target_max_size / w
            
        # 如果图片本身比目标尺寸小，就不放大了，避免失真
        if scale >= 1.0:
            return image

        new_w = int(w * scale)
        new_h = int(h * scale)
        # 使用 INTER_AREA 插值方法，这在缩小图像时效果最好，能避免波纹
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_img


    def binarize_otsu(self):
        """
        第一步：Otsu 二值化
        目标：将图片变成纯黑底、纯白线的图像。
        """
        # 使用 THRESH_BINARY_INV + THRESH_OTSU
        # INV ：最终线条是白色的（前景），背景是黑色的
        thresh_val, self.binary_img = cv2.threshold(
            self.gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        print(f"[Info] Otsu 计算的阈值: {thresh_val:.2f}")
        return self.binary_img
    
    def apply_morphology(self, kernel_size=3):
        """
        第二步：形态学闭运算
        目标：修复线条中的小断裂，填补内部孔洞。
        :param kernel_size: 核大小。越大填补能力越强，但细节丢失越多。
        """
        if self.binary_img is None:
            raise ValueError("请先运行 binarize_otsu！")

        # 1. 定义核 (Kernel)
        # 扫描的笔头，3x3 是一个标准的方形笔头。
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 2. 执行闭运算
        # cv2.MORPH_CLOSE = 先膨胀后腐蚀
        self.processed_img = cv2.morphologyEx(self.binary_img, cv2.MORPH_CLOSE, kernel)
        
        print(f"[Info] 形态学闭运算完成，Kernel大小: {kernel_size}x{kernel_size}")
        return self.processed_img
    
    def remove_small_noise(self, min_area=50):
        """
        第三步：连通域去噪
        目标：分析图像中的所有独立白色区域，删除面积过小的区域。
        :param min_area: 面积阈值。小于这个像素数的区域会被丢弃。
        """
        if self.processed_img is None:
            # 如果没做形态学，就用二值图兜底
            self.processed_img = self.binary_img

        # 1. 连通域分析
        # num_labels: 发现了多少个区域
        # labels: 一张地图，告诉我们每个像素属于哪个区域(1, 2, 3...)
        # stats: 统计数据，包含每个区域的 x, y, 宽, 高, 面积
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            self.processed_img, connectivity=8
        )
        
        # 2. 创建一张新的干净画布 (全黑)
        clean_img = np.zeros_like(self.processed_img)
        
        print(f"[Info] 初步发现 {num_labels - 1} 个连通区域 (Label 0 是背景)")

        kept_count = 0
        # 3. 遍历所有白色区域 (从1开始，因为0是背景黑色)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= min_area:
                # 如果面积达标，就把这个区域“复制”到新画布上
                clean_img[labels == i] = 255
                kept_count += 1
            # else: 面积太小，不做操作，新画布上默认就是黑的，相当于删除了
                
        print(f"[Info] 去噪完成: 保留了 {kept_count} 个有效笔迹，移除了 {num_labels - 1 - kept_count} 个噪点")
        
        # 更新处理后的图像
        self.processed_img = clean_img
        return self.processed_img


    def show_current_step(self, title="Result"):
        """调试用：显示当前处理结果"""
        if self.binary_img is None:
             print("还没进行二值化，无法显示")
             return
        
        plt.figure(figsize=(8, 8))
        plt.title(title)
        # cmap='gray' 告诉 matplotlib 这是一张灰度图
        plt.imshow(self.binary_img, cmap='gray')
        plt.axis('off') # 不显示坐标轴
        plt.show()

# ==========================================
# 执行部分
# ==========================================
if __name__ == "__main__":
    # --- 配置 ---
    # 图片命名为 test.jpg 放在代码同一目录下
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "img")
    IMAGE_FILE = os.path.join(img_dir, "test.jpg")
    OUTPUT_FILE = os.path.join(img_dir, "step01_final_img.png")
    
    try:
        # 1. 初始化并加载、标准化图片
        # 设定最大边长为 1000 像素
        processor = ImageProcessor(IMAGE_FILE, target_max_size=1000)
        
        # 2. 执行 Otsu 二值化
        processor.binarize_otsu()
        
        # 3. 执行形态学闭运算
        #kernal_size决定了膨胀的大小
        processor.apply_morphology(kernel_size=3)

        #4. 去噪
        final_img = processor.remove_small_noise(min_area=100)

        # 5. 显示结果
        # 显示原图 (灰度)
        plt.subplot(1, 2, 1)
        plt.title("Original (Standardized)")
        plt.imshow(processor.gray_img, cmap='gray')
        plt.axis('off')
        
        # 显示处理后的最终图
        plt.subplot(1, 2, 2)
        plt.title("Step 3: Final Cleaned Image")
        plt.imshow(final_img, cmap='gray')
        plt.axis('off')
        
        plt.show()
        
        # 保存图片
        cv2.imwrite(OUTPUT_FILE, final_img)
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print(f"请确保在代码目录下放了一张名为 '{IMAGE_FILE}' 的图片用于测试。")
    except Exception as e:
        print(f"\n❌ 发生其他错误: {e}")