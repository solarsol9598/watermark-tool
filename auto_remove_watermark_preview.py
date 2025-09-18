import cv2
import numpy as np
import os
import glob

input_folder = "input"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# 限制检测范围的长宽（单位：像素）
search_width = 120
search_height = 60

def detect_watermark(img):
    """检测左上角水印蒙版"""
    h, w = img.shape[:2]
    roi_w = min(search_width, w)
    roi_h = min(search_height, h)

    # 裁剪左上角ROI
    roi = img[0:roi_h, 0:roi_w]

    # 转灰度 & 边缘检测
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 二值化
    _, mask_roi = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)

    # 扩展（防止漏掉水印边缘）
    kernel = np.ones((3, 3), np.uint8)
    mask_roi = cv2.dilate(mask_roi, kernel, iterations=2)

    # 全图蒙版
    mask = np.zeros((h, w), np.uint8)
    mask[0:roi_h, 0:roi_w] = mask_roi

    return mask, (0, 0, roi_w, roi_h)

def preview_detection():
    """预览前几张图片的检测效果"""
    files = glob.glob(f"{input_folder}/*.*")
    if not files:
        print("❌ 没有找到需要处理的图片")
        return

    for file in files[:3]:  # 只预览前3张
        img = cv2.imread(file)
        if img is None:
            print(f"[跳过] 无法读取 {file}")
            continue

        mask, (x, y, w, h) = detect_watermark(img)
        preview = img.copy()
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红框显示检测区域

        print(f"预览检测：{file}")
        cv2.imshow("Preview", preview)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):  # 按n退出
            cv2.destroyAllWindows()
            return False
    cv2.destroyAllWindows()
    return True

def batch_process():
    """批量去水印"""
    files = glob.glob(f"{input_folder}/*.*")
    for file in files:
        img = cv2.imread(file)
        mask, _ = detect_watermark(img)

        result = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        output_path = os.path.join(output_folder, os.path.basename(file))
        cv2.imwrite(output_path, result)
        print(f"[完成] {file} -> {output_path}")
    print("✅ 批量去水印完成！")

if __name__ == "__main__":
    if preview_detection():
        confirm = input("是否开始批量处理？(y/n)：").strip().lower()
        if confirm == 'y':
            batch_process()
        else:
            print("❌ 已取消批量处理")
