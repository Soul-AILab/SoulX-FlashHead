#!/usr/bin/env python3
"""
人脸裁剪处理脚本
从单张图像中检测人脸，裁剪并调整大小到指定尺寸
"""
import os
from PIL import Image
import numpy as np

from flash_head.utils.cpu_face_handler import CPUFaceHandler

def get_scaled_bbox(
    bbox, img_w, img_h, ratio: float = 1.0, face_image: Image.Image = None
):
    """
    根据人脸边界框计算缩放后的裁剪区域
    
    Args:
        bbox: 人脸边界框 [x1, y1, x2, y2]
        img_w: 图像宽度
        img_h: 图像高度
        ratio: 缩放比例，数值越大，人脸在画面中的比例越小（周围留白越多）
        face_image: PIL Image 对象
    
    Returns:
        裁剪后的人脸图像
    """
    x1, y1, x2, y2 = bbox

    # Calculate center point
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate width and height
    width = x2 - x1

    # Scale width and height
    new_width = width * ratio
    new_height = new_width

    # tile pix
    dis_x_left = new_width * 0.5
    dis_x_right = new_width - dis_x_left  # 0.5new_width
    dis_y_up = new_height * 0.55
    dis_y_down = new_height - dis_y_up  # 0.45new_height

    # Calculate new coordinates
    new_x1 = int(max(0, center_x - dis_x_left))
    new_y1 = int(max(0, center_y - dis_y_up))
    new_x2 = int(min(img_w, center_x + dis_x_right))
    new_y2 = int(min(img_h, center_y + dis_y_down))
    scaled_bbox = [new_x1, new_y1, new_x2, new_y2]
    crop_face = face_image.crop(scaled_bbox)
    return crop_face


def process_image(
    input_path,
    face_ratio=2.0,
    target_size=(512, 512),
):
    """
    处理单张图像，进行人脸检测和裁剪
    
    Args:
        input_path: 输入图像路径
        face_ratio: 人脸缩放比例，建议范围：1.5-3.0，默认2.0
        target_size: 输出图像尺寸，默认(512, 512)
    
    Returns:
        imgae: 处理后的图像
    """
    # 初始化人脸检测器
    face_detector = CPUFaceHandler()
    
    # 验证输入文件
    if not os.path.isfile(input_path):
        raise ValueError(f"File not found: {input_path}")
    
    try:
        # 读取图像
        image = Image.open(input_path)
        image = image.convert("RGB")
        image_rgb = np.array(image)
        img_h, img_w = image_rgb.shape[:2]
        
        # 检测人脸
        boxes, scores = face_detector(image_rgb)
        
        if len(boxes) == 0:
            raise ValueError("No face detected")
        
        # 转换边界框坐标（从相对坐标转为绝对坐标）
        boxes_abs = [
            boxes[0][0] * img_w,
            boxes[0][1] * img_h,
            boxes[0][2] * img_w,
            boxes[0][3] * img_h
        ]
        
        # 裁剪人脸
        crop_face = get_scaled_bbox(boxes_abs, img_w, img_h, face_ratio, image)
        
        # 调整大小
        crop_face = crop_face.resize(target_size)
        
        return crop_face
            
    except Exception as e:
        raise ValueError(f"Error processing {input_path}: {e}")