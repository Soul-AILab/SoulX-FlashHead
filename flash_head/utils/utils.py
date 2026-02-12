import torch
import torch.nn as nn
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import pyloudnorm as pyln

def rgb_to_lab_torch(rgb: torch.Tensor) -> torch.Tensor:
    """
    PyTorch GPU版本：RGB转Lab颜色空间（输入范围[0,1]，张量形状任意，最后一维为通道数）
    参考CIE 1931标准转换公式
    """
    # 转换为线性RGB（sRGB伽马校正逆过程）
    linear_rgb = torch.where(
        rgb > 0.04045,
        ((rgb + 0.055) / 1.055) ** 2.4,
        rgb / 12.92
    )
    
    # 线性RGB转XYZ（使用sRGB标准白点D65）
    xyz_from_rgb = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=rgb.dtype, device=rgb.device)
    
    # 维度适配：确保输入为(B, ..., C)，矩阵乘法后保持空间维度
    shape = linear_rgb.shape
    linear_rgb_flat = linear_rgb.reshape(-1, 3)  # (N, 3)，N=B*T*H*W
    xyz_flat = linear_rgb_flat @ xyz_from_rgb.T  # (N, 3)
    xyz = xyz_flat.reshape(shape)  # 恢复原形状
    
    # XYZ转Lab（使用D65白点参数）
    xyz_ref = torch.tensor([0.95047, 1.0, 1.08883], dtype=rgb.dtype, device=rgb.device)
    xyz_normalized = xyz / xyz_ref[None, None, None, None, :]  # 广播适配(B, C, T, H, W)
    
    # 应用Lab转换公式
    epsilon = 0.008856
    kappa = 903.3
    xyz_normalized = torch.clamp(xyz_normalized, 1e-8, 1.0)  # 避免log(0)
    
    f_xyz = torch.where(
        xyz_normalized > epsilon,
        xyz_normalized ** (1/3),
        (kappa * xyz_normalized + 16) / 116
    )
    
    L = 116 * f_xyz[..., 1] - 16  # Y通道对应亮度
    a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])  # X-Y对应红绿
    b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])  # Y-Z对应蓝黄
    
    lab = torch.stack([L, a, b], dim=-1)  # 最后一维拼接为Lab通道
    return lab

def lab_to_rgb_torch(lab: torch.Tensor) -> torch.Tensor:
    """
    PyTorch GPU版本：Lab转RGB颜色空间（输出范围[0,1]，张量形状任意，最后一维为通道数）
    """
    # Lab分离通道
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    
    # Lab转XYZ
    f_y = (L + 16) / 116
    f_x = (a / 500) + f_y
    f_z = f_y - (b / 200)
    
    epsilon = 0.008856
    kappa = 903.3
    
    x = torch.where(f_x ** 3 > epsilon, f_x ** 3, (116 * f_x - 16) / kappa)
    y = torch.where(L > kappa * epsilon, ((L + 16) / 116) ** 3, L / kappa)
    z = torch.where(f_z ** 3 > epsilon, f_z ** 3, (116 * f_z - 16) / kappa)
    
    # 乘以D65白点参数
    xyz_ref = torch.tensor([0.95047, 1.0, 1.08883], dtype=lab.dtype, device=lab.device)
    xyz = torch.stack([x, y, z], dim=-1) * xyz_ref[None, None, None, None, :]
    
    # XYZ转线性RGB
    rgb_from_xyz = torch.tensor([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ], dtype=lab.dtype, device=lab.device)
    
    # 维度适配：矩阵乘法
    shape = xyz.shape
    xyz_flat = xyz.reshape(-1, 3)  # (N, 3)
    linear_rgb_flat = xyz_flat @ rgb_from_xyz.T  # (N, 3)
    linear_rgb = linear_rgb_flat.reshape(shape)  # 恢复原形状
    
    # 线性RGB转sRGB（伽马校正）
    rgb = torch.where(
        linear_rgb > 0.0031308,
        1.055 * (linear_rgb ** (1/2.4)) - 0.055,
        12.92 * linear_rgb
    )
    
    # 确保输出在[0,1]范围内
    rgb = torch.clamp(rgb, 0.0, 1.0)
    return rgb

def match_and_blend_colors_torch(
    source_chunk: torch.Tensor, 
    reference_image: torch.Tensor, 
    strength: float
) -> torch.Tensor:
    """
    全GPU批量运算版本：将视频chunk的颜色匹配到参考图像并混合（支持B>1、T帧并行）
    
    Args:
        source_chunk (torch.Tensor): 视频chunk (B, C, T, H, W)，范围[-1, 1]
        reference_image (torch.Tensor): 参考图像 (B, C, 1, H, W)，范围[-1, 1]（B需与source_chunk一致）
        strength (float): 颜色校正强度 (0.0-1.0)，0.0无校正，1.0完全校正
    
    Returns:
        torch.Tensor: 颜色校正后的视频chunk (B, C, T, H, W)，范围[-1, 1]
    """
    # 强度为0直接返回原图
    if strength <= 0.0:
        return source_chunk.clone()
    
    # 验证强度范围
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"Strength必须在0.0-1.0之间，当前值：{strength}")
    
    # 验证输入形状（确保B一致，参考图T=1）
    B, C, T, H, W = source_chunk.shape
    assert reference_image.shape == (B, C, 1, H, W), \
        f"参考图像形状需为(B, C, 1, H, W)，当前为{reference_image.shape}"
    assert C == 3, f"仅支持3通道RGB图像，当前通道数：{C}"
    
    # 保持设备和数据类型一致
    device = source_chunk.device
    dtype = source_chunk.dtype
    reference_image = reference_image.to(device=device, dtype=dtype)
    
    # 1. 从[-1,1]转换到[0,1]（GPU上直接运算）
    source_01 = (source_chunk + 1.0) / 2.0
    ref_01 = (reference_image + 1.0) / 2.0
    
    # 2. 调整维度顺序：(B, C, T, H, W) → (B, T, H, W, C)（适配颜色空间转换）
    # 参考图：(B, C, 1, H, W) → (B, 1, H, W, C)
    source_permuted = source_01.permute(0, 2, 3, 4, 1)  # 通道移到最后一维
    ref_permuted = ref_01.permute(0, 2, 3, 4, 1)
    
    # 3. RGB转Lab（批量处理所有帧）
    source_lab = rgb_to_lab_torch(source_permuted)
    ref_lab = rgb_to_lab_torch(ref_permuted)  # (B, 1, H, W, 3)
    
    # 4. 批量颜色迁移：匹配L/a/b通道的均值和标准差（核心逻辑）
    # 计算参考图各通道的均值和标准差（对H、W维度求统计，保持B维度）
    ref_mean = ref_lab.mean(dim=[2, 3], keepdim=True)  # (B, 1, 1, 1, 3)
    ref_std = ref_lab.std(dim=[2, 3], keepdim=True, unbiased=False)  # (B, 1, 1, 1, 3)
    
    # 计算源视频各通道的均值和标准差（对H、W维度求统计，保持B、T维度）
    source_mean = source_lab.mean(dim=[2, 3], keepdim=True)  # (B, T, 1, 1, 3)
    source_std = source_lab.std(dim=[2, 3], keepdim=True, unbiased=False)  # (B, T, 1, 1, 3)
    
    # 避免标准差为0的除法错误（用1.0替代0）
    source_std_safe = torch.where(source_std < 1e-8, torch.ones_like(source_std), source_std)
    
    # 颜色迁移公式：(源 - 源均值) * (参考标准差/源标准差) + 参考均值
    corrected_lab = (source_lab - source_mean) * (ref_std / source_std_safe) + ref_mean
    
    # 5. Lab转RGB（批量转换所有校正后的帧）
    corrected_rgb_01 = lab_to_rgb_torch(corrected_lab)
    
    # 6. 批量混合原始帧和校正帧（按强度加权）
    blended_rgb_01 = (1 - strength) * source_permuted + strength * corrected_rgb_01
    
    # 7. 还原维度顺序和数值范围：(B, T, H, W, C) → (B, C, T, H, W)，范围[0,1]→[-1,1]
    blended_rgb_01 = blended_rgb_01.permute(0, 4, 1, 2, 3)  # 通道移回第二维
    blended_rgb_minus1_1 = (blended_rgb_01 * 2.0) - 1.0
    
    # 8. 确保输出格式正确（连续内存布局）
    output = blended_rgb_minus1_1.contiguous().to(device=device, dtype=dtype)
    
    return output

def resize_and_centercrop(cond_image, target_size):
    """
    Resize image or tensor to the target size without padding.
    """

    # Get the original size
    if isinstance(cond_image, torch.Tensor):
        _, orig_h, orig_w = cond_image.shape
    else:
        orig_h, orig_w = cond_image.height, cond_image.width

    target_h, target_w = target_size
    
    # Calculate the scaling factor for resizing
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w
    
    # Compute the final size
    scale = max(scale_h, scale_w)
    final_h = math.ceil(scale * orig_h)
    final_w = math.ceil(scale * orig_w)
    
    # Resize
    if isinstance(cond_image, torch.Tensor):
        if len(cond_image.shape) == 3:
            cond_image = cond_image[None]
        resized_tensor = nn.functional.interpolate(cond_image, size=(final_h, final_w), mode='nearest').contiguous() 
        # crop
        cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size) 
        cropped_tensor = cropped_tensor.squeeze(0)
    else:
        resized_image = cond_image.resize((final_w, final_h), resample=Image.BILINEAR)
        resized_image = np.array(resized_image)
        # tensor and crop
        resized_tensor = torch.from_numpy(resized_image)[None, ...].permute(0, 3, 1, 2).contiguous()
        cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
        cropped_tensor = cropped_tensor[:, :, None, :, :] 

    return cropped_tensor