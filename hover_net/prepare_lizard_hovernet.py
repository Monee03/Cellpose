# prepare_lizard_hovernet.py
"""
Lizard 数据集切块与 Hover-Net 训练格式转换脚本
将 [H, W, 3] 原图和 [H, W] 单通道实例图，切块融合成 [500, 500, 5] 的 .npy 训练矩阵
"""

import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

# 路径配置（对齐您的本地路径）
SRC_TRAIN_IMG = "../edgetam/dataset/Lizard/images/train/"
SRC_TRAIN_MSK = "../edgetam/dataset/Lizard/masks/train/"
SRC_VAL_IMG   = "../edgetam/dataset/Lizard/images/val/"
SRC_VAL_MSK   = "../edgetam/dataset/Lizard/masks/val/"

OUT_DIR_TRAIN = "./dataset_hovernet/lizard/train/"
OUT_DIR_VAL   = "./dataset_hovernet/lizard/val/"

PATCH_SIZE = 500  # 官方要求的标准训练尺寸
STRIDE     = 250  # 重叠步长

def extract_patches(img_dir, msk_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    
    patch_count = 0
    print(f"正在转换并切分数据至: {out_dir} ...")
    for img_path in tqdm(img_files):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        msk_path = os.path.join(msk_dir, f"{stem}.png")
        if not os.path.exists(msk_path):
            continue
            
        # 读取图像和实例图
        img = np.array(Image.open(img_path).convert("RGB"))
        msk = np.array(Image.open(msk_path))
        if msk.ndim == 3:
            msk = msk[:, :, 0]
        msk = msk.astype(np.int32)
        
        H, W = img.shape[:2]
        
        # 滑动窗口切块
        for y in range(0, H - PATCH_SIZE + 1, STRIDE):
            for x in range(0, W - PATCH_SIZE + 1, STRIDE):
                img_patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                msk_patch = msk[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                
                # 过滤不包含任何细胞核的纯背景空白块（加速收敛）
                if msk_patch.max() == 0:
                    continue
                
                # 构建 [500, 500, 5] 的经典矩阵：
                # Channel 0-2: RGB
                # Channel 3: inst_map (实例图)
                # Channel 4: type_map (分类图，如果没有多分类，默认设为二值前景 1)
                type_patch = (msk_patch > 0).astype(np.int32)
                
                combined_patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 5), dtype=np.uint8)
                combined_patch[..., :3] = img_patch
                combined_patch[..., 3]  = msk_patch.astype(np.uint8)
                combined_patch[..., 4]  = type_patch.astype(np.uint8)
                
                # 保存为 .npy
                out_path = os.path.join(out_dir, f"{stem}_{y}_{x}.npy")
                np.save(out_path, combined_patch)
                patch_count += 1
                
    print(f"  ✓ 转换完成！共生成 {patch_count} 个 .npy 训练分块。")

if __name__ == "__main__":
    # 处理训练集
    extract_patches(SRC_TRAIN_IMG, SRC_TRAIN_MSK, OUT_DIR_TRAIN)
    # 处理验证集
    extract_patches(SRC_VAL_IMG, SRC_VAL_MSK, OUT_DIR_VAL)