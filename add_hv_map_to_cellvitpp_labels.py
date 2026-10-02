import os
import glob
import argparse
import numpy as np
from tqdm import tqdm


def generate_hv_map(inst_map):
    """
    根据 instance map 生成 HoVer-Net / CellViT 常用的 HV map。
    输出 shape: (2, H, W)
    hv_map[0] = horizontal map
    hv_map[1] = vertical map
    """
    inst_map = inst_map.astype(np.int32)
    H, W = inst_map.shape

    hv_map = np.zeros((2, H, W), dtype=np.float32)

    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]

    for inst_id in inst_ids:
        mask = inst_map == inst_id
        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        cx = xs.mean()
        cy = ys.mean()

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        width = max(x_max - x_min, 1)
        height = max(y_max - y_min, 1)

        # 归一化到大约 [-1, 1]
        hv_map[0, ys, xs] = (xs - cx) / width
        hv_map[1, ys, xs] = (ys - cy) / height

    return hv_map


def process_label_file(label_path):
    label = np.load(label_path, allow_pickle=True).item()

    if "inst_map" not in label:
        raise KeyError(f"{label_path} 中没有 inst_map")

    inst_map = label["inst_map"].astype(np.int32)

    hv_map = generate_hv_map(inst_map)

    label["hv_map"] = hv_map
    label["nuclei_binary_map"] = (inst_map > 0).astype(np.int64)
    label["nuclei_type_map"] = label["type_map"].astype(np.int64)

    np.save(label_path, label, allow_pickle=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="./dataset_cellvitpp",
        help="预处理后的 CellViT++ 数据根目录"
    )
    args = parser.parse_args()

    label_files = glob.glob(os.path.join(args.root, "*", "*", "labels", "*.npy"))

    print(f"找到 label 文件数: {len(label_files)}")

    for lf in tqdm(label_files):
        process_label_file(lf)

    print("全部 hv_map 添加完成。")


if __name__ == "__main__":
    main()
