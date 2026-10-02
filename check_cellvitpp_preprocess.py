import os
import glob
import argparse
import numpy as np
from PIL import Image


def check(root):
    print(f"\n========== Checking: {root} ==========")

    img_dir = os.path.join(root, "images")
    lab_dir = os.path.join(root, "labels")

    print("image dir:", img_dir, "exists:", os.path.isdir(img_dir))
    print("label dir:", lab_dir, "exists:", os.path.isdir(lab_dir))

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    label_files = sorted(glob.glob(os.path.join(lab_dir, "*.npy")))

    print("images:", len(img_files))
    print("labels:", len(label_files))

    if len(img_files) == 0 or len(label_files) == 0:
        print("[警告] images 或 labels 为空")
        return

    img_stems = {os.path.splitext(os.path.basename(p))[0] for p in img_files}
    lab_stems = {os.path.splitext(os.path.basename(p))[0] for p in label_files}

    missing_labels = sorted(img_stems - lab_stems)
    missing_images = sorted(lab_stems - img_stems)

    print("missing labels:", len(missing_labels))
    print("missing images:", len(missing_images))

    if missing_labels[:5]:
        print("前几个缺 label 的 image:", missing_labels[:5])
    if missing_images[:5]:
        print("前几个缺 image 的 label:", missing_images[:5])

    # 检查前 3 个 label
    for lf in label_files[:3]:
        stem = os.path.splitext(os.path.basename(lf))[0]
        img_path = os.path.join(img_dir, stem + ".png")

        print(f"\n--- sample: {stem} ---")
        print("image exists:", os.path.exists(img_path))
        print("label path:", lf)

        label = np.load(lf, allow_pickle=True).item()

        print("keys:", label.keys())

        inst_map = label["inst_map"]
        type_map = label["type_map"]

        if os.path.exists(img_path):
            img = np.array(Image.open(img_path).convert("RGB"))
            print("image shape:", img.shape)
        else:
            img = None

        print("inst_map shape:", inst_map.shape, "dtype:", inst_map.dtype)
        print("type_map shape:", type_map.shape, "dtype:", type_map.dtype)

        n_inst = len(np.unique(inst_map)) - 1
        print("n_inst:", n_inst)
        print("type ids:", np.unique(type_map))

        ids = np.unique(inst_map)
        fg_ids = ids[ids > 0]
        if len(fg_ids) > 0:
            continuous = fg_ids.max() == len(fg_ids)
            print("instance labels continuous:", continuous)
        else:
            print("[警告] 该样本没有实例")

        if img is not None and img.shape[:2] != inst_map.shape:
            print("[错误] image 和 inst_map 尺寸不一致！")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="./dataset_cellvitpp",
        help="CellViT++ 预处理后的数据根目录"
    )
    args = parser.parse_args()

    root = args.root

    print("当前工作目录:", os.getcwd())
    print("输入 root:", root)
    print("root 绝对路径:", os.path.abspath(root))
    print("root exists:", os.path.isdir(root))

    if not os.path.isdir(root):
        print("[错误] root 不存在，请检查路径")
        return

    for dataset in ["consep", "lizard"]:
        for split in ["train", "val", "test"]:
            split_root = os.path.join(root, dataset, split)
            print(f"\n准备检查: {split_root}")

            if os.path.isdir(split_root):
                check(split_root)
            else:
                print("[跳过] 目录不存在:", split_root)


if __name__ == "__main__":
    main()