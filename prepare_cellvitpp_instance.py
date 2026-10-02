import os
import glob
import argparse
import shutil
import json
import numpy as np
from PIL import Image
from tqdm import tqdm


DATA_ROOT = "../edgetam/dataset"
OUT_ROOT = "./dataset_cellvitpp"


DATASET_MAP = {
    "consep": "Consep",
    "lizard": "Lizard",
}


def relabel_instance(mask: np.ndarray) -> np.ndarray:
    """
    将实例标签重新编号为连续整数：
    原始: [0, 3, 7, 10]
    转换: [0, 1, 2, 3]
    """
    mask = mask.astype(np.int32)
    out = np.zeros_like(mask, dtype=np.int32)

    ids = np.unique(mask)
    ids = ids[ids > 0]

    for new_id, old_id in enumerate(ids, start=1):
        out[mask == old_id] = new_id

    return out


def read_mask(mask_path):
    mask = np.array(Image.open(mask_path))

    # 如果是 RGB mask，默认取第一个通道。
    # 如果你的实例ID是RGB编码的，需要在这里改成RGB解码。
    if mask.ndim == 3:
        mask = mask[..., 0]

    return mask.astype(np.int32)


def get_instance_info(inst_map):
    """
    根据 inst_map 生成实例信息：
    - inst_type: 每个实例类别。这里只做实例分割，全部设为1。
    - inst_centroid: 每个实例中心点 [x, y]
    - inst_bbox: 每个实例 bbox [x_min, y_min, x_max, y_max]
    """
    inst_type = {}
    inst_centroid = {}
    inst_bbox = {}

    ids = np.unique(inst_map)
    ids = ids[ids > 0]

    for inst_id in ids:
        ys, xs = np.where(inst_map == inst_id)

        if len(xs) == 0:
            continue

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        cx = float(xs.mean())
        cy = float(ys.mean())

        inst_type[int(inst_id)] = 1
        inst_centroid[int(inst_id)] = [cx, cy]
        inst_bbox[int(inst_id)] = [x_min, y_min, x_max, y_max]

    return inst_type, inst_centroid, inst_bbox


def save_label_npy(out_path, inst_map):
    """
    保存 CellViT++ 可用的 label npy。
    只做实例分割时，type_map 中所有细胞核统一设置为1。
    """
    inst_map = relabel_instance(inst_map)

    type_map = (inst_map > 0).astype(np.int32)

    inst_type, inst_centroid, inst_bbox = get_instance_info(inst_map)

    label = {
        "inst_map": inst_map.astype(np.int32),
        "type_map": type_map.astype(np.int32),
        "inst_type": inst_type,
        "inst_centroid": inst_centroid,
        "inst_bbox": inst_bbox,
    }

    np.save(out_path, label, allow_pickle=True)


def get_tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))

    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)

    return starts


def tile_image_and_mask(image, mask, tile_size=256, overlap=0):
    """
    训练/验证用切片。
    如果原图已经小于等于 tile_size，则直接返回整图。
    """
    H, W = mask.shape
    stride = tile_size - overlap

    if stride <= 0:
        raise ValueError("overlap 必须小于 tile_size")

    ys = get_tile_starts(H, tile_size, stride)
    xs = get_tile_starts(W, tile_size, stride)

    tiles = []

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)

            img_tile = image[y0:y1, x0:x1]
            mask_tile = mask[y0:y1, x0:x1]

            tiles.append((img_tile, mask_tile, y0, x0))

    return tiles


def process_split(
    dataset_key,
    split,
    tile_train_val=True,
    patch_size=256,
    overlap=0,
    min_instances=1,
):
    dataset_name = DATASET_MAP[dataset_key]

    img_dir = os.path.join(DATA_ROOT, dataset_name, "images", split)
    mask_dir = os.path.join(DATA_ROOT, dataset_name, "masks", split)

    out_img_dir = os.path.join(OUT_ROOT, dataset_key, split, "images")
    out_label_dir = os.path.join(OUT_ROOT, dataset_key, split, "labels")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    if len(img_files) == 0:
        print(f"[跳过] 未找到图像: {img_dir}")
        return

    total_saved = 0
    total_skipped = 0
    inst_counts = []

    print(f"\n========== Processing {dataset_key} / {split} ==========")
    print(f"image_dir: {img_dir}")
    print(f"mask_dir : {mask_dir}")
    print(f"out_dir  : {os.path.join(OUT_ROOT, dataset_key, split)}")

    for img_path in tqdm(img_files):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, stem + ".png")

        if not os.path.exists(mask_path):
            print(f"[警告] 找不到 mask: {mask_path}")
            total_skipped += 1
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = read_mask(mask_path)

        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"图像和mask尺寸不一致: {img_path}, "
                f"image={image.shape}, mask={mask.shape}"
            )

        # train/val 可以切 patch，test 建议保持整图，便于最终 whole-map 评估
        do_tile = tile_train_val and split in ["train", "val"]

        if do_tile:
            tiles = tile_image_and_mask(
                image=image,
                mask=mask,
                tile_size=patch_size,
                overlap=overlap,
            )
        else:
            tiles = [(image, mask, 0, 0)]

        for idx, (img_tile, mask_tile, y0, x0) in enumerate(tiles):
            mask_tile = relabel_instance(mask_tile)

            n_inst = len(np.unique(mask_tile)) - 1

            # train/val 中空 patch 可以跳过
            if split in ["train", "val"] and n_inst < min_instances:
                continue

            inst_counts.append(n_inst)

            if do_tile:
                out_stem = f"{stem}_y{y0}_x{x0}"
            else:
                out_stem = stem

            out_img_path = os.path.join(out_img_dir, out_stem + ".png")
            out_label_path = os.path.join(out_label_dir, out_stem + ".npy")

            Image.fromarray(img_tile).save(out_img_path)
            save_label_npy(out_label_path, mask_tile)

            total_saved += 1

    if len(inst_counts) > 0:
        print(
            f"保存样本数: {total_saved}, 跳过: {total_skipped}, "
            f"实例数 min={np.min(inst_counts)}, "
            f"median={np.median(inst_counts):.1f}, "
            f"max={np.max(inst_counts)}"
        )
    else:
        print(f"[警告] 没有保存任何样本。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["consep", "lizard", "all"],
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="train/val 切片大小"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="train/val 切片重叠大小"
    )
    parser.add_argument(
        "--min_instances",
        type=int,
        default=1,
        help="train/val patch 至少包含多少个实例"
    )
    parser.add_argument(
        "--no_tile_train_val",
        action="store_true",
        help="不对 train/val 切片，直接整图保存"
    )

    args = parser.parse_args()

    datasets = ["consep", "lizard"] if args.dataset == "all" else [args.dataset]

    for d in datasets:
        for split in ["train", "val", "test"]:
            process_split(
                dataset_key=d,
                split=split,
                tile_train_val=not args.no_tile_train_val,
                patch_size=args.patch_size,
                overlap=args.overlap,
                min_instances=args.min_instances,
            )

    # 保存一个简单 config，方便之后写训练配置时查路径
    config = {
        "root": OUT_ROOT,
        "datasets": {
            "consep": os.path.join(OUT_ROOT, "consep"),
            "lizard": os.path.join(OUT_ROOT, "lizard"),
        },
        "label_format": {
            "inst_map": "HxW int32, 0=background, 1..N=instances",
            "type_map": "HxW int32, 0=background, 1=nucleus",
            "inst_type": "dict, all instances are class 1",
            "inst_centroid": "dict, [x, y]",
            "inst_bbox": "dict, [x_min, y_min, x_max, y_max]",
        }
    }

    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(os.path.join(OUT_ROOT, "dataset_cellvitpp_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\n全部预处理完成。")
    print(f"输出目录: {OUT_ROOT}")


if __name__ == "__main__":
    main()
