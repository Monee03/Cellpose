from pathlib import Path
import shutil
import csv
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT = Path("./dataset_cellvitpp").resolve()

DATASETS = {
    "consep": {
        "tissue": "Colon",
        "nuclei_weight_name": "Neoplastic",
    },
    "lizard": {
        "tissue": "Colon",
        "nuclei_weight_name": "Neoplastic",
    },
}

SPLIT_TO_FOLD = {
    "train": 0,
    "val": 1,
    "test": 2,
}

NUCLEI_TYPES = [
    "Background",
    "Neoplastic",
    "Inflammatory",
    "Connective",
    "Dead",
    "Epithelial",
]

CELL_COUNT_COLUMNS = [
    "Neoplastic",
    "Inflammatory",
    "Connective",
    "Dead",
    "Epithelial",
]


def load_annotation(path):
    data = np.load(path, allow_pickle=True).item()

    inst_map = data["inst_map"].astype(np.int32)

    if "type_map" in data:
        type_map = data["type_map"].astype(np.int32)
    else:
        type_map = (inst_map > 0).astype(np.int32)

    if inst_map.shape != type_map.shape:
        raise ValueError(
            f"inst_map/type_map 尺寸不一致: {path}, "
            f"{inst_map.shape}, {type_map.shape}"
        )

    return inst_map, type_map


def process_dataset(dataset_name, tissue_name):
    src_root = ROOT / dataset_name
    dst_root = ROOT / f"{dataset_name}_pannuke"

    if not src_root.exists():
        raise FileNotFoundError(src_root)

    dst_root.mkdir(parents=True, exist_ok=True)

    total_images = 0

    for split, fold_id in SPLIT_TO_FOLD.items():
        src_img_dir = src_root / split / "images"
        src_label_dir = src_root / split / "labels"

        dst_fold = dst_root / f"fold{fold_id}"
        dst_img_dir = dst_fold / "images"
        dst_label_dir = dst_fold / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(src_img_dir.glob("*.png"))

        type_rows = []
        count_rows = []

        print(
            f"\n处理 {dataset_name}/{split} -> "
            f"{dst_root.name}/fold{fold_id}: {len(image_files)} 张"
        )

        for img_path in tqdm(image_files):
            stem = img_path.stem
            label_path = src_label_dir / f"{stem}.npy"

            if not label_path.exists():
                raise FileNotFoundError(
                    f"缺少标签: {label_path}"
                )

            image = Image.open(img_path).convert("RGB")
            image_array = np.asarray(image)

            inst_map, type_map = load_annotation(label_path)

            if image_array.shape[:2] != inst_map.shape:
                raise ValueError(
                    f"图像和标签尺寸不一致: {stem}, "
                    f"image={image_array.shape[:2]}, "
                    f"mask={inst_map.shape}"
                )

            # 保存 RGB 图像
            image.save(dst_img_dir / img_path.name)

            # 只保存 PanNukeDataset 必需的两个字段
            output_label = {
                "inst_map": inst_map.astype(np.int32),
                "type_map": type_map.astype(np.int32),
            }

            np.save(
                dst_label_dir / f"{stem}.npy",
                output_label,
                allow_pickle=True,
            )

            instance_ids = np.unique(inst_map)
            instance_ids = instance_ids[instance_ids > 0]

            count_dict = {
                "Image": img_path.name,
                "Neoplastic": 0,
                "Inflammatory": 0,
                "Connective": 0,
                "Dead": 0,
                "Epithelial": 0,
            }

            # 根据每个实例的 type_map 多数投票统计类别
            for instance_id in instance_ids:
                pixels = type_map[inst_map == instance_id]
                pixels = pixels[pixels > 0]

                if len(pixels) == 0:
                    class_id = 1
                else:
                    values, counts = np.unique(
                        pixels, return_counts=True
                    )
                    class_id = int(values[np.argmax(counts)])

                # 超出 CellViT 预训练模型类别范围时统一设为1
                if class_id < 1 or class_id > 5:
                    class_id = 1

                count_dict[CELL_COUNT_COLUMNS[class_id - 1]] += 1

            type_rows.append({
                "img": img_path.name,
                "type": tissue_name,
            })
            count_rows.append(count_dict)

            total_images += 1

        # PanNukeDataset 要求 types.csv
        with open(dst_fold / "types.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["img", "type"])
            writer.writeheader()
            writer.writerows(type_rows)

        # PanNukeDataset.load_cell_count() 读取该文件
        with open(
            dst_fold / "cell_count.csv", "w", newline=""
        ) as f:
            fieldnames = ["Image"] + CELL_COUNT_COLUMNS
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(count_rows)

    # 数据集级配置
    dataset_config = {
        "nuclei_types": {
            0: "Background",
            1: "Neoplastic",
            2: "Inflammatory",
            3: "Connective",
            4: "Dead",
            5: "Epithelial",
        },
        "tissue_types": {
            0: tissue_name,
        },
    }

    with open(dst_root / "dataset_config.yaml", "w") as f:
        yaml.safe_dump(
            dataset_config,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    with open(dst_root / "weight_config.yaml", "w") as f:
        yaml.safe_dump(
            {"tissue": {tissue_name: total_images}},
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"\n完成: {dst_root}")


if __name__ == "__main__":
    for dataset_name, cfg in DATASETS.items():
        process_dataset(dataset_name, cfg["tissue"])