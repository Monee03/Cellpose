import argparse
import csv
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# 与官方 PanNuke dataset_config.yaml 完全一致：名称 -> 索引
PANNUKE_TISSUES = [
    "Adrenal_gland", "Bile-duct", "Bladder", "Breast", "Cervix", "Colon",
    "Esophagus", "HeadNeck", "Kidney", "Liver", "Lung", "Ovarian",
    "Pancreatic", "Prostate", "Skin", "Stomach", "Testis", "Thyroid", "Uterus",
]
PANNUKE_NUCLEI = ["Background", "Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]


def check_fold(fold_dir: Path, tissue: str, rewrite_types: bool):
    img_dir, lbl_dir = fold_dir / "images", fold_dir / "labels"
    problems = []

    if not img_dir.is_dir() or not lbl_dir.is_dir():
        return 0, {}, [f"{fold_dir.name}: 缺少 images/ 或 labels/ 目录"]

    images = sorted(img_dir.glob("*.png"))
    if not images:
        return 0, {}, [f"{fold_dir.name}: images/ 下没有 png"]

    sizes = {}
    for p in images:
        lbl = lbl_dir / f"{p.stem}.npy"
        if not lbl.exists():
            problems.append(f"{fold_dir.name}: 缺少标签 {lbl.name}")
            continue
        with Image.open(p) as im:
            w, h = im.size
            mode = im.mode
        sizes[(h, w)] = sizes.get((h, w), 0) + 1
        if mode != "RGB":
            problems.append(f"{fold_dir.name}: {p.name} 不是 RGB (mode={mode})")

        try:
            d = np.load(lbl, allow_pickle=True)[()]
            inst, typ = d["inst_map"], d["type_map"]
        except Exception as e:
            problems.append(f"{fold_dir.name}: {lbl.name} 读取失败或缺少 inst_map/type_map ({e})")
            continue
        if inst.shape != (h, w) or typ.shape != (h, w):
            problems.append(f"{fold_dir.name}: {p.name} 图像{(h, w)} 与标签 inst{inst.shape}/type{typ.shape} 尺寸不一致")
        if typ.max() > 5 or typ.min() < 0:
            problems.append(f"{fold_dir.name}: {lbl.name} type_map 取值超出 0~5 (max={typ.max()})")

    # types.csv：列必须是 img,type；img 是带扩展名的文件名；type 必须在 tissue 映射里
    types_csv = fold_dir / "types.csv"
    if rewrite_types or not types_csv.exists():
        with open(types_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["img", "type"])
            w.writeheader()
            w.writerows({"img": p.name, "type": tissue} for p in images)
        print(f"  已写入 {types_csv}")
    else:
        with open(types_csv) as f:
            rows = list(csv.DictReader(f))
        if not rows or set(rows[0].keys()) != {"img", "type"}:
            problems.append(f"{fold_dir.name}: types.csv 列名必须是 img,type（可加 --rewrite_types_csv 重写）")
        else:
            bad = {r["type"] for r in rows if r["type"] not in PANNUKE_TISSUES}
            if bad:
                problems.append(f"{fold_dir.name}: types.csv 含未知组织名 {bad}（可加 --rewrite_types_csv 重写）")
            listed = {r["img"] for r in rows}
            missing = [p.name for p in images if p.name not in listed]
            if missing:
                problems.append(f"{fold_dir.name}: types.csv 缺少 {len(missing)} 张图（可加 --rewrite_types_csv 重写）")

    return len(images), sizes, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, help="含 fold0/fold1/fold2 的目录")
    ap.add_argument("--tissue", default="Colon")
    ap.add_argument("--rewrite_types_csv", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset_root).resolve()
    assert root.is_dir(), f"目录不存在: {root}"
    assert args.tissue in PANNUKE_TISSUES, f"tissue 必须是 {PANNUKE_TISSUES} 之一"

    total, all_problems = 0, []
    for fold in (0, 1, 2):
        n, sizes, problems = check_fold(root / f"fold{fold}", args.tissue, args.rewrite_types_csv)
        total += n
        all_problems += problems
        note = ""
        if fold in (0, 1) and sizes and set(sizes) != {(256, 256)}:
            note = "  <-- 训练/验证 fold 不是 256x256，需重新切 patch 或调整 input_shape"
        print(f"fold{fold}: {n} 张, 尺寸分布 {sizes}{note}")

    # 关键：名称 -> 索引
    with open(root / "dataset_config.yaml", "w") as f:
        yaml.safe_dump(
            {"tissue_types": {t: i for i, t in enumerate(PANNUKE_TISSUES)},
             "nuclei_types": {n: i for i, n in enumerate(PANNUKE_NUCLEI)}},
            f, sort_keys=False)
    with open(root / "weight_config.yaml", "w") as f:
        yaml.safe_dump({"tissue": {args.tissue: total}}, f)

    print(f"\n已写入: {root/'dataset_config.yaml'}")
    print(f"已写入: {root/'weight_config.yaml'}")

    if all_problems:
        print(f"\n发现 {len(all_problems)} 个问题（前 20 条）:")
        for p in all_problems[:20]:
            print("  -", p)
        raise SystemExit(1)
    print("\n校验通过，可以训练。")


if __name__ == "__main__":
    main()