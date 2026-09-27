import os, argparse, glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from cellpose import models, train, io, utils

io.logger_setup()          # ★★★ 必须！否则看不到任何训练日志
"""
python train_cellpose.py --dataset consep --epochs 150
python train_cellpose.py --dataset lizard --epochs 150 --lr 0.001 --batch_size 8

"""

DATASET_DENSE_ROOT = "./edgetam/dataset"
DATASET_CONFIG = {
    "consep": dict(
        train_img=f"{DATASET_DENSE_ROOT}/Consep/images/train",
        train_msk=f"{DATASET_DENSE_ROOT}/Consep/masks/train",
        val_img=f"{DATASET_DENSE_ROOT}/Consep/images/val",
        val_msk=f"{DATASET_DENSE_ROOT}/Consep/masks/val"),
    "lizard": dict(
        train_img=f"{DATASET_DENSE_ROOT}/Lizard/images/train",
        train_msk=f"{DATASET_DENSE_ROOT}/Lizard/masks/train",
        val_img=f"{DATASET_DENSE_ROOT}/Lizard/images/val",
        val_msk=f"{DATASET_DENSE_ROOT}/Lizard/masks/val"),
}


def load_mask(p):
    """保证读到的是实例标签，且不被 8bit 截断"""
    if p.endswith(".npy"):
        m = np.load(p)
    else:
        m = np.array(Image.open(p))
    if m.ndim == 3:            # 若是 RGB 编码的实例图，需按你的编码方式还原
        m = m[..., 0].astype(np.int32) + m[..., 1].astype(np.int32) * 256
    return m.astype(np.int32)


def tile(img, msk, size=256, stride=192):
    """把大图切成小 patch，显著增加样本量"""
    H, W = msk.shape
    out_i, out_m = [], []
    for y in range(0, max(H - size, 0) + 1, stride):
        for x in range(0, max(W - size, 0) + 1, stride):
            sm = msk[y:y+size, x:x+size]
            if len(np.unique(sm)) - 1 < 2:      # patch 内实例太少就丢
                continue
            out_i.append(img[y:y+size, x:x+size])
            out_m.append(sm)
    return out_i, out_m


def load_dataset(img_dir, msk_dir, do_tile=True):
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    images, masks = [], []
    for f in tqdm(files, desc=f"load {img_dir}"):
        stem = os.path.splitext(os.path.basename(f))[0]
        mp = None
        for ext in (".png", ".tif", ".tiff", ".npy"):
            cand = os.path.join(msk_dir, stem + ext)
            if os.path.exists(cand):
                mp = cand; break
        if mp is None:
            continue
        img = np.array(Image.open(f).convert("RGB"))
        msk = load_mask(mp)
        if do_tile and min(msk.shape) > 320:
            ti, tm = tile(img, msk)
            images += ti; masks += tm
        else:
            images.append(img); masks.append(msk)
    # 统计，确认掩码是实例图
    ninst = [len(np.unique(m)) - 1 for m in masks]
    print(f"样本数={len(images)}  每图实例数: min={min(ninst)} med={int(np.median(ninst))} max={max(ninst)}")
    return images, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="consep", choices=["consep", "lizard"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)      # ★ AdamW，不要 0.1
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--pretrained", default="nuclei")      # 病理核用 nuclei 更合适
    ap.add_argument("--save_dir", default="./cellpose_checkpoints")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    model = models.CellposeModel(gpu=torch.cuda.is_available(),
                                 model_type=args.pretrained)

    cfg = DATASET_CONFIG[args.dataset]
    tr_x, tr_y = load_dataset(cfg["train_img"], cfg["train_msk"])
    va_x, va_y = load_dataset(cfg["val_img"], cfg["val_msk"])

    # 估计核直径，用于 rescale
    diams = np.array([utils.diameters(m)[0] for m in tr_y])
    print("训练集中位核直径 =", np.median(diams))

    model_path, tl, vl = train.train_seg(
        net=model.net,
        train_data=tr_x, train_labels=tr_y,
        test_data=va_x, test_labels=va_y,
        channels=[0, 0],
        normalize=True,
        rescale=True,
        min_train_masks=1,          # ★ 别让它偷偷丢图
        n_epochs=args.epochs,
        learning_rate=args.lr,      # ★
        weight_decay=1e-5,
        SGD=False,                  # AdamW
        batch_size=args.batch_size,
        nimg_per_epoch=max(len(tr_x), 200),   # 小数据集时每 epoch 重复采样
        save_path=args.save_dir,
        model_name=f"cellpose_{args.dataset}",   # ← 自动区分 consep / lizard
        save_every=25,
        save_each=True,      # ★ 每 25 轮存成独立文件 cellpose_consep_epoch_0025 / 0050 / ...
    )
    print("权重:", model_path, "final train loss:", tl[-1], "val loss:", vl[-1])


if __name__ == "__main__":
    main()