#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CellViT 整图推理 + 统一 whole-map 评估（CoNSeP / Lizard）

  python infer_cellvit_wholemap.py --dataset consep \
      --checkpoint ./logs_local/2026-09-04T021153_consep/checkpoints/model_best.pth
  python infer_cellvit_wholemap.py --dataset lizard \
      --checkpoint ./logs_local/<run>_lizard/checkpoints/model_best.pth
  冒烟测试只跑 1 张、不写 CSV：加 --limit 1

流程：整图 → 256×256 重叠滑窗(batch 前向)
      → softmax(binary)/softmax(type)/hv 三张预测图按线性渐变权重拼回整图
      → 在整图上做一次 CellViT 自带的 HoVer-Net 式后处理 → pred_inst
      → 保存 .npy/.png → Dice/IoU/Prec/Rec/AJI/PQ/DQ/SQ（与 cellpose 评估脚本同一实现）→ CSV


      python infer_cellvit_wholemap.py --dataset lizard\
  --checkpoint ./logs_local/2026-09-04T025902_lizard/checkpoints/model_best.pth
  python infer_cellvit_wholemap.py --dataset consep \
  --checkpoint ./logs_local/2026-09-04T021153_consep/checkpoints/model_best.pth 
  work/CellViT-plus-plus/myenv
"""
import os
import sys
import csv
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cellvit.models.cell_segmentation.cellvit import CellViT  # noqa: E402

DEFAULT_MAG = {"consep": 40, "lizard": 20}


# ============================================================ 通用
def relabel(m):
    """实例 ID 重编号为 0,1,2,...（0=背景），向量化实现"""
    m = np.asarray(m)
    u, inv = np.unique(m, return_inverse=True)
    inv = inv.reshape(m.shape).astype(np.int32)
    if u.size and u[0] != 0:          # 极端情况：图里没有背景像素
        inv += 1
    return inv


def torch_load_any(path):
    try:                              # torch>=2.6 默认 weights_only=True，训练 ckpt 含 numpy 标量会被拒
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                 # 老版本 torch 没有该参数
        return torch.load(path, map_location="cpu")


# ============================================================ 模型
def load_state_dict_from_ckpt(path):
    ck = torch_load_any(path)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        print(f"[ckpt] epoch={ck.get('epoch')}  best_epoch={ck.get('best_epoch')}  "
              f"best_metric(val bPQ)={ck.get('best_metric')}")
        sd = ck["model_state_dict"]
    elif isinstance(ck, dict) and "state_dict" in ck:
        sd = ck["state_dict"]
    else:
        sd = ck
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def build_model(ckpt, device):
    model = CellViT(num_nuclei_classes=6, num_tissue_classes=19,
                    embed_dim=384, input_channels=3, depth=12, num_heads=6,
                    extract_layers=[3, 6, 9, 12], regression_loss=False)
    print(model.load_state_dict(load_state_dict_from_ckpt(ckpt), strict=True))
    return model.to(device).eval()


# ============================================================ 滑窗：拼接预测图，而不是拼接实例
def tile_starts(length, tile, stride):
    last = max(length - tile, 0)
    s = list(range(0, last + 1, stride))
    if s[-1] != last:
        s.append(last)                # 最后一块贴齐图像边缘
    return s


def ramp_1d(n, ramp, low_is_border, high_is_border):
    """tile 内部边缘用 0→1 线性渐变权重；贴着图像边缘的一侧权重恒为 1"""
    w = np.ones(n, np.float32)
    r = np.linspace(0.0, 1.0, ramp + 2, dtype=np.float32)[1:-1]
    if not low_is_border:
        w[:ramp] = r
    if not high_is_border:
        w[n - ramp:] = r[::-1]
    return w


@torch.no_grad()
def predict_maps(model, img, device, tile=256, overlap=64, batch_size=8):
    """返回整图级预测图 dict(numpy, C×H×W)：nuclei_binary_map(2) / nuclei_type_map(6) / hv_map(2)"""
    H, W = img.shape[:2]
    pad_h, pad_w = max(tile - H, 0), max(tile - W, 0)          # 图像比 tile 小时的保险
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    Hp, Wp = img.shape[:2]

    x_all = (img.astype(np.float32) / 255.0 - 0.5) / 0.5        # 与训练 A.Normalize(mean=.5,std=.5) 一致
    stride = tile - overlap
    coords = [(y, x) for y in tile_starts(Hp, tile, stride)
              for x in tile_starts(Wp, tile, stride)]

    chans = {"nuclei_binary_map": 2, "nuclei_type_map": 6, "hv_map": 2}
    acc = {k: np.zeros((c, Hp, Wp), np.float32) for k, c in chans.items()}
    cnt = np.zeros((Hp, Wp), np.float32)

    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        batch = np.stack([x_all[y:y + tile, x:x + tile] for y, x in chunk])       # B,H,W,3
        t = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(device)
        out = model(t)
        maps = {"nuclei_binary_map": F.softmax(out["nuclei_binary_map"], 1),   # 后处理要求 softmax
                "nuclei_type_map":   F.softmax(out["nuclei_type_map"], 1),
                "hv_map":            out["hv_map"]}                            # hv 保持回归输出
        maps = {k: v.float().cpu().numpy() for k, v in maps.items()}

        for b, (y, x) in enumerate(chunk):
            wy = ramp_1d(tile, overlap, y == 0, y + tile >= Hp)
            wx = ramp_1d(tile, overlap, x == 0, x + tile >= Wp)
            w = np.outer(wy, wx)
            for k in acc:
                acc[k][:, y:y + tile, x:x + tile] += maps[k][b] * w
            cnt[y:y + tile, x:x + tile] += w

    assert (cnt > 0).all(), "有像素未被任何 tile 覆盖，请检查 tile_size/overlap"
    return {k: (v / cnt)[:, :H, :W] for k, v in acc.items()}


def instance_map_from_maps(model, maps, magnification):
    """整图上调用 CellViT 自带的 HoVer-Net 式后处理（Sobel+watershed），返回 (H,W) int32 实例图"""
    pred = {k: torch.from_numpy(np.ascontiguousarray(v))[None] for k, v in maps.items()}  # (1,C,H,W)
    try:
        out = model.calculate_instance_map(pred, magnification=magnification)
    except TypeError:
        out = model.calculate_instance_map(pred, magnification)
    inst = out[0] if isinstance(out, (tuple, list)) else out
    if isinstance(inst, (list, tuple)):
        inst = inst[0]
    inst = inst.cpu().numpy() if torch.is_tensor(inst) else np.asarray(inst)
    if inst.ndim == 3:
        inst = inst[0]
    return relabel(inst.astype(np.int32))


# ============================================================ 指标（与 evaluate_cellpose_wholemap.py 同一实现）
def _areas(inst):
    return np.bincount(inst.ravel())             # areas[id] = 像素数


def compute_aji(pred_inst, gt_inst):
    pred_ids = np.unique(pred_inst); pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt_inst);     gt_ids = gt_ids[gt_ids > 0]
    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0.0
    pa, ga = _areas(pred_inst), _areas(gt_inst)
    inter_sum, union_sum, matched = 0, 0, set()
    for gid in gt_ids:
        gm = gt_inst == gid
        ov = np.unique(pred_inst[gm]); ov = ov[ov > 0]
        best_iou, best_pid, best_inter = -1.0, -1, 0
        for pid in ov:
            inter = int((gm & (pred_inst == pid)).sum())
            union = int(ga[gid] + pa[pid] - inter)
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_pid, best_inter = iou, int(pid), inter
        if best_iou > 0:
            inter_sum += best_inter
            union_sum += int(ga[gid] + pa[best_pid] - best_inter)
            matched.add(best_pid)
        else:
            union_sum += int(ga[gid])
    for pid in set(pred_ids.tolist()) - matched:
        union_sum += int(pa[pid])
    return inter_sum / union_sum if union_sum > 0 else 0.0


def compute_pq(pred_inst, gt_inst):
    pred_ids = np.unique(pred_inst); pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt_inst);     gt_ids = gt_ids[gt_ids > 0]
    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0, 1.0, 1.0
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0.0, 0.0, 0.0
    pa, ga = _areas(pred_inst), _areas(gt_inst)
    tp, sum_iou, matched_pred, matched_gt = 0, 0.0, set(), set()
    for gid in gt_ids:
        gm = gt_inst == gid
        ov = np.unique(pred_inst[gm]); ov = ov[ov > 0]
        for pid in ov:
            pid = int(pid)
            if pid in matched_pred:
                continue
            inter = int((gm & (pred_inst == pid)).sum())
            union = int(ga[gid] + pa[pid] - inter)
            iou = inter / union if union > 0 else 0.0
            if iou > 0.5:
                tp += 1; sum_iou += iou
                matched_pred.add(pid); matched_gt.add(int(gid))
                break
    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - len(matched_gt)
    den = tp + 0.5 * fp + 0.5 * fn
    dq = tp / den if den > 0 else 0.0
    sq = sum_iou / tp if tp > 0 else 0.0
    return dq * sq, dq, sq


def compute_metrics(pred_inst, gt_inst):
    pb, gb = pred_inst > 0, gt_inst > 0
    inter, union = int((pb & gb).sum()), int((pb | gb).sum())
    ps, gs = int(pb.sum()), int(gb.sum())
    pq, dq, sq = compute_pq(pred_inst, gt_inst)
    return {"dice": 2 * inter / (ps + gs) if ps + gs > 0 else 1.0,
            "iou": inter / union if union > 0 else 1.0,
            "precision": inter / ps if ps > 0 else 1.0,
            "recall": inter / gs if gs > 0 else 1.0,
            "aji": compute_aji(pred_inst, gt_inst),
            "pq": pq, "dq": dq, "sq": sq}


def append_summary(path, row):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    header = list(row.keys())
    if exists:
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        if set(header) != set(row):
            print(f"[warn] {path} 表头与当前字段不一致，按文件已有表头写入（缺失填 N/A）")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, restval="N/A", extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"[csv ] 已追加 -> {path}")


# ============================================================ 主流程
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["consep", "lizard"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--magnification", type=int, choices=[20, 40], default=None,
                    help="后处理的核尺寸阈值；默认 consep=40, lizard=20")
    ap.add_argument("--data_root", default="./dataset_cellvitpp",
                    help="包含 <dataset>_pannuke/fold2/{images,labels} 的目录")
    ap.add_argument("--image_dir", default=None, help="覆盖测试图目录")
    ap.add_argument("--label_dir", default=None, help="覆盖 GT(.npy 含 inst_map) 目录")
    ap.add_argument("--pred_dir", default=None, help="默认 ./results_cellvit/<dataset>/pred_inst")
    ap.add_argument("--output_dir", default="./results_wholemap", help="summary_wholemap.csv 所在目录")
    ap.add_argument("--model_tag", default=None, help="CSV 中 model 列，默认 cellvit256_ft_x<mag>")
    ap.add_argument("--tile_size", type=int, default=256, help="训练 patch 尺寸，保持 256")
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张做冒烟测试（不写 CSV）")
    a = ap.parse_args()

    mag = a.magnification or DEFAULT_MAG[a.dataset]
    assert a.tile_size % 16 == 0 and 0 < a.overlap < a.tile_size, "tile_size 需为 16 的倍数且 0<overlap<tile_size"

    root = Path(a.data_root) / f"{a.dataset}_pannuke" / "fold2"
    img_dir = Path(a.image_dir) if a.image_dir else root / "images"
    lab_dir = Path(a.label_dir) if a.label_dir else root / "labels"
    pred_dir = Path(a.pred_dir) if a.pred_dir else Path("./results_cellvit") / a.dataset / "pred_inst"
    assert img_dir.is_dir(), f"测试图目录不存在: {img_dir.resolve()}"
    assert lab_dir.is_dir(), f"GT 目录不存在: {lab_dir.resolve()}"
    img_files = sorted(img_dir.glob("*.png"))
    assert img_files, f"{img_dir.resolve()} 下没有 .png"
    if a.limit > 0:
        img_files = img_files[:a.limit]
    pred_dir.mkdir(parents=True, exist_ok=True)
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[data] {len(img_files)} 张测试图 <- {img_dir}")
    print(f"[data] GT <- {lab_dir}")
    print(f"[out ] pred_inst -> {pred_dir}")
    print(f"[cfg ] magnification={mag} tile={a.tile_size} overlap={a.overlap} "
          f"batch={a.batch_size} device={device}")

    model = build_model(a.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    agg = {k: [] for k in ["dice", "iou", "precision", "recall", "aji", "pq", "dq", "sq"]}
    times, n_pred_all, n_gt_all = [], 0, 0

    for ip in img_files:
        stem = ip.stem
        gp = lab_dir / f"{stem}.npy"
        assert gp.exists(), f"缺少 GT: {gp}"
        img = np.array(Image.open(ip).convert("RGB"))
        gt = relabel(np.load(gp, allow_pickle=True).item()["inst_map"].astype(np.int32))
        assert gt.shape == img.shape[:2], f"{stem}: 图像 {img.shape[:2]} 与 GT {gt.shape} 尺寸不一致"

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        maps = predict_maps(model, img, device, a.tile_size, a.overlap, a.batch_size)
        pred = instance_map_from_maps(model, maps, mag)          # 计时含整图后处理
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

        np.save(pred_dir / f"{stem}.npy", pred)                    # 无损，供统一评估脚本复核
        Image.fromarray(pred.astype(np.uint16)).save(pred_dir / f"{stem}.png")

        m = compute_metrics(pred, gt)
        for k in agg:
            agg[k].append(m[k])
        n_pred, n_gt = int(pred.max()), int(gt.max())
        n_pred_all += n_pred; n_gt_all += n_gt
        print(f"{stem}: AJI={m['aji']:.3f} PQ={m['pq']:.3f} DQ={m['dq']:.3f} SQ={m['sq']:.3f} "
              f"Dice={m['dice']:.3f} | pred={n_pred} gt={n_gt} | {times[-1]:.0f} ms")

    S = {k: float(np.mean(v)) for k, v in agg.items()}
    Std = {k: float(np.std(v)) for k, v in agg.items()}
    mt = float(np.mean(times)); fps = 1000.0 / mt if mt > 0 else 0.0
    tag = a.model_tag or f"cellvit256_ft_x{mag}"

    print(f"\n===== CellViT-256 (fine-tuned)  {a.dataset.upper()}  n={len(img_files)} =====")
    for k in ["aji", "pq", "dq", "sq", "dice", "iou", "precision", "recall"]:
        print(f"  {k.upper():9s}: {S[k]:.4f} ± {Std[k]:.4f}")
    print(f"  Params(M): {n_params:.2f}   time(ms/img): {mt:.1f}   FPS: {fps:.3f}")
    ratio = n_pred_all / max(n_gt_all, 1)
    print(f"  预测核总数/GT核总数 = {n_pred_all}/{n_gt_all} = {ratio:.2f}"
          + ("   <-- 明显偏离 1，请检查后处理/倍率" if not 0.6 <= ratio <= 1.5 else ""))

    if a.limit > 0:
        print("[note] --limit 模式，不写入 CSV")
        return
    append_summary(os.path.join(a.output_dir, "summary_wholemap.csv"), {
        "model": tag, "dataset": a.dataset, "params_M": round(n_params, 2),
        "dice": round(S["dice"], 4), "iou": round(S["iou"], 4),
        "precision": round(S["precision"], 4), "recall": round(S["recall"], 4),
        "aji": round(S["aji"], 4), "pq": round(S["pq"], 4),
        "dq_f1_05": round(S["dq"], 4), "sq": round(S["sq"], 4),
        "dice_std": round(Std["dice"], 4), "iou_std": round(Std["iou"], 4),
        "precision_std": round(Std["precision"], 4), "recall_std": round(Std["recall"], 4),
        "aji_std": round(Std["aji"], 4), "pq_std": round(Std["pq"], 4),
        "dq_f1_05_std": round(Std["dq"], 4), "sq_std": round(Std["sq"], 4),
        "time_ms": round(mt, 2), "fps": round(fps, 3)})


if __name__ == "__main__":
    main()