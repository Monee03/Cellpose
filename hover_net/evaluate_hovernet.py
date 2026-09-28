# evaluate_hovernet.py
"""
针对 Hover-Net 的专属全同轨评估脚本（集成全指标、IAADR自适应切片、GPU性能实测与 Sanity Check 版）
读取 Hover-Net 预测的 .mat 结果，自动适配设备，计算并输出完整的 12 项学术对比指标
python evaluate_hovernet.py \
  --pred_dir "./output_mat_lizard/mat/" \
  --true_dir "../edgetam/dataset/Lizard/masks/test/" \
  --img_dir "../edgetam/dataset/Lizard/images/test/" \
  --ann_path "../edgetam/dataset/Lizard/annotations/test_decontaminated.json" \
  --ckpt "./logs/01/net_epoch=50.tar" \
  --dataset lizard \
  --nr_types 2 \
  --adaptive_sahi


python evaluate_hovernet.py \
  --pred_dir "./output_mat_consep/mat/" \
  --true_dir "../edgetam/dataset/Consep/masks/test/" \
  --img_dir "../edgetam/dataset/Consep/images/test/" \
  --ann_path "../edgetam/dataset/Consep/annotations/test.json" \
  --ckpt "./logs/00/net_epoch=50.tar" \
  --dataset consep \
  --nr_types 2 

python run_infer.py \
--gpu 0 \
--nr_types 0 \
--type_info_path type_info.json \
--model_mode original \
--model_path ./consep_sep.tar \
--batch_size 8 \
--nr_inference_workers 4 \
--nr_post_proc_workers 0 \
tile \
--input_dir /home/featurize/work/edgetam/dataset/Consep/images/test \
--output_dir ./consep_out \
--save_raw_map

"""

import os
import sys
import csv
import json
import time
import argparse
import numpy as np
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm
import torch
import torch.nn.functional as F

# 尝试引入 Hover-Net 网络结构
try:
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from models.hovernet.net_desc import HoVerNet
except ImportError:
    try:
        from models.net_desc import HoVerNet
    except ImportError:
        HoVerNet = None

# ──────────────────────────────────────────────
# 尺度归一化与 SAHI 决策辅助函数
# ──────────────────────────────────────────────
D_TRIGGER    = 15.0        
D_TARGET     = 26.0        
D_MIN_AFTER  = 20.0        
TILE_CHOICES = [256, 384, 512, 768]
OVERLAP_MIN  = 0.25        
OVERLAP_MAX  = 0.40
IMG_SIZE     = 1024

def adaptive_sahi_decision(boxes, H, W):
    max_side = max(H, W)
    d = np.array([np.sqrt(max(1.0, x2 - x1) * max(1.0, y2 - y1)) for x1, y1, x2, y2 in boxes])
    d_med = float(np.median(d))
    geo_side = float(np.sqrt(H * W))
    d_canvas = d_med * IMG_SIZE / geo_side

    if d_canvas >= D_TRIGGER:
        return None, OVERLAP_MIN, "旁路"

    feasible = [t for t in TILE_CHOICES if (d_med * IMG_SIZE / t) >= D_MIN_AFTER and t <= 0.9 * max_side]
    if not feasible:
        return None, OVERLAP_MIN, "旁路"
    tile = max(feasible)

    overlap = float(np.clip(4.0 * d_med / tile, OVERLAP_MIN, OVERLAP_MAX))
    return tile, overlap, "激活"


def get_slices(h, w, size, overlap):
    stride = int(size * (1 - overlap))
    slices = []
    y = 0
    while y < h:
        y2 = min(y + size, h)
        y1 = max(0, y2 - size)
        x = 0
        while x < w:
            x2 = min(x + size, w)
            x1 = max(0, x2 - size)
            slices.append((y1, y2, x1, x2))
            if x2 == w: break
            x += stride
        if y2 == h: break
        y += stride
    return slices


def sync_device(device):
    """设备感知型同步"""
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def find_image_path(img_dir, stem):
    """自动支持多种常见病理格式原图后缀搜索"""
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".png"]
    for ext in exts:
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

# ──────────────────────────────────────────────
# 实例分割评估算法 (AJI / PQ / DQ / SQ)
# ──────────────────────────────────────────────

def compute_aji(pred_inst: np.ndarray, gt_inst: np.ndarray) -> float:
    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    pred_areas = {pid: (pred_inst == pid).sum() for pid in pred_ids}
    gt_areas = {gid: (gt_inst == gid).sum() for gid in gt_ids}

    overall_intersection = 0
    overall_union = 0
    matched_pred_ids = set()

    for gid in gt_ids:
        gt_mask = (gt_inst == gid)
        gt_area = gt_areas[gid]

        overlapping_pred_pixels = pred_inst[gt_mask]
        overlapping_pred_ids = np.unique(overlapping_pred_pixels)
        overlapping_pred_ids = overlapping_pred_ids[overlapping_pred_ids > 0]

        best_iou = -1.0
        best_pid = -1
        best_inter = 0

        for pid in overlapping_pred_ids:
            pred_mask = (pred_inst == pid)
            inter = (gt_mask & pred_mask).sum()
            union = gt_area + pred_areas[pid] - inter
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_pid = pid
                best_inter = inter

        if best_iou > 0:
            overall_intersection += best_inter
            overall_union += (gt_area + pred_areas[best_pid] - best_inter)
            matched_pred_ids.add(best_pid)
        else:
            overall_union += gt_area

    unmatched_pred_ids = set(pred_ids) - matched_pred_ids
    for pid in unmatched_pred_ids:
        overall_union += pred_areas[pid]

    return overall_intersection / overall_union if overall_union > 0 else 0.0


def compute_pq(pred_inst: np.ndarray, gt_inst: np.ndarray) -> tuple:
    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0, 1.0, 1.0
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0.0, 0.0, 0.0

    pred_areas = {pid: (pred_inst == pid).sum() for pid in pred_ids}
    gt_areas = {gid: (gt_inst == gid).sum() for gid in gt_ids}

    tp = 0
    matched_pred = set()
    matched_gt = set()
    sum_iou = 0.0

    for gid in gt_ids:
        gt_mask = (gt_inst == gid)
        gt_area = gt_areas[gid]

        overlapping_pred_pixels = pred_inst[gt_mask]
        overlapping_pred_ids = np.unique(overlapping_pred_pixels)
        overlapping_pred_ids = overlapping_pred_ids[overlapping_pred_ids > 0]

        for pid in overlapping_pred_ids:
            if pid in matched_pred:
                continue
            pred_mask = (pred_inst == pid)
            inter = (gt_mask & pred_mask).sum()
            union = gt_area + pred_areas[pid] - inter
            iou = inter / union if union > 0 else 0
            if iou > 0.5:
                tp += 1
                sum_iou += iou
                matched_pred.add(pid)
                matched_gt.add(gid)
                break

    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - len(matched_gt)

    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0
    sq = sum_iou / tp if tp > 0 else 0.0
    pq = dq * sq
    return pq, dq, sq


def compute_wholemap_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray, 
                             pred_inst: np.ndarray, gt_inst: np.ndarray) -> dict:
    pred_bin = pred_bin.astype(bool)
    gt_bin   = gt_bin.astype(bool)
    inter = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    ps, gs = pred_bin.sum(), gt_bin.sum()

    dice = 2 * inter / (ps + gs) if (ps + gs) > 0 else 1.0
    iou  = inter / union          if union > 0       else 1.0
    prec = inter / ps             if ps > 0          else 1.0
    rec  = inter / gs             if gs > 0          else 1.0

    aji = compute_aji(pred_inst, gt_inst)
    pq, dq, sq = compute_pq(pred_inst, gt_inst)

    return {
        "dice": float(dice), "iou": float(iou),
        "precision": float(prec), "recall": float(rec),
        "aji": float(aji), "pq": float(pq),
        "dq": float(dq), "sq": float(sq)
    }


def load_annotations(ann_path: str) -> dict:
    with open(ann_path) as f:
        data = json.load(f)
    images = {img["id"]: img for img in data["images"]}
    img_anns = {}
    for ann in data["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)
    
    file_to_boxes = {}
    for img_id, img_info in images.items():
        fname = img_info["file_name"]
        boxes = []
        for ann in img_anns.get(img_id, []):
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
        file_to_boxes[fname] = boxes
    return file_to_boxes


def append_summary(summary_path, row_dict):
    exists = os.path.exists(summary_path)
    if exists:
        with open(summary_path, "r", newline="") as f:
            old_header = f.readline().strip().split(",")
        if old_header != list(row_dict.keys()):
            print(f"⚠ 汇总表表头与当前字段不一致({len(old_header)}列 vs {len(row_dict)}列)，"
                  f"建议归档旧表后重新生成: {summary_path}")
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row_dict)
    print(f"✓ 测试数据已追加保存至汇总表: {summary_path}")


# ──────────────────────────────────────────────
# 深度递归设备强制迁移器
# ──────────────────────────────────────────────
def force_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, torch.nn.Module):
        obj.to(device)
        for key, value in obj.__dict__.items():
            if isinstance(value, (torch.nn.Module, torch.Tensor)):
                setattr(obj, key, force_to_device(value, device))
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, (torch.nn.Module, torch.Tensor)):
                        new_list.append(force_to_device(item, device))
                    else:
                        new_list.append(item)
                setattr(obj, key, new_list)
            elif isinstance(value, dict):
                new_dict = {}
                for k, v in value.items():
                    if isinstance(v, (torch.nn.Module, torch.Tensor)):
                        new_dict[k] = force_to_device(v, device)
                    else:
                        new_dict[k] = v
                setattr(obj, key, new_dict)
    return obj


# ──────────────────────────────────────────────
# 主运行流
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, default="./output_mat/mat/",
                        help="Hover-Net 预测生成的 mat 文件目录")
    parser.add_argument("--true_dir", type=str, default="../edgetam/dataset/Lizard/masks/test/",
                        help="真值标签 PNG 图像目录")
    parser.add_argument("--img_dir", type=str, default="../edgetam/dataset/Lizard/images/test/",
                        help="原始测试集图像目录（耗时实测需要）")
    parser.add_argument("--ann_path", type=str, default="../edgetam/dataset/Lizard/annotations/test_decontaminated.json",
                        help="测试集 COCO 标注文件（IAADR自适应计算密度需要）")
    parser.add_argument("--ckpt", type=str, default="./hovernet_original_consep_notype_pytorch.tar",
                        help="Hover-Net 权重文件路径")
    parser.add_argument("--dataset", type=str, default="lizard", choices=["consep", "lizard"])
    parser.add_argument("--adaptive_sahi", action="store_true",
                        help="激活 IAADR 自适应切片耗时实测")
    parser.add_argument("--nr_types", type=int, default=2,
                        help="Hover-Net 类型数")
    parser.add_argument("--output_dir", type=str, default="./results_wholemap",
                        help="汇总 CSV 表格输出目录")
    args = parser.parse_args()

    # 自动建立输出汇总文件夹，杜绝文件 IO 报错
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载标注信息以获取 Bounding Boxes 供 IAADR 决策
    boxes_meta = load_annotations(args.ann_path) if os.path.exists(args.ann_path) else {}

    # 1. GPU 耗时与参数量统计 (HoVerNet 模型加载)
    params_M = 31.28
    inference_times = []
    model = None

    if HoVerNet is not None and os.path.exists(args.ckpt):
        try:
            print(f"正在加载 Hover-Net 结构与权重并测试迁移设备...")
            import inspect
            sig = inspect.signature(HoVerNet.__init__)
            params = sig.parameters
            
            kwargs = {}
            if "mode" in params:
                kwargs["mode"] = "original"
                
            type_arg_name = None
            for arg in ["nr_types", "num_types", "num_classes", "n_classes"]:
                if arg in params:
                    type_arg_name = arg
                    break
                    
            if type_arg_name is not None and args.nr_types > 0:
                kwargs[type_arg_name] = args.nr_types
                
            model = HoVerNet(**kwargs)
            checkpoint = torch.load(args.ckpt, map_location="cpu")
            state_dict = checkpoint["desc"] if isinstance(checkpoint, dict) and "desc" in checkpoint else checkpoint
            
            # 去除 DataParallel 保存的 module. 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                new_state_dict[k] = v
                
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            
            # 深度强制设备迁移
            model = force_to_device(model, device)
            model.eval()

            params_M = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  ✓ 网络加载成功，参数量实测为: {params_M:.2f} M")
            print(f"    Missing Keys: {len(missing)} | Unexpected Keys: {len(unexpected)}")
        except Exception as e:
            model = None
            print(f"  ⚠️ 模型架构兼容或加载失败: {e}。已激活安全降级模式（仅进行指标计算）。")
    else:
        print("  ⚠️ 未检测到有效的本地 HoVerNet 模型结构或权重，已采用官方原始参数量兜底。")

    # 2. 提前进行 Sanity Check（前向传播健康审计）
    if model is not None:
        try:
            actual_model_device = next(model.parameters()).device
            # ★ 修复：使用标准的黄金尺寸 270x270 进行健康审计，完美规避下采样 rounding 不对称问题
            dummy = torch.randn(1, 3, 270, 270).to(actual_model_device)
            with torch.inference_mode():
                sync_device(actual_model_device)
                t0 = time.perf_counter()
                _ = model(dummy)
                sync_device(actual_model_device)
                elapsed = (time.perf_counter() - t0) * 1000.0
            print(f"  ✓ [Sanity Check] HoVerNet 单次 270x270 预热前向传播成功: {elapsed:.2f} ms")
        except Exception as e:
            print(f"  ⚠️ [Sanity Check 失败] HoVerNet 运行前向测试失败: {repr(e)}。已关闭性能实测分支。")
            model = None

    # 过滤预测的文件
    pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith(".mat")])
    if not pred_files:
        raise FileNotFoundError(f"在 {args.pred_dir} 中未找到预测的 .mat 结果，请先执行推理步骤。")

    all_metrics = {
        "dice": [], "iou": [], "precision": [], "recall": [],
        "aji": [], "pq": [], "dq": [], "sq": []
    }

    failed_paths = []

    print(f"自适应路由决策引擎 (IAADR): {'已激活' if args.adaptive_sahi else '未激活'}")
    print(f"正在全同轨评估 Hover-Net 在 {args.dataset.upper()} 数据集上的表现...")
    
    for fname in tqdm(pred_files):
        stem = os.path.splitext(fname)[0]
        
        # 1. 读取预测结果 inst_map
        pred_path = os.path.join(args.pred_dir, fname)
        pred_data = loadmat(pred_path)
        pred_inst = pred_data["inst_map"].astype(np.int32)
        
        # 2. 读取对应真值标签 (PNG 实例图)
        gt_path = os.path.join(args.true_dir, f"{stem}.png")
        if not os.path.exists(gt_path):
            failed_paths.append(gt_path)
            continue
            
        gt_raw = np.array(Image.open(gt_path))
        if gt_raw.ndim == 3:
            gt_raw = gt_raw[:, :, 0]
        gt_inst = gt_raw.astype(np.int32)
        
        # 3. 稳健硬件级耗时评测（支持 IAADR 自适应切片模拟）
        if model is not None:
            img_path = find_image_path(args.img_dir, stem)
            if img_path is None:
                tqdm.write(f"  ⚠️ [跳过耗时实测] 找不到对应的原图: {stem}")
            else:
                img_np = np.array(Image.open(img_path).convert("RGB"))
                H, W = img_np.shape[:2]
                
                # 获取该图像的细胞框列表
                boxes = boxes_meta.get(f"{stem}.png", [])
                
                # 计算自适应切片决策
                sahi_size = None
                sahi_overlap = OVERLAP_MIN
                if args.adaptive_sahi and len(boxes) > 0:
                    sahi_size, sahi_overlap, _ = adaptive_sahi_decision(boxes, H, W)
                
                try:
                    actual_model_device = next(model.parameters()).device
                    
                    with torch.inference_mode():
                        if sahi_size is not None:
                            # ─── 自适应切片时：对图像切块，串行跑前向并统计总时耗 ───
                            slices = get_slices(H, W, sahi_size, sahi_overlap)
                            
                            sync_device(actual_model_device)
                            t0 = time.perf_counter()
                            
                            for sl in slices:
                                sy1, sy2, sx1, sx2 = sl
                                slice_np = img_np[sy1:sy2, sx1:sx2]
                                slice_tensor = torch.from_numpy(slice_np).permute(2, 0, 1).float().unsqueeze(0).to(actual_model_device) / 255.0
                                slice_tensor = F.interpolate(slice_tensor, size=(sahi_size, sahi_size), mode="bilinear", align_corners=False)
                                _ = model(slice_tensor)
                                
                            sync_device(actual_model_device)
                            elapsed = (time.perf_counter() - t0) * 1000.0
                            inference_times.append(elapsed)
                        else:
                            # ─── 标准推理时：直接跑全图前向 ───
                            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(actual_model_device) / 255.0
                            img_tensor = F.interpolate(img_tensor, size=(1024, 1024), mode="bilinear", align_corners=False)
                            
                            sync_device(actual_model_device)
                            t0 = time.perf_counter()
                            _ = model(img_tensor)
                            sync_device(actual_model_device)
                            elapsed = (time.perf_counter() - t0) * 1000.0
                            inference_times.append(elapsed)
                except Exception as e:
                    tqdm.write(f"  ⚠️ [耗时测量失败] {stem}: {repr(e)}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # 4. 提取二值图并计算指标
        pred_bin = (pred_inst > 0).astype(np.uint8)
        gt_binary = (gt_inst > 0).astype(np.uint8)
        
        m = compute_wholemap_metrics(pred_bin, gt_binary, pred_inst, gt_inst)
        for k in all_metrics:
            all_metrics[k].append(m[k])

    # ★ 核心诊断面板：如果评估列表为空，打印直观排错信息
    if len(all_metrics["dice"]) == 0:
        print("\n❌ [错误] 评估失败！脚本未能成功配对并计算任何一张图片的指标。")
        print(f"  - 预测文件目录 (.mat): {args.pred_dir} (共找到 {len(pred_files)} 个文件)")
        print(f"  - 真值掩码目录 (.png): {args.true_dir}")
        print("  - 脚本尝试寻找却【未能在真值目录下找到】的真值文件路径示例：")
        for p in failed_paths[:5]:
            print(f"    * 缺失路径: {p}")
        print("\n💡 请检查：")
        print("  1. 请核对 `--true_dir` 是否指向了包含有对应 PNG 真值文件的正确目录。")
        print("  2. 请核对预测的 mat 文件名（如 test_1.mat 去掉 .mat 后）是否与真值文件名（如 test_1.png）能完全一致配对。")
        return

    # 计算平均值与逐图标准差
    summary_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    std_metrics     = {k: float(np.std(v))  for k, v in all_metrics.items()}
    
    # 耗时与 FPS
    mean_time = np.mean(inference_times) if inference_times else 0.0
    fps = 1000.0 / mean_time if mean_time > 0.0 else 0.0

    print(f"\n==================== Hover-Net {args.dataset.upper()} 测试结果 ====================")
    print(f"  参数量 (Params M): {params_M:.2f} M")
    print(f"  AJI:              {summary_metrics['aji']:.4f} ± {std_metrics['aji']:.4f}")
    print(f"  PQ:               {summary_metrics['pq']:.4f} ± {std_metrics['pq']:.4f}")
    print(f"  DQ (F1 05):       {summary_metrics['dq']:.4f} ± {std_metrics['dq']:.4f}")
    print(f"  SQ:               {summary_metrics['sq']:.4f} ± {std_metrics['sq']:.4f}")
    print(f"  Dice:             {summary_metrics['dice']:.4f} ± {std_metrics['dice']:.4f}")
    print(f"  IoU:              {summary_metrics['iou']:.4f} ± {std_metrics['iou']:.4f}")
    print(f"  Precision:        {summary_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"  Recall:           {summary_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"  推理耗时 (ms):     {f'{mean_time:.2f} ms' if mean_time > 0.0 else 'N/A'}")
    print(f"  FPS:              {f'{fps:.2f} 帧/秒' if mean_time > 0.0 else 'N/A'}")
    print(f"==========================================================================")

    # 5. 追加写入汇总 summary_wholemap.csv
    summary_csv_path = os.path.join(args.output_dir, "summary_wholemap.csv")
    append_summary(summary_csv_path, {
        "model":     "hovernet_original",
        "dataset":   args.dataset,
        "params_M":  round(params_M, 2),
        "dice":      round(summary_metrics["dice"], 4),
        "iou":       round(summary_metrics["iou"], 4),
        "precision": round(summary_metrics["precision"], 4),
        "recall":    round(summary_metrics["recall"], 4),
        "aji":       round(summary_metrics["aji"], 4),
        "pq":        round(summary_metrics["pq"], 4),
        "dq_f1_05":  round(summary_metrics["dq"], 4),
        "sq":        round(summary_metrics["sq"], 4),
        "dice_std":      round(std_metrics["dice"], 4),
        "iou_std":       round(std_metrics["iou"], 4),
        "precision_std": round(std_metrics["precision"], 4),
        "recall_std":    round(std_metrics["recall"], 4),
        "aji_std":       round(std_metrics["aji"], 4),
        "pq_std":        round(std_metrics["pq"], 4),
        "dq_f1_05_std":  round(std_metrics["dq"], 4),
        "sq_std":        round(std_metrics["sq"], 4),
        "time_ms":   round(mean_time, 2) if mean_time > 0.0 else "N/A",
        "fps":       round(fps, 2) if mean_time > 0.0 else "N/A",
    })

if __name__ == "__main__":
    main()