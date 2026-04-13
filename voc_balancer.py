#!/usr/bin/env python3
"""
voc_balancer.py — PASCAL VOC 訓練資料平衡工具

策略:
  1. 統計每個 label 在整個資料集中出現的次數 (instance count) 及涵蓋的影像數
  2. 計算目標 instance 數（預設：中位數 或 使用者指定）
  3. 過多的 label → 優先從「只含該單一 label」的影像中刪除，
     但至少保留此類影像的 5~10%（預設 7%，可調）
  4. 不足的 label → 對含該 label 的影像進行複製擴增（augmentation: flip / rotate / brightness）
  5. 輸出平衡後的資料集到指定目錄，並生成統計報告
"""

import argparse
import copy
import math
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# ── 可選：Pillow 用於影像 augmentation ──────────────────────────────────────
try:
    from PIL import Image, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────────────────────

def parse_voc_xml(xml_path: Path) -> list[str]:
    """回傳 XML 中所有 <name> 標籤的 label 清單（可重複）。"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return [obj.find("name").text.strip()
                for obj in root.findall("object")
                if obj.find("name") is not None]
    except ET.ParseError as e:
        print(f"  [WARN] 解析失敗 {xml_path.name}: {e}")
        return []


def scan_dataset(ann_dir: Path, img_dir: Optional[Path]):
    """
    掃描資料集，回傳：
      samples       : list of dict {xml, img, labels, label_set}
      label_counts  : Counter {label: total_instance_count}
      label_images  : {label: [sample_index, ...]}
    """
    samples = []
    label_counts: Counter = Counter()
    label_images: dict[str, list[int]] = defaultdict(list)

    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"在 {ann_dir} 找不到任何 .xml 檔案")

    for idx, xml_path in enumerate(xml_files):
        labels = parse_voc_xml(xml_path)
        if not labels:
            continue  # 跳過空 annotation

        # 尋找對應影像
        img_path = None
        if img_dir:
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                candidate = img_dir / (xml_path.stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break

        sample = {
            "xml": xml_path,
            "img": img_path,
            "labels": labels,
            "label_set": set(labels),
        }
        samples.append(sample)

        for lbl in labels:
            label_counts[lbl] += 1
        for lbl in set(labels):
            label_images[lbl].append(idx)

    return samples, label_counts, label_images


def compute_target(label_counts: Counter, strategy: str, target_val: Optional[int]) -> int:
    """計算目標 instance 數。"""
    counts = list(label_counts.values())
    if target_val is not None:
        return target_val
    if strategy == "median":
        counts_sorted = sorted(counts)
        n = len(counts_sorted)
        return int((counts_sorted[(n - 1) // 2] + counts_sorted[n // 2]) / 2)
    if strategy == "mean":
        return int(sum(counts) / len(counts))
    if strategy == "min":
        return min(counts)
    if strategy == "max":
        return max(counts)
    raise ValueError(f"未知 strategy: {strategy}")


# ─────────────────────────────────────────────────────────────────────────────
# 影像 Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_image(src_img: Path, dst_img: Path, method: str):
    """對影像進行簡單 augmentation 並儲存到 dst_img。"""
    if not PIL_AVAILABLE:
        shutil.copy2(src_img, dst_img)
        return

    img = Image.open(src_img)

    if method == "flip_h":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif method == "flip_v":
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif method == "rotate90":
        img = img.rotate(90, expand=True)
    elif method == "rotate180":
        img = img.rotate(180)
    elif method == "brightness":
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    elif method == "contrast":
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    else:
        pass  # copy only

    img.save(dst_img)


def augment_xml(src_xml: Path, dst_xml: Path, new_stem: str, method: str,
                img_w: Optional[int] = None, img_h: Optional[int] = None):
    """
    複製並修改 XML，調整 filename，
    若是水平/垂直翻轉，同步更新 bndbox 座標。
    """
    tree = ET.parse(src_xml)
    root = tree.getroot()

    # 更新 filename
    fn_node = root.find("filename")
    if fn_node is not None:
        old_fn = Path(fn_node.text)
        fn_node.text = new_stem + old_fn.suffix

    # 取得影像尺寸（用於翻轉座標計算）
    size_node = root.find("size")
    if size_node is not None:
        w = int(size_node.find("width").text) if img_w is None else img_w
        h = int(size_node.find("height").text) if img_h is None else img_h
    else:
        w, h = img_w or 0, img_h or 0

    if method in ("flip_h", "flip_v") and w and h:
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin = float(bb.find("xmin").text)
            ymin = float(bb.find("ymin").text)
            xmax = float(bb.find("xmax").text)
            ymax = float(bb.find("ymax").text)

            if method == "flip_h":
                bb.find("xmin").text = str(w - xmax)
                bb.find("xmax").text = str(w - xmin)
            else:  # flip_v
                bb.find("ymin").text = str(h - ymax)
                bb.find("ymax").text = str(h - ymin)

    tree.write(dst_xml, encoding="utf-8", xml_declaration=True)


# ─────────────────────────────────────────────────────────────────────────────
# 核心平衡邏輯
# ─────────────────────────────────────────────────────────────────────────────

def balance_dataset(
    samples: list,
    label_counts: Counter,
    label_images: dict,
    target: int,
    single_label_keep_ratio: float,
    aug_methods: list[str],
    rng: random.Random,
) -> tuple[list, dict]:
    """
    回傳：
      kept_indices : set of int（要保留的原始 sample index）
      aug_list     : list of (src_idx, method, new_stem_suffix)（要擴增的 sample）
    """
    kept = set(range(len(samples)))     # 預設全保留
    aug_list = []                       # (src_idx, method)

    # ── Step 1: 對過多的 label 進行 under-sampling ──────────────────────────
    current_counts = Counter(label_counts)  # 可變副本（instance 數）

    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        if current_counts[label] <= target:
            continue

        excess = current_counts[label] - target

        # 找出「只含此 label」的 sample（且目前仍在 kept 中）
        single_only = [
            idx for idx in label_images[label]
            if idx in kept and samples[idx]["label_set"] == {label}
        ]
        # 至少保留 single_label_keep_ratio 的數量
        min_keep = max(1, math.ceil(len(single_only) * single_label_keep_ratio))
        max_removable = len(single_only) - min_keep

        to_remove_from_single = min(excess, max(0, max_removable))

        if to_remove_from_single > 0:
            candidates = rng.sample(single_only, to_remove_from_single)
            for idx in candidates:
                # 從 kept 移除，並更新 current_counts
                kept.discard(idx)
                for lbl in samples[idx]["labels"]:
                    current_counts[lbl] -= 1
            excess -= to_remove_from_single

        # 若仍有 excess，從多 label 影像中再移除（但小心不要殃及其他已達標的 label）
        if excess > 0:
            multi_candidates = [
                idx for idx in label_images[label]
                if idx in kept and len(samples[idx]["label_set"]) > 1
            ]
            rng.shuffle(multi_candidates)
            for idx in multi_candidates:
                if excess <= 0:
                    break
                # 確保移除此影像不會讓其他 label 降到 target 以下
                safe = True
                for lbl in samples[idx]["label_set"]:
                    if lbl != label and current_counts[lbl] - 1 < target * 0.5:
                        safe = False
                        break
                if safe:
                    kept.discard(idx)
                    for lbl in samples[idx]["labels"]:
                        current_counts[lbl] -= 1
                    excess -= 1

    # ── Step 2: 對不足的 label 進行 over-sampling（augmentation）─────────────
    # 重新計算保留後的 counts
    live_counts: Counter = Counter()
    for idx in kept:
        for lbl in samples[idx]["labels"]:
            live_counts[lbl] += 1

    aug_counter = 0
    for label, count in sorted(live_counts.items(), key=lambda x: x[1]):
        if count >= target:
            continue

        deficit = target - count
        candidate_idxs = [i for i in label_images[label] if i in kept]
        if not candidate_idxs:
            continue

        # 若沒有 aug_methods（--no_aug 或空清單），跳過 over-sampling
        if not aug_methods:
            continue

        methods_cycle = aug_methods * (deficit // len(aug_methods) + 1)
        rng.shuffle(methods_cycle)

        added = 0
        for i, method in enumerate(methods_cycle):
            if added >= deficit:
                break
            src_idx = candidate_idxs[i % len(candidate_idxs)]
            aug_list.append((src_idx, method, f"_aug{aug_counter:05d}"))
            aug_counter += 1
            added += 1

    return kept, aug_list


# ─────────────────────────────────────────────────────────────────────────────
# 輸出
# ─────────────────────────────────────────────────────────────────────────────

def write_output(
    samples: list,
    kept: set,
    aug_list: list,
    out_ann_dir: Path,
    out_img_dir: Optional[Path],
    dry_run: bool,
):
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    if out_img_dir:
        out_img_dir.mkdir(parents=True, exist_ok=True)

    copied_xml = 0
    copied_img = 0

    # ── 複製保留的原始檔案 ────────────────────────────────────────────────────
    for idx in sorted(kept):
        s = samples[idx]
        dst_xml = out_ann_dir / s["xml"].name
        if not dry_run:
            shutil.copy2(s["xml"], dst_xml)
        copied_xml += 1

        if s["img"] and out_img_dir:
            dst_img = out_img_dir / s["img"].name
            if not dry_run:
                shutil.copy2(s["img"], dst_img)
            copied_img += 1

    # ── 寫入擴增樣本 ──────────────────────────────────────────────────────────
    aug_xml = 0
    aug_img = 0

    for src_idx, method, suffix in aug_list:
        s = samples[src_idx]
        new_stem = s["xml"].stem + suffix
        dst_xml = out_ann_dir / (new_stem + ".xml")

        if not dry_run:
            augment_xml(s["xml"], dst_xml, new_stem, method)
        aug_xml += 1

        if s["img"] and out_img_dir:
            dst_img = out_img_dir / (new_stem + s["img"].suffix)
            if not dry_run:
                augment_image(s["img"], dst_img, method)
            aug_img += 1

    return copied_xml, copied_img, aug_xml, aug_img


# ─────────────────────────────────────────────────────────────────────────────
# 報告
# ─────────────────────────────────────────────────────────────────────────────

def print_report(
    label_counts_before: Counter,
    samples: list,
    kept: set,
    aug_list: list,
    target: int,
):
    # 計算平衡後的 counts
    after_counts: Counter = Counter()
    for idx in kept:
        for lbl in samples[idx]["labels"]:
            after_counts[lbl] += 1
    for src_idx, method, _ in aug_list:
        for lbl in samples[src_idx]["labels"]:
            after_counts[lbl] += 1

    all_labels = sorted(set(label_counts_before) | set(after_counts))
    col_w = max(len(l) for l in all_labels) + 2

    print("\n" + "═" * (col_w + 42))
    print(f"{'Label':<{col_w}} {'Before':>8}  {'After':>8}  {'Δ':>8}  {'Target':>8}")
    print("─" * (col_w + 42))
    for lbl in all_labels:
        b = label_counts_before.get(lbl, 0)
        a = after_counts.get(lbl, 0)
        delta = a - b
        mark = "↑" if delta > 0 else ("↓" if delta < 0 else "─")
        print(f"{lbl:<{col_w}} {b:>8}  {a:>8}  {mark}{abs(delta):>7}  {target:>8}")
    print("═" * (col_w + 42))

    total_before = sum(label_counts_before.values())
    total_after = sum(after_counts.values())
    print(f"\n原始樣本數   : {len(samples)}")
    print(f"保留樣本數   : {len(kept)}")
    print(f"移除樣本數   : {len(samples) - len(kept)}")
    print(f"擴增樣本數   : {len(aug_list)}")
    print(f"輸出總樣本數 : {len(kept) + len(aug_list)}")
    print(f"Instance 總數: {total_before} → {total_after}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voc_balancer",
        description="PASCAL VOC 訓練資料平衡工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 以中位數為目標，自動平衡
  python voc_balancer.py --ann_dir data/Annotations --img_dir data/JPEGImages --out_dir balanced/

  # 指定目標 instance 數為 500，保留 10% single-label 影像
  python voc_balancer.py --ann_dir data/Annotations --out_dir balanced/ --target 500 --keep_ratio 0.10

  # 僅印出報告，不實際複製檔案
  python voc_balancer.py --ann_dir data/Annotations --out_dir balanced/ --dry_run
        """,
    )

    # 路徑
    p.add_argument("--ann_dir",  required=True,  type=Path, help="PASCAL VOC Annotations 資料夾")
    p.add_argument("--img_dir",  default=None,   type=Path, help="影像資料夾（可選；若不提供則只處理 XML）")
    p.add_argument("--out_dir",  required=True,  type=Path, help="輸出根目錄")
    p.add_argument("--out_ann_subdir", default="Annotations", help="輸出 XML 子目錄名稱（預設 Annotations）")
    p.add_argument("--out_img_subdir", default="JPEGImages",  help="輸出影像子目錄名稱（預設 JPEGImages）")

    # 目標策略
    p.add_argument("--strategy", default="median",
                   choices=["median", "mean", "min", "max"],
                   help="目標 instance 數計算策略（預設 median）")
    p.add_argument("--target", type=int, default=None,
                   help="直接指定目標 instance 數（覆蓋 --strategy）")

    # 保留比率
    p.add_argument("--keep_ratio", type=float, default=0.07,
                   help="單一 label 影像至少保留比率，0.05~0.10（預設 0.07）")

    # Augmentation
    p.add_argument("--aug_methods", nargs="+",
                   default=["flip_h", "rotate90", "brightness", "contrast"],
                   choices=["flip_h", "flip_v", "rotate90", "rotate180", "brightness", "contrast", "none"],
                   help="擴增方法（預設: flip_h rotate90 brightness contrast）")

    # 其他
    p.add_argument("--seed",    type=int, default=42, help="隨機種子（預設 42）")
    p.add_argument("--dry_run", action="store_true",  help="僅印出報告，不寫入檔案")
    p.add_argument("--no_aug",  action="store_true",  help="關閉 over-sampling 擴增，只做 under-sampling")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── 掃描 ──────────────────────────────────────────────────────────────────
    print(f"\n[1/4] 掃描 Annotations: {args.ann_dir}")
    samples, label_counts, label_images = scan_dataset(args.ann_dir, args.img_dir)
    print(f"      找到 {len(samples)} 個有效樣本，{len(label_counts)} 種 label")

    if not samples:
        print("[ERROR] 沒有找到有效樣本，請確認路徑正確。")
        return 1

    # ── 計算目標 ─────────────────────────────────────────────────────────────
    target = compute_target(label_counts, args.strategy, args.target)
    print(f"\n[2/4] 目標 instance 數: {target}  （策略: {args.strategy if args.target is None else '指定值'}）")

    keep_ratio = max(0.05, min(0.10, args.keep_ratio))
    aug_methods = args.aug_methods if not args.no_aug else []

    # ── 平衡 ─────────────────────────────────────────────────────────────────
    print(f"\n[3/4] 執行平衡（keep_ratio={keep_ratio:.0%}）...")
    kept, aug_list = balance_dataset(
        samples, label_counts, label_images,
        target, keep_ratio, aug_methods, rng,
    )

    # ── 報告 ─────────────────────────────────────────────────────────────────
    print_report(label_counts, samples, kept, aug_list, target)

    # ── 輸出 ─────────────────────────────────────────────────────────────────
    out_ann_dir = args.out_dir / args.out_ann_subdir
    out_img_dir = (args.out_dir / args.out_img_subdir) if args.img_dir else None

    print(f"\n[4/4] 寫入輸出目錄: {args.out_dir}")
    if args.dry_run:
        print("      ── DRY RUN，不寫入任何檔案 ──")
    else:
        if not PIL_AVAILABLE:
            print("      [WARN] Pillow 未安裝，影像 augmentation 將以複製取代（pip install Pillow）")

    cx, ci, ax, ai = write_output(
        samples, kept, aug_list,
        out_ann_dir, out_img_dir,
        dry_run=args.dry_run,
    )

    print(f"\n      XML  : 複製 {cx} + 擴增 {ax} = {cx+ax} 個")
    if out_img_dir:
        print(f"      Image: 複製 {ci} + 擴增 {ai} = {ci+ai} 個")
    print("\n✅ 完成！\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
