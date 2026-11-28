# -*- coding: utf-8 -*-
"""非本模型样本拒识 —— 校准与验证脚本
=====================================

用仓库自带的测试集为每个分类模型拟合一份「训练分布」档案，写进
``models/ood_stats/<task>.npz``，服务启动后 ``ood_detection`` 会自动读取。

用法
----
    # 校准全部任务，并在留出集上验证误拒率
    python py/build_ood_stats.py

    # 只校准某一个
    python py/build_ood_stats.py --task mango_leaf

    # 每类最多用 60 张（默认 80），数据少或想跑快一点
    python py/build_ood_stats.py --limit 60

    # 只验证，不重新校准（校准文件已存在时）
    python py/build_ood_stats.py --verify-only

    # 校准 + 交叉验证：把别的模型的数据集喂给本模型，看能不能识别出“不是我的样本”
    python py/build_ood_stats.py --cross

输出含义
--------
    FPR（误拒率）   同分布样本被拒绝的比例，越低越好，正常应 < 1%
    SUSP（疑似率）  同分布样本被判为「疑似」的比例，过高说明阈值太紧
    CROSS           跨任务样本被拒绝或标疑似的比例，越高说明越能拦住误传
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import console_encoding  # noqa: F401  - 统一控制台编码，避免中文乱码

import numpy as np
import torch
from PIL import Image

import ood_detection as ood
from model_registry import REGISTRY, TASKS, _restore_console_encoding

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 每个任务的校准数据：根目录 + 用于拟合的切分 + 用于验证的切分
CLASSIFICATION_DATASETS: Dict[str, Dict[str, str]] = {
    "mango_leaf": {
        "root": "Mango Leaf Dataset",
        "fit": "test",
        "holdout": "val",
    },
    "mango_variety": {
        "root": "Mango Dataset",
        "fit": "test",
        "holdout": "val",
    },
    "grape_disease": {
        "root": "Grape Disease Dataset",
        "fit": "test",
        "holdout": "val",
    },
    "leaf_disease": {
        "root": "Leaf Disease Segment/classification",
        "fit": "test",
        "holdout": "val",
    },
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# 数据收集
# ---------------------------------------------------------------------------

def collect_images(root: Path, limit: Optional[int], seed: int = 42) -> Tuple[List[Path], List[str]]:
    """遍历 ImageFolder 结构，返回 (路径列表, 类名列表)。"""
    if not root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {root}")

    rng = random.Random(seed)
    paths: List[Path] = []
    labels: List[str] = []

    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for class_dir in class_dirs:
        files = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if limit and len(files) > limit:
            files = rng.sample(files, limit)
        for file in files:
            paths.append(file)
            labels.append(class_dir.name)
    return paths, labels


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------

def _model_forward(task: str, model: torch.nn.Module, tensors: torch.Tensor):
    """一次前向拿到特征、logits、概率。"""
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(tensors.to(device))
        features = model.forward_features(tensors.to(device))
    probs = torch.softmax(logits, dim=1)
    return (
        features.detach().float().cpu().numpy(),
        logits.detach().float().cpu().numpy(),
        probs.detach().float().cpu().numpy(),
    )


def extract_registry_features(task: str, paths: Sequence[Path], batch_size: int = 16):
    entry = REGISTRY.get(task)
    model = entry.model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")).eval()
    transform = TASKS[task]["transform"]
    if transform is None:
        raise ValueError(f"{task} 没有配置图像预处理")

    feature_batches, logit_batches, prob_batches = [], [], []
    batch: List[torch.Tensor] = []

    def flush():
        if not batch:
            return
        stack = torch.stack(batch)
        features, logits, probs = _model_forward(task, model, stack)
        feature_batches.append(features)
        logit_batches.append(logits)
        prob_batches.append(probs)
        batch.clear()

    started = time.time()
    for index, path in enumerate(paths, 1):
        try:
            with Image.open(path) as image:
                batch.append(transform(image.convert("RGB")))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {path.name}: {exc}")
            continue
        if len(batch) >= batch_size:
            flush()
        if index % 50 == 0:
            print(f"  ... {index}/{len(paths)} 张，{time.time() - started:.1f}s")
    flush()

    return (
        np.concatenate(feature_batches, axis=0),
        np.concatenate(logit_batches, axis=0),
        np.concatenate(prob_batches, axis=0),
    )


def extract_leaf_features(paths: Sequence[Path]):
    """叶片分割+识别管线：用分类头的 hidden 表示 + 物种头。"""
    import cv2

    from leaf_disease_models import get_pipeline

    pipeline = get_pipeline()
    model = pipeline.cls_model
    device = pipeline.device

    feature_rows, logit_rows, prob_rows = [], [], []
    started = time.time()
    for index, path in enumerate(paths, 1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"  [skip] {path.name}: 无法读取")
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = pipeline._cls_tensor(rgb)
        with torch.no_grad():
            hidden = model.forward_features(tensor)
            species_logits = model.species(hidden)
        feature_rows.append(hidden.detach().float().cpu().numpy()[0])
        logit_rows.append(species_logits.detach().float().cpu().numpy()[0])
        prob_rows.append(torch.softmax(species_logits, dim=1)[0].float().cpu().numpy())
        if index % 50 == 0:
            print(f"  ... {index}/{len(paths)} 张，{time.time() - started:.1f}s")
        del tensor, hidden, species_logits
    return (
        np.stack(feature_rows),
        np.stack(logit_rows),
        np.stack(prob_rows),
    )


def extract_features(task: str, paths: Sequence[Path], batch_size: int = 16):
    if task == "leaf_disease":
        return extract_leaf_features(paths)
    return extract_registry_features(task, paths, batch_size=batch_size)


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

def evaluate_profile(guard: ood.OODGuard, features, logits, probs,
                     labels: Optional[Sequence[str]] = None,
                     class_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """在给定样本上统计 拒绝 / 疑似 比例，顺带算一次 top-1 准确率。"""
    statuses: List[str] = []
    correct = 0
    labeled = 0

    index_of = {name: i for i, name in enumerate(class_names or [])}
    # 类名和模型输出头对不上时（例如叶片管线的文件夹名 vs 物种头）不算准确率
    comparable = labels is not None and len(index_of) == probs.shape[1]
    for i in range(features.shape[0]):
        result = guard.evaluate(features[i], logits[i], probs[i])
        statuses.append(result["status"])
        if comparable and labels[i] in index_of:
            labeled += 1
            if int(np.argmax(probs[i])) == index_of[labels[i]]:
                correct += 1

    total = max(1, len(statuses))
    rejected = statuses.count(ood.STATUS_REJECTED)
    suspicious = statuses.count(ood.STATUS_SUSPICIOUS)
    return {
        "total": len(statuses),
        "rejected": rejected,
        "suspicious": suspicious,
        "reject_rate": rejected / total,
        "suspicious_rate": suspicious / total,
        "accuracy": (correct / labeled) if labeled else None,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def calibrate(task: str, limit: Optional[int], batch_size: int, pca_dim: int) -> Optional[ood.OODProfile]:
    config = CLASSIFICATION_DATASETS[task]
    fit_root = PROJECT_ROOT / config["root"] / config["fit"]

    print(f"\n=== 校准 {task} ({TASKS[task]['name'] if task in TASKS else '叶片分割+识别'}) ===")
    print(f"  数据目录: {fit_root}")
    paths, labels = collect_images(fit_root, limit)
    if len(paths) < ood.MIN_SAMPLES:
        print(f"  [FAIL] 可用图片只有 {len(paths)} 张，少于 {ood.MIN_SAMPLES} 张，跳过")
        return None
    print(f"  样本数: {len(paths)}（{len(set(labels))} 个类）")

    features, logits, probs = extract_features(task, paths, batch_size=batch_size)
    print(f"  特征维度: {features.shape[1]}，PCA 降到 {min(pca_dim, features.shape[0] - 1, features.shape[1])}")

    # 留出集只用来定阈值：不同数据集的留出集难易差别很大，两边都看才不会某一侧成批误拒
    holdout_root = PROJECT_ROOT / config["root"] / config["holdout"]
    holdout = None
    if holdout_root.exists():
        hold_paths, hold_labels = collect_images(holdout_root, limit)
        if len(hold_paths) >= ood.MIN_SAMPLES:
            hf, hl, hp = extract_features(task, hold_paths, batch_size=batch_size)
            holdout = (hf, hl, hp, hold_labels)
            print(f"  留出集: {len(hold_paths)} 张（只参与定阈值与验证）")

    profile = ood.fit_profile(
        task, features, logits, probs, labels, pca_dim=pca_dim,
        holdout_features=holdout[0] if holdout else None,
    )
    path = ood.save_profile(profile)
    print(f"  已保存: {path}")
    print(
        f"  阈值: maha_reject={profile.maha_reject:.2f}  maha_warn={profile.maha_warn:.2f}"
    )
    print(
        f"  参考值: maha99={profile.ref_maha:.2f}  energy99={profile.ref_energy:.2f}  "
        f"pmax01={profile.ref_pmax * 100:.1f}%"
    )

    # 拟合集自检（应当没有拒绝）
    guard = ood.OODGuard(task, profile)
    stats = evaluate_profile(guard, features, logits, probs, labels, profile.classes)
    print(
        f"  拟合集: 拒绝 {stats['rejected']}/{stats['total']}，疑似 {stats['suspicious']}，"
        f"top-1 {('%.2f%%' % (stats['accuracy'] * 100)) if stats['accuracy'] else '—'}"
    )

    # 留出集验证：没参与拟合的同类数据
    if holdout:
        hf, hl, hp, hold_labels = holdout
        hold = evaluate_profile(guard, hf, hl, hp, hold_labels, profile.classes)
        print(
            f"  留出集({holdout_root.name}): 误拒率 {hold['reject_rate'] * 100:.2f}%  "
            f"疑似率 {hold['suspicious_rate'] * 100:.2f}%  "
            f"top-1 {('%.2f%%' % (hold['accuracy'] * 100)) if hold['accuracy'] else '—'}"
        )
    return profile


def cross_check(tasks: Sequence[str], limit: int = 20) -> None:
    """把 A 模型的数据集喂给 B 模型，检验跨任务误传能不能被拦下。"""
    print("\n=== 交叉验证：别的模型的数据喂给本模型 ===")
    print("（行 = 被测试的模型，列 = 图片来源；数值 = 被拒绝 / 被标疑似的比例）")

    guards = {task: ood.OODGuard(task) for task in tasks}
    samples: Dict[str, Tuple[List[Path], List[str]]] = {}
    for task in tasks:
        config = CLASSIFICATION_DATASETS[task]
        root = PROJECT_ROOT / config["root"] / config["fit"]
        try:
            samples[task] = collect_images(root, limit)
        except FileNotFoundError:
            continue

    header = "模型 \\ 图片".ljust(16) + "".join(task.ljust(16) for task in tasks)
    print(header)
    for target in tasks:
        guard = guards[target]
        if guard.profile is None:
            print(target.ljust(16) + "(无校准文件)")
            continue
        row = target.ljust(16)
        for source in tasks:
            if source not in samples:
                row += "n/a".ljust(16)
                continue
            paths, _ = samples[source]
            features, logits, probs = extract_features(target, paths)
            stats = evaluate_profile(guard, features, logits, probs)
            cell = f"{(stats['reject_rate'] + stats['suspicious_rate']) * 100:.0f}%"
            row += cell.ljust(16)
        print(row)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="校准非本模型样本拒识档案")
    parser.add_argument("--task", action="append", choices=sorted(CLASSIFICATION_DATASETS),
                        help="只处理指定任务，可重复")
    parser.add_argument("--limit", type=int, default=80, help="每类最多用多少张，默认 80")
    parser.add_argument("--batch", type=int, default=16, help="批量大小，默认 16")
    parser.add_argument("--pca", type=int, default=48, help="PCA 维度，默认 48")
    parser.add_argument("--verify-only", action="store_true", help="跳过校准，只验证已有档案")
    parser.add_argument("--cross", action="store_true", help="额外做跨任务交叉验证")
    args = parser.parse_args(argv)

    tasks = args.task or sorted(CLASSIFICATION_DATASETS)
    print("=" * 72)
    print("非本模型样本拒识 · 校准脚本")
    print("=" * 72)
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"校准目录: {ood.STATS_DIR}")

    if not args.verify_only:
        for task in tasks:
            try:
                calibrate(task, args.limit, args.batch, args.pca)
            except FileNotFoundError as exc:
                print(f"  [FAIL] {task}: {exc}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {task}: {exc}")
                import traceback
                traceback.print_exc()
        _restore_console_encoding()

    if args.cross:
        cross_check(tasks, limit=min(args.limit, 20))

    print("\n完成。启动服务后会自动加载 models/ood_stats/*.npz。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
