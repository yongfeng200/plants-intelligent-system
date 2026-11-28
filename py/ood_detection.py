# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # cv2 只在图像质量检查里用到，缺了也只是少一层判定
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("PLANT_MODELS_DIR", PROJECT_ROOT / "models"))
STATS_DIR = Path(os.environ.get("PLANT_OOD_DIR", MODELS_DIR / "ood_stats"))

# 阈值宽松度：>1 更宽松（更少拒绝），<1 更严格。
# 换数据源、换拍摄设备后如果误拒变多，把它调到 1.3 左右即可，不用重新校准。
STRICTNESS = float(os.environ.get("PLANT_OOD_STRICTNESS", "1.0"))

STATUS_OK = "ok"
STATUS_SUSPICIOUS = "suspicious"
STATUS_REJECTED = "rejected"

# 图像质量硬阈值
MIN_SIDE = 48            # 最短边小于该值基本没有判别信息
MIN_CONTRAST = 6.0       # 灰度标准差，低于它认为接近纯色
MIN_SHARPNESS = 6.0      # 拉普拉斯方差，低于它认为严重模糊（只提示，不拒绝）
MIN_SAMPLES = 20         # 校准至少需要这么多张，否则统计量不可信

PROFILE_VERSION = 2

# 每个任务用在提示文案里的说法
TASK_HINTS: Dict[str, Dict[str, Any]] = {
    "mango_leaf": {
        "subject": "芒果叶片",
        "expect": "8 类芒果叶片状态：炭疽病、细菌性溃疡病、切叶象甲、枝枯病、瘿蚊、健康、白粉病、煤污病",
        "tips": [
            "上传单张芒果叶片的清晰特写，叶片占画面主体，背景尽量简单",
            "不要上传整棵树、果实、其他作物、截图或文档照片",
        ],
        "example_dir": "Mango Leaf Dataset/test",
    },
    "mango_variety": {
        "subject": "芒果果实",
        "expect": "8 个芒果品种：安瓦尔·拉图尔、乔恩萨（黑/夏季巴希什特/白）、多塞里、法吉里、兰格拉、信德瑞",
        "tips": [
            "上传单颗或几颗芒果果实，果实占画面主体",
            "不要上传叶片、其他水果、聊天截图或风景照",
        ],
        "example_dir": "Mango Dataset/test",
    },
    "grape_disease": {
        "subject": "葡萄叶片",
        "expect": "4 类葡萄叶片状态：黑腐病、黑麻疹（ESCA）、健康、叶斑病",
        "tips": [
            "上传单张葡萄叶片的清晰特写，叶片占画面主体",
            "不要上传葡萄果实、芒果/番茄等其他作物、截图",
        ],
        "example_dir": "Grape Disease Dataset/test",
    },
    "leaf_disease": {
        "subject": "作物叶片",
        "expect": "5 种作物（玉米、葡萄、马铃薯、南瓜、番茄）的 10 种叶部病害",
        "tips": [
            "上传单张叶片、尽量拍到完整叶面和病斑",
            "不要上传果实、整株全景、人像或截图",
        ],
        "example_dir": "Leaf Disease Segment/classification/test",
    },
    "grape_segment": {
        "subject": "葡萄果实",
        "expect": "画面中能看出成串或成簇的葡萄果实",
        "tips": [
            "上传葡萄果穗照片，果实清晰、不要过度遮挡",
            "不要上传叶片特写、其他水果或纯风景",
        ],
        "example_dir": "Grape Detection Segment/yolo_seg_dataset/test",
    },
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class OODProfile:
    """一个任务的校准结果。所有数组都是 numpy，便于存 npz。"""

    task: str
    mean: np.ndarray            # (d,)  特征逐维均值
    std: np.ndarray             # (d,)  特征逐维标准差
    components: np.ndarray      # (k, d) PCA 主方向（行向量）
    class_means: np.ndarray     # (K, k) 各类在 PCA 空间的中心
    precision: np.ndarray       # (k, k) 池化协方差的逆（带收缩）
    score_mean: np.ndarray      # (3,)  三个指标的均值，仅用于诊断
    score_std: np.ndarray       # (3,)  三个指标的方差，仅用于诊断
    weights: np.ndarray         # (3,)  合成权重，仅用于诊断
    maha_reject: float          # 马氏距离拒绝阈值（主判据）
    maha_warn: float            # 马氏距离提示阈值
    reject_z: float             # 总分数拒绝阈值（仅诊断用）
    warn_z: float               # 总分数提示阈值
    z_ind: float                # 单指标“异常”阈值（z 分数）
    ref_maha: float             # 校准集中马氏距离的 99% 分位，用于文案
    ref_energy: float           # 校准集中能量分数的 99% 分位
    ref_pmax: float             # 校准集中最高概率的 1% 分位
    n_samples: int = 0
    classes: List[str] = field(default_factory=list)
    version: int = PROFILE_VERSION

    @property
    def feature_dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def pca_dim(self) -> int:
        return int(self.components.shape[0])


def profile_path(task: str) -> Path:
    return STATS_DIR / f"{task}.npz"


# ---------------------------------------------------------------------------
# 分数计算
# ---------------------------------------------------------------------------

def _project(features: np.ndarray, profile: OODProfile) -> np.ndarray:
    """标准化 + PCA 投影： (N, d) -> (N, k)"""
    normalized = (np.asarray(features, dtype=np.float64) - profile.mean) / profile.std
    return normalized @ profile.components.T


def _mahalanobis(projected: np.ndarray, profile: OODProfile) -> np.ndarray:
    """到最近类原型的马氏距离，返回 (N,)。"""
    best = np.full(projected.shape[0], np.inf, dtype=np.float64)
    for mu in profile.class_means:
        diff = projected - mu
        dist = np.einsum("ij,jk,ik->i", diff, profile.precision, diff)
        best = np.minimum(best, dist)
    return np.sqrt(np.maximum(best, 0.0))


def _energy(logits: np.ndarray) -> np.ndarray:
    """energy = logsumexp(logits)，越大越不像训练分布。"""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted.max(axis=1) + np.log(np.exp(shifted).sum(axis=1))


def _neg_log_pmax(probs: np.ndarray) -> np.ndarray:
    """-log(最大概率)，越大越不确定。"""
    probs = np.asarray(probs, dtype=np.float64)
    return -np.log(np.clip(probs.max(axis=1), 1e-12, 1.0))


def _normalize_entropy(probs: np.ndarray) -> np.ndarray:
    """熵 / log(K)，落在 [0, 1]，1 表示完全平均。"""
    probs = np.asarray(probs, dtype=np.float64)
    k = probs.shape[1]
    safe = np.clip(probs, 1e-12, 1.0)
    entropy = -(safe * np.log(safe)).sum(axis=1)
    return entropy / (np.log(k) if k > 1 else 1.0)


def _raw_scores(features: np.ndarray, logits: np.ndarray,
                probs: np.ndarray, profile: Optional[OODProfile]) -> Dict[str, np.ndarray]:
    """返回三个原始指标：马氏距离（无校准文件时为 nan）、能量、-log(pmax)。"""
    scores: Dict[str, np.ndarray] = {
        "energy": _energy(logits),
        "neg_log_pmax": _neg_log_pmax(probs),
        "entropy": _normalize_entropy(probs),
        "max_prob": np.asarray(probs, dtype=np.float64).max(axis=1),
    }
    if profile is not None and features is not None:
        projected = _project(features, profile)
        scores["mahalanobis"] = _mahalanobis(projected, profile)
    else:
        scores["mahalanobis"] = np.array([np.nan], dtype=np.float64)
    return scores


# ---------------------------------------------------------------------------
# 校准
# ---------------------------------------------------------------------------

def fit_profile(task: str,
                features: np.ndarray,
                logits: np.ndarray,
                probs: np.ndarray,
                labels: Optional[Sequence[Any]] = None,
                pca_dim: int = 48,
                shrink: float = 0.1,
                holdout_features: Optional[np.ndarray] = None) -> OODProfile:
    """用同分布的样本（一般是测试集）拟合一份校准档案。

    features: (N, d) 倒数第二层特征
    logits:   (N, K) 模型原始输出
    probs:    (N, K) softmax 概率
    labels:   (N,)   每条样本的类名（用于算类中心；为空时只用一个全局中心）
    holdout_features: 另一切分（如 val）的特征，只用来定阈值，不参与拟合。
                      不同数据集的留出集难易差别很大，只用拟合集定阈值容易被少数
                      离群图片撑大，两边都看才不会某一侧成批误拒。
    """
    features = np.asarray(features, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)

    n, d = features.shape
    if n < MIN_SAMPLES:
        raise ValueError(f"校准样本太少（{n} < {MIN_SAMPLES}），统计量不可信")

    pca_dim = int(max(2, min(pca_dim, n - 1, d)))

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1e-6

    normalized = (features - mean) / std
    # 只需主方向：用 SVD 拿 Vt 的前 pca_dim 行
    _, _, vt = np.linalg.svd(normalized, full_matrices=False)
    components = vt[:pca_dim].copy()

    projected = normalized @ components.T

    if labels is None:
        groups = [np.arange(n)]
        class_names: List[str] = []
    else:
        labels = list(labels)
        unique = sorted({str(label) for label in labels})
        groups = [np.array([i for i, label in enumerate(labels) if str(label) == name]) for name in unique]
        class_names = unique

    class_means = np.stack([projected[idx].mean(axis=0) for idx in groups])

    # 池化类内协方差 + 向单位阵收缩，保证可逆、数值稳定
    residuals = np.concatenate([projected[idx] - class_means[g] for g, idx in enumerate(groups)])
    cov = residuals.T @ residuals / max(1, residuals.shape[0])
    dim = cov.shape[0]
    cov = (1.0 - shrink) * cov + shrink * (np.trace(cov) / dim + 1e-6) * np.eye(dim)
    precision = np.linalg.inv(cov)

    placeholder = OODProfile(
        task=task, mean=mean, std=std, components=components,
        class_means=class_means, precision=precision,
        score_mean=np.zeros(3), score_std=np.ones(3), weights=np.array([0.5, 0.3, 0.2]),
        maha_reject=float("inf"), maha_warn=float("inf"),
        reject_z=3.0, warn_z=2.0, z_ind=3.0,
        ref_maha=float("nan"), ref_energy=float("nan"), ref_pmax=float("nan"),
        n_samples=n, classes=class_names,
    )

    raw = _raw_scores(features, logits, probs, placeholder)
    stacked = np.stack([raw["mahalanobis"], raw["energy"], raw["neg_log_pmax"]], axis=1)
    score_mean = stacked.mean(axis=0)
    score_std = stacked.std(axis=0)
    score_std[score_std < 1e-6] = 1e-6

    z = (stacked - score_mean) / score_std

    # 有些模型过度自信：训练分布内 energy 和 -log(pmax) 几乎是常数。这类指标的
    # 方差趋近于 0，OOD 样本稍微一动就会算出几十万倍的 z，把马氏距离的结论淹没。
    # 所以只在训练分布内真正有波动的指标才有资格参与判定。
    scale_ref = np.abs(score_mean) + np.array([1.0, 1.0, 0.1])
    informative = score_std > 0.001 * scale_ref
    weights = np.array([0.5, 0.3, 0.2])
    weights[~informative] = 0.0
    if weights.sum() <= 0:  # 极端情况：至少保留特征层
        weights = np.array([1.0, 0.0, 0.0])
    weights = weights / weights.sum()

    combined = z @ weights

    reject_z = float(np.quantile(combined, 0.995))
    warn_z = float(np.quantile(combined, 0.98))
    z_ind = float(np.quantile((z * (weights > 0)).max(axis=1), 0.99))

    # 兜底：阈值不能低到让正常样本成批被拒，也不能高到形同虚设
    reject_z = max(reject_z, 2.0)
    warn_z = float(min(max(warn_z, 1.0), reject_z * 0.85))
    z_ind = float(min(max(z_ind, 2.5), 6.0))

    # 主判据：马氏距离阈值。
    # 取「拟合集 95 分位」与「留出集 98 分位」中较大的那个：
    #   - 只看拟合集，会被少数离群图片把阈值撑大（芒果叶片实测 p99 比 val p99 高 1.9）；
    #   - 只看留出集，又可能在两个切分难度差异大时把阈值压得过低。
    maha_fit = raw["mahalanobis"]
    maha_reject = float(np.quantile(maha_fit, 0.95))
    if holdout_features is not None and len(holdout_features) >= MIN_SAMPLES:
        holdout = np.asarray(holdout_features, dtype=np.float64)
        maha_hold = _mahalanobis(_project(holdout, placeholder), placeholder)
        maha_reject = max(maha_reject, float(np.quantile(maha_hold, 0.98)))
    else:
        maha_reject = float(np.quantile(maha_fit, 0.97))
    # 提示阈值贴着拒绝阈值，只留一条窄灰带：宁可少打扰，也不要让大量正常图片
    # 被标成「疑似」（实测按 fit p90 取会给芒果品种带来 17.5% 的疑似率）
    maha_warn = float(maha_reject * 0.94)

    placeholder.score_mean = score_mean
    placeholder.score_std = score_std
    placeholder.weights = weights
    placeholder.maha_reject = maha_reject
    placeholder.maha_warn = maha_warn
    placeholder.reject_z = reject_z
    placeholder.warn_z = warn_z
    placeholder.z_ind = z_ind
    placeholder.ref_maha = float(np.quantile(raw["mahalanobis"], 0.99))
    placeholder.ref_energy = float(np.quantile(raw["energy"], 0.99))
    placeholder.ref_pmax = float(np.quantile(raw["max_prob"], 0.01))
    return placeholder


def save_profile(profile: OODProfile, path: Optional[Path] = None) -> Path:
    path = Path(path) if path else profile_path(profile.task)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": profile.version,
        "task": profile.task,
        "classes": list(profile.classes),
        "n_samples": int(profile.n_samples),
        "pca_dim": profile.pca_dim,
        "feature_dim": profile.feature_dim,
        "thresholds": {
            "maha_reject": profile.maha_reject,
            "maha_warn": profile.maha_warn,
            "reject_z": profile.reject_z,
            "warn_z": profile.warn_z,
            "z_ind": profile.z_ind,
            "ref_maha": profile.ref_maha,
            "ref_energy": profile.ref_energy,
            "ref_pmax": profile.ref_pmax,
        },
    }
    np.savez_compressed(
        path,
        mean=profile.mean,
        std=profile.std,
        components=profile.components,
        class_means=profile.class_means,
        precision=profile.precision,
        score_mean=profile.score_mean,
        score_std=profile.score_std,
        weights=profile.weights,
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    return path


def load_profile(task: str, path: Optional[Path] = None) -> Optional[OODProfile]:
    path = Path(path) if path else profile_path(task)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            thresholds = meta.get("thresholds", {})
            version = int(meta.get("version", 1))
            if version < PROFILE_VERSION:
                print(
                    f"[OOD] {path.name} 是旧版校准文件（v{version}），"
                    "请重新运行 python py/build_ood_stats.py"
                )
                return None
            return OODProfile(
                task=task,
                mean=data["mean"],
                std=data["std"],
                components=data["components"],
                class_means=data["class_means"],
                precision=data["precision"],
                score_mean=data["score_mean"],
                score_std=data["score_std"],
                weights=data["weights"] if "weights" in data else np.array([0.5, 0.3, 0.2]),
                maha_reject=float(thresholds.get("maha_reject", float("inf"))),
                maha_warn=float(thresholds.get("maha_warn", float("inf"))),
                reject_z=float(thresholds.get("reject_z", 3.0)),
                warn_z=float(thresholds.get("warn_z", 2.0)),
                z_ind=float(thresholds.get("z_ind", 3.0)),
                ref_maha=float(thresholds.get("ref_maha", float("nan"))),
                ref_energy=float(thresholds.get("ref_energy", float("nan"))),
                ref_pmax=float(thresholds.get("ref_pmax", float("nan"))),
                n_samples=int(meta.get("n_samples", 0)),
                classes=list(meta.get("classes") or []),
                version=version,
            )
    except Exception as exc:  # noqa: BLE001 - 校准文件坏了就退回保守模式
        print(f"[OOD] 读取校准文件失败 {path.name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# 图像质量
# ---------------------------------------------------------------------------

def check_image_quality(image_rgb: Optional[np.ndarray]) -> List[Dict[str, str]]:
    """返回问题列表；level=reject 表示根本不该送进模型。"""
    issues: List[Dict[str, str]] = []
    if image_rgb is None:
        return issues

    array = np.asarray(image_rgb)
    if array.size == 0:
        return [{"level": "reject", "code": "empty", "message": "图片内容为空，无法读取"}]

    height, width = array.shape[0], array.shape[1]
    if min(height, width) < MIN_SIDE:
        issues.append({
            "level": "reject",
            "code": "too_small",
            "message": f"图片分辨率过低（{width}×{height}），最短边至少需要 {MIN_SIDE} 像素",
        })
        return issues

    if cv2 is None:
        return issues

    if array.ndim == 3:
        gray = cv2.cvtColor(array[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = array.astype(np.uint8)

    contrast = float(gray.std())
    if contrast < MIN_CONTRAST:
        issues.append({
            "level": "reject",
            "code": "flat",
            "message": f"图片几乎是纯色（对比度 {contrast:.1f}），看不到可识别的内容",
        })
        return issues

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < MIN_SHARPNESS:
        issues.append({
            "level": "warn",
            "code": "blurry",
            "message": f"图片较为模糊（清晰度 {sharpness:.1f}），可能影响识别",
        })
    return issues


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------

def _format_reasons(scores: Dict[str, float], profile: OODProfile) -> List[str]:
    """给用户看的拒绝原因。

    这里刻意不出现马氏距离、能量、z 分数这些内部数值——用户只需要知道
    「为什么这张图不行」，数值留给 ood.scores 供开发排查。
    """
    reasons: List[str] = []
    maha = scores.get("mahalanobis", float("nan"))
    if not np.isnan(maha) and maha > profile.maha_warn:
        reasons.append("图片的整体特征与模型学习过的样本差别较大")
    if scores.get("max_prob", 1.0) < 0.45:
        reasons.append(
            f"模型对任何一种类别都没有把握（最高只有 {scores['max_prob'] * 100:.0f}%）"
        )
    return reasons


def task_hints(task: str, calibrated: bool = True) -> List[str]:
    """给具体任务生成“该上传什么图”的提示，用于拒绝时的界面文案。"""
    meta = TASK_HINTS.get(task, {})
    hints = list(meta.get("tips") or [])
    expect = meta.get("expect")
    if expect:
        hints.append(f"本模型只能识别：{expect}")
    if not calibrated:
        hints.append("提示：尚未生成校准文件，建议在服务器上运行 python py/build_ood_stats.py")
    return hints


class OODGuard:
    """一个任务的拒识判定器。线程安全地持有校准档案。"""

    def __init__(self, task: str, profile: Optional[OODProfile] = None):
        self.task = task
        self.profile = profile if profile is not None else load_profile(task)

    # -- 文案 -------------------------------------------------------------
    def _hints(self) -> List[str]:
        return task_hints(self.task, calibrated=self.profile is not None)

    # -- 主入口 -----------------------------------------------------------
    def evaluate(self,
                 features: Optional[np.ndarray],
                 logits: np.ndarray,
                 probs: np.ndarray,
                 image_rgb: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """返回判定结果字典，调用方据此决定是否给出预测。"""
        logits = np.asarray(logits, dtype=np.float64).reshape(1, -1)
        probs = np.asarray(probs, dtype=np.float64).reshape(1, -1)
        features_arr = None
        if features is not None and self.profile is not None:
            features_arr = np.asarray(features, dtype=np.float64).reshape(1, -1)
            if features_arr.shape[1] != self.profile.feature_dim:
                # 特征维度对不上（换了主干），只用作废的特征层，别让整个请求 500
                print(
                    f"[OOD] {self.task} 特征维度不匹配："
                    f"{features_arr.shape[1]} vs 校准 {self.profile.feature_dim}，已忽略特征层判定"
                )
                features_arr = None

        raw = _raw_scores(
            features_arr if features_arr is not None else np.zeros((1, 1)),
            logits, probs,
            self.profile if features_arr is not None else None,
        )

        score = {
            "mahalanobis": float(raw["mahalanobis"][0]),
            "energy": float(raw["energy"][0]),
            "neg_log_pmax": float(raw["neg_log_pmax"][0]),
            "max_prob": float(raw["max_prob"][0]),
            "entropy": float(raw["entropy"][0]),
        }

        quality = check_image_quality(image_rgb)
        subject = TASK_HINTS.get(self.task, {}).get("subject", "该模型")

        # ---- 图像层直接拒绝
        hard = [item for item in quality if item["level"] == "reject"]
        if hard:
            return {
                "status": STATUS_REJECTED,
                "score": None,
                "scores": score,
                "reasons": [item["message"] for item in hard],
                "message": f"这张图片无法作为{subject}样本使用。",
                "hints": self._hints(),
                "quality": quality,
                "calibrated": self.profile is not None,
            }

        # ---- 没有校准档案（或模型拿不到倒数第二层特征）：只看置信度与熵
        if self.profile is None or features_arr is None:
            status = STATUS_OK
            reasons: List[str] = []
            pmax = score["max_prob"]
            if pmax < 0.55 and score["entropy"] > 0.7:
                status = STATUS_REJECTED
                reasons.append(f"模型对哪一类都没把握（最高概率 {pmax * 100:.1f}%，熵 {score['entropy']:.2f}）")
            elif pmax < 0.75 or score["entropy"] > 0.75:
                status = STATUS_SUSPICIOUS
                reasons.append(f"置信度偏低（最高概率 {pmax * 100:.1f}%）")
            message = (
                f"这张图片可能不是本模型能识别的{subject}样本。"
                if status != STATUS_OK else ""
            )
            return {
                "status": status,
                "score": None,
                "scores": score,
                "reasons": reasons,
                "message": message,
                "hints": task_hints(self.task, calibrated=False),
                "quality": quality,
                "calibrated": False,
            }

        # ---- 主判定：马氏距离
        # 实测过：把三个指标加权合成反而稀释了信号（芒果叶片模型上，合成分数在
        # 误拒 1.2% 时只拦下 33.8% 的葡萄叶片；单用马氏距离阈值在误拒 0.8% 时能拦 94.6%）。
        # 所以特征距离当主判据，置信度只作为「模型自己也拿不准」的补充。
        profile = self.profile
        maha = score["mahalanobis"]
        pmax = score["max_prob"]

        strictness = STRICTNESS if STRICTNESS > 0 else 1.0
        maha_reject = profile.maha_reject * strictness
        maha_warn = profile.maha_warn * strictness

        if maha > maha_reject or pmax < 0.25:
            status = STATUS_REJECTED
        elif maha > maha_warn or pmax < 0.45:
            status = STATUS_SUSPICIOUS
        else:
            status = STATUS_OK

        reasons = _format_reasons(score, profile)
        reasons += [item["message"] for item in quality if item["level"] == "warn"]

        message = ""
        if status == STATUS_REJECTED:
            message = f"这张图片不是可识别的{subject}样本，系统没有输出预测结果。"
        elif status == STATUS_SUSPICIOUS:
            message = f"这张图片与模型见过的{subject}样本差异较大，结果仅供参考。"

        # 合成分数只作为诊断信息返回，不参与判定
        stacked = np.array([[maha, score["energy"], score["neg_log_pmax"]]])
        z = ((stacked - profile.score_mean) / profile.score_std)[0]
        weights = profile.weights if profile.weights is not None else np.array([0.5, 0.3, 0.2])
        combined = float(z @ weights)

        return {
            "status": status,
            "score": round(combined, 3),
            "scores": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in score.items()},
            "thresholds": {
                "maha_reject": round(maha_reject, 2),
                "maha_warn": round(maha_warn, 2),
            },
            "reasons": reasons,
            "message": message,
            "hints": self._hints(),
            "quality": quality,
            "calibrated": True,
        }


# ---------------------------------------------------------------------------
# 全局缓存
# ---------------------------------------------------------------------------

_GUARDS: Dict[str, OODGuard] = {}
_LOCK = threading.Lock()


def get_guard(task: str) -> OODGuard:
    """按任务取判定器；校准文件只读一次。"""
    guard = _GUARDS.get(task)
    if guard is not None:
        return guard
    with _LOCK:
        guard = _GUARDS.get(task)
        if guard is None:
            guard = OODGuard(task)
            _GUARDS[task] = guard
            if guard.profile is None:
                print(
                    f"[OOD] {task} 未找到校准文件 {profile_path(task).name}，"
                    "退化为置信度模式；建议运行 python py/build_ood_stats.py"
                )
        return guard


def reload_guard(task: str) -> OODGuard:
    """校准脚本写完文件后调用，让运行中的服务立刻生效。"""
    with _LOCK:
        _GUARDS.pop(task, None)
    return get_guard(task)


def available_profiles() -> Dict[str, Dict[str, Any]]:
    """给状态页用：哪些任务已经有校准档案。"""
    result: Dict[str, Dict[str, Any]] = {}
    for task in TASK_HINTS:
        path = profile_path(task)
        profile = load_profile(task) if path.exists() else None
        result[task] = {
            "exists": profile is not None,
            "path": str(path),
            "samples": profile.n_samples if profile else 0,
            "classes": profile.classes if profile else [],
            "reject_z": round(profile.reject_z, 2) if profile else None,
            "warn_z": round(profile.warn_z, 2) if profile else None,
        }
    return result


# ---------------------------------------------------------------------------
# 响应辅助
# ---------------------------------------------------------------------------

def rejection_payload(ood: Dict[str, Any],
                      model: Optional[Dict[str, Any]] = None,
                      extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """拒绝时返回给前端的结构：不带预测类别，只带原因和建议。"""
    payload: Dict[str, Any] = {
        "valid": False,
        "status": ood.get("status", STATUS_REJECTED),
        "message": ood.get("message") or "这张图片无法被本模型识别。",
        "reasons": list(ood.get("reasons") or []),
        "hints": list(ood.get("hints") or []),
        "checks": list(ood.get("quality") or []),
        "ood": {
            "score": ood.get("score"),
            "scores": ood.get("scores"),
            "calibrated": bool(ood.get("calibrated")),
        },
    }
    if model:
        payload["model"] = model
    if extra:
        payload.update(extra)
    return payload


def success_fields(ood: Dict[str, Any]) -> Dict[str, Any]:
    """正常返回时附加的字段：valid 恒为 True，疑似样本额外带 warning。"""
    fields: Dict[str, Any] = {
        "valid": True,
        "ood_status": ood.get("status", STATUS_OK),
    }
    if ood.get("status") == STATUS_SUSPICIOUS:
        fields["warning"] = {
            "message": ood.get("message") or "这张图片可能不是本模型的典型样本，结果仅供参考。",
            "reasons": list(ood.get("reasons") or []),
            "hints": list(ood.get("hints") or []),
        }
    return fields
