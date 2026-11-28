# -*- coding: utf-8 -*-
"""
统一模型注册表
================

把「训练脚本保存的模型文件」和「Web 推理」严格对齐，避免出现
模型路径写错、网络结构写错、加载失败后偷偷使用随机模拟结果的问题。

提供的能力：
1. 统一的模型路径 / 设备 / 类别 / 准确率配置；
2. 与训练 Notebook 完全一致的网络结构与预处理；
3. 懒加载 + 线程锁，避免重复加载显存；
4. ``ModelNotReady`` 异常：模型没准备好时明确报错，绝不伪造预测结果；
5. 统一的推理结果结构（中文/英文标签、概率、耗时、模型信息）。

不在本表里的模块：叶片病灶分割 + 物种/疾病识别。它是 UNet++ 与 DINOv2 两个权重
串起来的组合推理，收敛在 ``leaf_disease_models.LeafDiseasePipeline``，不要重复登记。
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

import ood_detection
from ood_detection import get_guard

# ---------------------------------------------------------------------------
# 路径与设备
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 允许通过环境变量指定模型目录，便于容器化部署或本地调试。
MODELS_DIR = Path(os.environ.get("PLANT_MODELS_DIR", PROJECT_ROOT / "models"))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def device_name() -> str:
    """返回可展示的推理设备名称。"""
    if DEVICE.type == "cuda":
        try:
            return f"CUDA · {torch.cuda.get_device_name(0)}"
        except Exception:  # pragma: no cover - 极少数驱动异常
            return "CUDA"
    return "CPU"


# ---------------------------------------------------------------------------
# 类别定义（顺序必须与训练时 ImageFolder 的字母序一致）
# ---------------------------------------------------------------------------

MANGO_LEAF_CLASSES = [
    {"cn": "炭疽病", "en": "Anthracnose"},
    {"cn": "细菌性溃疡病", "en": "Bacterial Canker"},
    {"cn": "切叶象甲", "en": "Cutting Weevil"},
    {"cn": "枝枯病", "en": "Die Back"},
    {"cn": "瘿蚊", "en": "Gall Midge"},
    {"cn": "健康", "en": "Healthy"},
    {"cn": "白粉病", "en": "Powdery Mildew"},
    {"cn": "煤污病", "en": "Sooty Mould"},
]

MANGO_VARIETY_CLASSES = [
    {"cn": "安瓦尔·拉图尔", "en": "Anwar Ratool"},
    {"cn": "乔恩萨（黑）", "en": "Chaunsa (Black)"},
    {"cn": "乔恩萨（夏季巴希什特）", "en": "Chaunsa (Summer Bahisht)"},
    {"cn": "乔恩萨（白）", "en": "Chaunsa (White)"},
    {"cn": "多塞里", "en": "Dosehri"},
    {"cn": "法吉里", "en": "Fajri"},
    {"cn": "兰格拉", "en": "Langra"},
    {"cn": "信德瑞", "en": "Sindhri"},
]

GRAPE_DISEASE_CLASSES = [
    {"cn": "黑腐病", "en": "Black Rot"},
    {"cn": "黑麻疹（ESCA）", "en": "ESCA"},
    {"cn": "健康", "en": "Healthy"},
    {"cn": "叶斑病", "en": "Leaf Blight"},
]

GRAPE_VARIETY_CLASSES = [
    {"cn": "霞多丽", "en": "Chardonnay"},
    {"cn": "灰皮诺", "en": "PinotGris"},
    {"cn": "黑皮诺", "en": "PinotNoir"},
]


# ---------------------------------------------------------------------------
# 预处理（与对应训练 Notebook 的 eval_transform 完全一致）
# ---------------------------------------------------------------------------

# train/MangoLeafDiseaseMDCNTraining.ipynb: DATASET_MEAN / DATASET_STD
_MANGO_LEAF_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.6690, 0.6907, 0.6845], [0.2018, 0.1920, 0.2639]),
])

# train/MangoVarietyClassificationTraining.ipynb / GrapeDiseaseTraining.ipynb
_IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# 网络结构（与训练脚本逐层对齐）
# ---------------------------------------------------------------------------

class MDCN(nn.Module):
    """DenseNet-121 + 8 类分类头（MangoLeafDiseaseMDCNTraining.ipynb）。"""

    def __init__(self, num_classes: int = 8, dropout: float = 0.2):
        super().__init__()
        backbone = models.densenet121(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )
        self.num_classes = num_classes

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """倒数第二层特征（1024 维），OOD 判定用。"""
        return torch.flatten(self.pool(torch.relu(self.features(x))), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)


class GrapeDiseaseNet(nn.Module):
    """ResNet50 + CBAM（train/GrapeDiseaseTraining.ipynb）。"""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.cbam = CBAM(2048)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )

    def forward_features(self, x):
        """CBAM 之后的池化特征（2048 维），OOD 判定用。"""
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        x = self.cbam(x)
        return torch.flatten(b.avgpool(x), 1)

    def forward(self, x):
        return self.backbone.fc(self.forward_features(x))


class MangoVarietyNet(nn.Module):
    """EfficientNet-B4（train/MangoVarietyClassificationTraining.ipynb）。"""

    def __init__(self, num_classes: int = 8):
        super().__init__()
        base = models.efficientnet_b4(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(base.classifier[1].in_features, num_classes),
        )

    def forward_features(self, x):
        """全局池化后的特征（1792 维），OOD 判定用。"""
        return torch.flatten(self.avgpool(self.features(x)), 1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


# ---------------------------------------------------------------------------
# 任务配置
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict[str, Any]] = {
    "mango_leaf": {
        "name": "芒果叶片病害识别",
        "short": "芒果叶片病害",
        "architecture": "MDCN · DenseNet-121",
        "model_type": "芒果叶片病害",
        "classes": MANGO_LEAF_CLASSES,
        "transform": _MANGO_LEAF_TRANSFORM,
        "checkpoints": ["mango_mdcn_best_checkpoint.pth", "mango_mdcn_model.pth"],
        "accuracy": 0.9975,
        "builder": lambda: MDCN(num_classes=len(MANGO_LEAF_CLASSES)),
    },
    "mango_variety": {
        "name": "芒果品种识别",
        "short": "芒果品种",
        "architecture": "EfficientNet-B4",
        "model_type": "芒果品种",
        "classes": MANGO_VARIETY_CLASSES,
        "transform": _IMAGENET_TRANSFORM,
        "checkpoints": ["mango_variety_best_checkpoint.pth", "mango_variety_model.pth"],
        "accuracy": 1.0,
        "builder": lambda: MangoVarietyNet(num_classes=len(MANGO_VARIETY_CLASSES)),
    },
    "grape_disease": {
        "name": "葡萄病害识别",
        "short": "葡萄病害",
        "architecture": "ResNet50 + CBAM",
        "model_type": "葡萄病害",
        "classes": GRAPE_DISEASE_CLASSES,
        "transform": _IMAGENET_TRANSFORM,
        "checkpoints": ["grape_disease_model.pth", "grape_disease_best_checkpoint.pth"],
        "accuracy": 0.9989,
        "builder": lambda: GrapeDiseaseNet(num_classes=len(GRAPE_DISEASE_CLASSES)),
    },
    "grape_segment": {
        "name": "葡萄果实分割识别",
        "short": "葡萄实例分割",
        "architecture": "YOLOv8-seg",
        "model_type": "葡萄实例分割",
        "classes": GRAPE_VARIETY_CLASSES,
        "transform": None,
        "checkpoints": ["grape_segment_model.pt"],
        # 实例分割不报 accuracy，用训练时的 mAP50(B)
        "accuracy": 0.9652,
        "accuracy_label": "mAP50",
        "builder": None,
    },
}


class ModelNotReady(RuntimeError):
    """模型未加载 / 权重不匹配时抛出，调用方必须返回明确错误。"""


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _torch_load(path: Path):
    """兼容不同 PyTorch 版本的加载函数。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint):
    """从 checkpoint 中提取 state_dict，兼容多种保存格式。"""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        # 整个文件就是一个 state_dict
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
        raise ModelNotReady("权重文件缺少 model_state_dict / state_dict")
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    raise ModelNotReady(f"无法识别的权重文件格式: {type(checkpoint)!r}")


def _pick_checkpoint(names: List[str]) -> Path:
    for name in names:
        path = MODELS_DIR / name
        if path.exists():
            return path
    raise ModelNotReady(
        "未找到模型文件，已尝试: " + "、".join(str(MODELS_DIR / n) for n in names)
    )


def _file_size_mb(path: Optional[Path]) -> Optional[float]:
    if path and path.exists():
        return round(path.stat().st_size / (1024 * 1024), 2)
    return None


def _class_names(task: str, lang: str = "cn") -> List[str]:
    return [item[lang] for item in TASKS[task]["classes"]]


def _restore_console_encoding() -> None:
    """恢复项目配置的控制台编码。

    ultralytics 在导入时会强制把 sys.stdout 改成 UTF-8，这会让 PyCharm
    的 Run 控制台（按 GBK 解码）再次出现中文乱码。
    """
    try:
        import console_encoding

        console_encoding.restore_console_encoding()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    key: str
    model: Any
    checkpoint: Path
    names: Dict[int, str]
    loaded_at: float


class ModelRegistry:
    """线程安全的懒加载模型注册表。"""

    def __init__(self):
        self._entries: Dict[str, ModelEntry] = {}
        self._errors: Dict[str, str] = {}
        self._lock = threading.Lock()

    # -- 分类模型 ---------------------------------------------------------
    def get(self, key: str) -> ModelEntry:
        if key not in TASKS:
            raise KeyError(f"未知模型任务: {key}")
        if key == "grape_segment":
            raise KeyError("grape_segment 请使用 get_detector()")

        entry = self._entries.get(key)
        if entry is not None:
            return entry

        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                return entry

            task = TASKS[key]
            checkpoint = _pick_checkpoint(task["checkpoints"])
            checkpoint_obj = _torch_load(checkpoint)
            state_dict = _extract_state_dict(checkpoint_obj)

            # 如果 checkpoint 里记录了类别顺序，必须与线上配置一致，避免标签错位
            if isinstance(checkpoint_obj, dict):
                saved_classes = checkpoint_obj.get("class_to_idx") or checkpoint_obj.get("classes")
                if saved_classes:
                    saved_order = list(saved_classes.keys()) if isinstance(saved_classes, dict) else list(saved_classes)
                    expected_order = [item["en"] for item in task["classes"]]
                    if saved_order != expected_order:
                        raise ModelNotReady(
                            f"{task['name']} 类别顺序与训练权重不一致："
                            f"权重={saved_order}，配置={expected_order}"
                        )

            state_dict = {
                (k[len("module."):] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }

            model = task["builder"]()
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as exc:
                raise ModelNotReady(
                    f"{task['name']} 权重与网络结构不匹配: {exc}"
                ) from exc

            model.to(DEVICE)
            model.eval()

            # 首次推理做一次预热，避免用户第一次上传时等待过久
            try:
                with torch.inference_mode():
                    model(torch.zeros(1, 3, 224, 224, device=DEVICE))
            except Exception as exc:  # noqa: BLE001 - 预热失败不影响正式推理
                print(f"[ModelRegistry] 预热失败（可忽略）: {exc}")

            entry = ModelEntry(
                key=key,
                model=model,
                checkpoint=checkpoint,
                names={i: item["en"] for i, item in enumerate(task["classes"])},
                loaded_at=time.time(),
            )
            self._entries[key] = entry
            self._errors.pop(key, None)
            print(f"[ModelRegistry] [OK] {task['name']} 已加载: {checkpoint.name} ({device_name()})")
            return entry

    # -- YOLO 检测/分割模型 ----------------------------------------------
    def get_detector(self):
        entry = self._entries.get("grape_segment")
        if entry is not None:
            return entry

        with self._lock:
            entry = self._entries.get("grape_segment")
            if entry is not None:
                return entry

            try:
                from ultralytics import YOLO
            except ImportError as exc:
                _restore_console_encoding()
                raise ModelNotReady("未安装 ultralytics，无法运行葡萄实例分割") from exc
            else:
                _restore_console_encoding()

            checkpoint = _pick_checkpoint(TASKS["grape_segment"]["checkpoints"])
            model = YOLO(str(checkpoint))
            _restore_console_encoding()
            names = {int(k): str(v) for k, v in (getattr(model, "names", {}) or {}).items()}

            entry = ModelEntry(
                key="grape_segment",
                model=model,
                checkpoint=checkpoint,
                names=names,
                loaded_at=time.time(),
            )
            self._entries["grape_segment"] = entry
            self._errors.pop("grape_segment", None)
            print(f"[ModelRegistry] [OK] 葡萄果实分割识别 已加载: {checkpoint.name} ({device_name()})")
            return entry

    # -- 状态 -------------------------------------------------------------
    def status(self) -> List[Dict[str, Any]]:
        result = []
        for key, task in TASKS.items():
            checkpoint = next(
                (MODELS_DIR / name for name in task["checkpoints"] if (MODELS_DIR / name).exists()),
                None,
            )
            loaded = key in self._entries
            result.append({
                "key": key,
                "name": task["name"],
                "architecture": task["architecture"],
                "path": str(checkpoint or (MODELS_DIR / task["checkpoints"][0])),
                "filename": checkpoint.name if checkpoint else task["checkpoints"][0],
                "exists": checkpoint is not None,
                "loaded": loaded,
                "status": "已加载" if loaded else ("文件就绪" if checkpoint else "文件缺失"),
                "size_mb": _file_size_mb(checkpoint),
                "accuracy": task["accuracy"],
                "accuracy_label": task.get("accuracy_label", "测试准确率"),
                "accuracy_text": f"{task['accuracy'] * 100:.2f}%" if task["accuracy"] else "—",
                "device": device_name() if loaded else "—",
                "classes": [item["cn"] for item in task["classes"]],
                "classes_en": [item["en"] for item in task["classes"]],
                "error": self._errors.get(key),
            })
        return result

    def record_error(self, key: str, message: str) -> None:
        self._errors[key] = message

    def preload(self, keys: Optional[List[str]] = None) -> Dict[str, str]:
        """预先加载模型，返回 {key: error}。"""
        targets = keys or [k for k in TASKS if k != "grape_segment"]
        errors: Dict[str, str] = {}
        for key in targets:
            try:
                if key == "grape_segment":
                    self.get_detector()
                else:
                    self.get(key)
            except Exception as exc:  # noqa: BLE001 - 启动阶段需要汇总所有错误
                self.record_error(key, str(exc))
                errors[key] = str(exc)
                print(f"[ModelRegistry] [FAIL] {TASKS[key]['name']} 加载失败: {exc}")
        return errors


REGISTRY = ModelRegistry()


# ---------------------------------------------------------------------------
# 推理接口
# ---------------------------------------------------------------------------

def get_model_entry(key: str) -> ModelEntry:
    try:
        return REGISTRY.get(key)
    except Exception as exc:
        REGISTRY.record_error(key, str(exc))
        if isinstance(exc, ModelNotReady):
            raise
        raise ModelNotReady(str(exc)) from exc


def get_detector_entry() -> ModelEntry:
    """获取 YOLO 检测/分割模型。"""
    try:
        return REGISTRY.get_detector()
    except Exception as exc:
        REGISTRY.record_error("grape_segment", str(exc))
        if isinstance(exc, ModelNotReady):
            raise
        raise ModelNotReady(str(exc)) from exc


def transform_image(key: str, image: Image.Image) -> torch.Tensor:
    transform = TASKS[key]["transform"]
    if transform is None:
        raise ModelNotReady(f"{key} 未配置图像预处理")
    return transform(image.convert("RGB")).unsqueeze(0)


def run_classification(key: str, image: Image.Image) -> Dict[str, Any]:
    """执行分类推理，返回统一结构；模型不可用时抛 ModelNotReady。

    除了预测类别，还会做一次「非本模型样本」判定，结果放在 ``valid`` 和 ``ood``：
    ``valid=False`` 表示这张图不该被这个模型诊断，调用方必须拒绝输出预测。
    """
    entry = get_model_entry(key)
    task = TASKS[key]
    tensor = transform_image(key, image).to(DEVICE)

    started = time.perf_counter()
    with torch.inference_mode():
        logits = entry.model(tensor)
        feature_fn = getattr(entry.model, "forward_features", None)
        features = feature_fn(tensor) if feature_fn is not None else None
        probs = F.softmax(logits, dim=1)[0]
    elapsed_ms = (time.perf_counter() - started) * 1000

    index = int(torch.argmax(probs).item())
    confidence = float(probs[index].item() * 100)
    probs_cn = _class_names(key, "cn")
    probs_en = _class_names(key, "en")

    # 非本模型样本判定：图像质量 + 置信度 + 特征空间马氏距离
    try:
        ood_result = get_guard(key).evaluate(
            features[0].float().cpu().numpy() if features is not None else None,
            logits[0].cpu().numpy(),
            probs.cpu().numpy(),
            image_rgb=np.asarray(image.convert("RGB")),
        )
    except Exception as exc:  # noqa: BLE001 - 判定失败不能拖垮正常推理
        print(f"[OOD] {key} 判定异常，按正常样本处理: {exc}")
        ood_result = {
            "status": ood_detection.STATUS_OK,
            "score": None,
            "scores": {},
            "reasons": [],
            "message": "",
            "hints": [],
            "quality": [],
            "calibrated": False,
        }

    return {
        "task": key,
        "index": index,
        "predict": probs_cn[index],
        "predict_cn": probs_cn[index],
        "predict_en": probs_en[index],
        "confidence": confidence,
        "pred_pros": [float(v) for v in probs.cpu().tolist()],
        "labels": probs_cn,
        "labels_en": probs_en,
        "model_type": task["model_type"],
        "elapsed_ms": round(elapsed_ms, 2),
        "model": model_meta(key, elapsed_ms),
        "valid": ood_result["status"] != ood_detection.STATUS_REJECTED,
        "ood": ood_result,
    }


def model_meta(key: str, elapsed_ms: Optional[float] = None) -> Dict[str, Any]:
    """返回响应体里的 model 字段，用于前端展示“真实模型推理”信息。"""
    task = TASKS[key]
    entry = REGISTRY._entries.get(key)  # 只读展示，不触发加载
    meta = {
        "key": key,
        "name": task["name"],
        "architecture": task["architecture"],
        "accuracy": task["accuracy"],
        "accuracy_text": f"{task['accuracy'] * 100:.2f}%" if task["accuracy"] else "—",
        "device": device_name(),
        "loaded": entry is not None,
        "checkpoint": entry.checkpoint.name if entry else None,
        "elapsed_ms": round(elapsed_ms, 2) if elapsed_ms is not None else None,
    }
    return meta


def preload_models(keys: Optional[List[str]] = None) -> Dict[str, str]:
    return REGISTRY.preload(keys)


# 便捷函数：给各蓝图使用
def mango_leaf_classes() -> List[str]:
    return _class_names("mango_leaf", "cn")


def class_list(key: str) -> List[Dict[str, str]]:
    return list(TASKS[key]["classes"])
