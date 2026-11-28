# -*- coding: utf-8 -*-
"""
叶片病灶分割 + 物种/疾病识别 推理后端
=====================================

模型
----
分割: UNet++（EfficientNet-B0 encoder）→ 单通道病灶 mask
分类: DINOv2 ViT-S/14 + 物种/疾病双头（训练时只微调最后 4 个 block）
      主干名、输入尺寸、TTA 开关全部从 checkpoint 里读，换骨干不用改这里

注意事项
--------
1. 分类看整叶原图，不做病灶 bbox 裁剪。病斑形态要结合叶形、叶脉和分布才能区分，
   裁剪会丢信息，也造成训练/推理分布不一致。
2. 预处理必须和训练 notebook 的 cls_eval_tf 完全一致：短边缩到 img_size，再中心裁成方图。
3. 联合概率只在「合法 物种×疾病」组合上归一化，置信度不再被外积压低。
4. 病灶 mask 做小连通域过滤，输出病斑个数与面积占比。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from ood_detection import get_guard

try:
    import timm
except ImportError:                                   # 只有加载分类模型时才真的需要
    timm = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("PLANT_MODELS_DIR", PROJECT_ROOT / "models"))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEG_CHECKPOINT = MODELS_DIR / "leaf_disease_segment_model.pth"
CLS_CHECKPOINT = MODELS_DIR / "leaf_disease_classify_model.pth"

MIN_LESION_PIXELS = 48          # 小于该像素数的连通域当作标注噪声丢掉
MAX_LESION_RATIO = 0.6          # 超过画面 60% 的 mask 基本是失败，按无效处理

SPECIES_CN = {
    "Corn": "玉米", "Grape": "葡萄", "Potato": "马铃薯",
    "Squash": "南瓜", "Tomato": "番茄",
    "Apple": "苹果", "Raspberry": "树莓", "Soybean": "大豆",
    "Strawberry": "草莓", "Blueberry": "蓝莓", "Cherry": "樱桃",
    "Peach": "桃", "Bell pepper": "甜椒",
}


class LeafModelNotReady(RuntimeError):
    """权重缺失或格式不对。"""


class LeafClassifier(nn.Module):
    """timm 主干 + 物种/疾病双头。

    结构和 train/LeafDiseaseSegmentAndClassify.ipynb §8 的 LeafClassifier 一模一样，
    否则 state_dict 对不上。推理时只有 forward 的前半段有意义，requires_grad 无所谓。
    """

    def __init__(self, backbone: str, num_species: int, num_diseases: int,
                 hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        # pretrained=False：权重全在 checkpoint 里，避免推理时联网下预训练模型
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        in_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.species = nn.Linear(hidden, num_species)
        self.disease = nn.Linear(hidden, num_diseases)

    def forward_features(self, images):
        """主干 + head 融合后的表示（hidden 维），OOD 判定用。"""
        return self.head(self.backbone(images))

    def forward(self, images):
        hidden = self.forward_features(images)
        return self.species(hidden), self.disease(hidden)


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                       # torch < 2.0 没有 weights_only
        return torch.load(path, map_location="cpu")


def _to_rgb(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise LeafModelNotReady(f"无法读取图片: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


_FRIENDLY_BACKBONE = {
    "vit_small_patch14_dinov2.lvd142m": "DINOv2 ViT-S/14",
    "vit_base_patch14_dinov2.lvd142m": "DINOv2 ViT-B/14",
    "vit_large_patch14_dinov2.lvd142m": "DINOv2 ViT-L/14",
    "efficientnet-b0": "EfficientNet-B0",
    "efficientnet-b1": "EfficientNet-B1",
}


def _friendly_backbone(name: str) -> str:
    """把 checkpoint 里的长骨干名换成界面上好读的写法（原名仍放在 tooltip 里）。"""
    return _FRIENDLY_BACKBONE.get(name, name)


_META: Dict[str, Any] = {}


def _checkpoint_meta() -> Dict[str, Any]:
    """只读 checkpoint 里的元信息，不建模型——页面首屏要显示物种/病害数量。

    读一次缓存住，88 MB 的文件不会每次建页面都读。
    """
    if _META:
        return _META
    if CLS_CHECKPOINT.exists():
        try:
            payload = _torch_load(CLS_CHECKPOINT)
            _META.update({
                "species": list(payload.get("species_names", [])),
                "diseases": list(payload.get("disease_names", [])),
                "classifier_backbone": payload.get("backbone"),
                "classifier_img_size": payload.get("img_size"),
                "arch": payload.get("arch"),
            })
        except Exception as exc:                        # noqa: BLE001
            print(f"[Leaf Disease] 读 checkpoint 元信息失败: {exc}")
    return _META


def _clean_mask(binary: np.ndarray) -> np.ndarray:
    """去掉小连通域，抑制标注噪声造成的碎斑。"""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_LESION_PIXELS:
            cleaned[labels == i] = 1
    return cleaned


class LeafDiseasePipeline:
    def __init__(self, device: torch.device = DEVICE):
        self.device = device
        self.seg_model = None
        self.cls_model = None
        self.species_names: List[str] = []
        self.disease_names: List[str] = []
        self.species_to_diseases: Dict[str, List[int]] = {}
        self.seg_size = 512
        self.cls_size = 518
        self.seg_encoder = "efficientnet-b0"
        self.cls_backbone = "vit_small_patch14_dinov2.lvd142m"
        self.cls_hidden = 256
        self.cls_dropout = 0.3
        self.cls_tta = True
        self.cls_mean = list(IMAGENET_MEAN)
        self.cls_std = list(IMAGENET_STD)
        self.arch = "dinov2_partial_finetune"
        self._valid: Optional[np.ndarray] = None
        self._loaded = False

    # ---------------------------------------------------------------- 加载权重
    def status(self) -> Dict[str, Any]:
        info = {
            "segment_exists": SEG_CHECKPOINT.exists(),
            "classifier_exists": CLS_CHECKPOINT.exists(),
            "loaded": self._loaded,
            "device": str(self.device),
            "species": self.species_names,
            "diseases": self.disease_names,
            "segment_backbone": self.seg_encoder,
            "segment_name": _friendly_backbone(self.seg_encoder),
            "classifier_backbone": self.cls_backbone,
            "classifier_name": _friendly_backbone(self.cls_backbone),
            "classifier_img_size": self.cls_size,
            "classifier_tta": self.cls_tta,
            "arch": self.arch,
            "segment_checkpoint": str(SEG_CHECKPOINT),
            "classifier_checkpoint": str(CLS_CHECKPOINT),
        }
        if not self._loaded:                    # 还没加载模型，退而求其次读 checkpoint 里的类名
            meta = _checkpoint_meta()
            if meta.get("species"):
                info["species"] = meta["species"]
                info["diseases"] = meta.get("diseases", [])
            for key in ("classifier_backbone", "classifier_img_size", "arch"):
                if meta.get(key):
                    info[key] = meta[key]
            info["classifier_name"] = _friendly_backbone(info["classifier_backbone"])
            info["segment_name"] = _friendly_backbone(info["segment_backbone"])
        return info

    def load(self):
        if self._loaded:
            return self
        if not SEG_CHECKPOINT.exists():
            raise LeafModelNotReady(
                f"未找到分割权重: {SEG_CHECKPOINT}\n请先运行 train/LeafDiseaseSegmentAndClassify.ipynb 的第 4 节。"
            )
        if not CLS_CHECKPOINT.exists():
            raise LeafModelNotReady(
                f"未找到分类权重: {CLS_CHECKPOINT}\n请先运行 train/LeafDiseaseSegmentAndClassify.ipynb 的第 9 节。"
            )
        if timm is None:
            raise LeafModelNotReady("未安装 timm，无法加载 DINOv2 分类模型：pip install timm")

        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise LeafModelNotReady("未安装 segmentation_models_pytorch，无法加载分割模型") from exc

        # ---------------------------------------------------------------- 分割
        seg_payload = _torch_load(SEG_CHECKPOINT)
        seg_config = seg_payload.get("config", {}) if isinstance(seg_payload, dict) else {}
        self.seg_encoder = seg_config.get("encoder", self.seg_encoder)
        self.seg_size = int(seg_config.get("img_size", self.seg_size))
        seg_state = seg_payload.get("model_state_dict", seg_payload)
        self.seg_model = smp.UnetPlusPlus(
            encoder_name=self.seg_encoder,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        ).to(self.device)
        self.seg_model.load_state_dict(seg_state)
        self.seg_model.eval()

        # ---------------------------------------------------------------- 分类
        cls_payload = _torch_load(CLS_CHECKPOINT)
        self.arch = cls_payload.get("arch", self.arch)
        self.species_names = list(cls_payload.get("species_names", []))
        self.disease_names = list(cls_payload.get("disease_names", []))
        if not self.species_names or not self.disease_names:
            raise LeafModelNotReady("分类权重缺少 species_names / disease_names，请用新 notebook 重新训练")

        self.cls_backbone = cls_payload.get("backbone", self.cls_backbone)
        self.cls_size = int(cls_payload.get("img_size", self.cls_size))
        self.cls_hidden = int(cls_payload.get("hidden", self.cls_hidden))
        self.cls_dropout = float(cls_payload.get("dropout", self.cls_dropout))
        self.cls_tta = bool(cls_payload.get("tta", self.cls_tta))

        # 新格式存的是 state_dict，老格式是 model_state_dict
        cls_state = cls_payload.get("state_dict") or cls_payload.get("model_state_dict")
        if cls_state is None:
            raise LeafModelNotReady("分类权重里没有 state_dict，请用新 notebook 重新训练")

        self.cls_model = LeafClassifier(
            backbone=self.cls_backbone,
            num_species=len(self.species_names),
            num_diseases=len(self.disease_names),
            hidden=self.cls_hidden,
            dropout=self.cls_dropout,
        ).to(self.device)
        missing, unexpected = self.cls_model.load_state_dict(cls_state, strict=False)
        if missing:
            raise LeafModelNotReady(
                f"分类权重和模型结构对不上，缺少 {len(missing)} 个参数，例如 {missing[:3]}"
            )
        self.cls_model.eval()

        # 归一化参数取主干自带的（DINOv2 就是 ImageNet 均值方差）
        try:
            data_cfg = timm.data.resolve_data_config(
                self.cls_model.backbone.pretrained_cfg,
                model=self.cls_model.backbone,
                verbose=False,
            )
            self.cls_mean = [float(v) for v in data_cfg.get("mean", IMAGENET_MEAN)]
            self.cls_std = [float(v) for v in data_cfg.get("std", IMAGENET_STD)]
        except Exception:                                   # noqa: BLE001
            self.cls_mean, self.cls_std = list(IMAGENET_MEAN), list(IMAGENET_STD)

        # ---------------------------------------------------------------- 物种 × 疾病 合法组合表
        self.species_to_diseases = {}
        for species, diseases in cls_payload.get("species_to_diseases", {}).items():
            indices = []
            for disease in diseases:
                if isinstance(disease, int):
                    indices.append(disease)
                elif disease in self.disease_names:
                    indices.append(self.disease_names.index(disease))
            self.species_to_diseases[species] = indices
        if not self.species_to_diseases:
            self.species_to_diseases = {
                s: list(range(len(self.disease_names))) for s in self.species_names
            }

        self._valid = np.zeros((len(self.species_names), len(self.disease_names)), dtype=bool)
        for s, species in enumerate(self.species_names):
            for d in self.species_to_diseases.get(species, []):
                self._valid[s, d] = True

        self._loaded = True
        return self

    # ---------------------------------------------------------------- 小工具
    def _seg_tensor(self, image_rgb: np.ndarray) -> torch.Tensor:
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.seg_size, self.seg_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return tf(image_rgb).unsqueeze(0).to(self.device)

    def _cls_tensor(self, image_rgb: np.ndarray) -> torch.Tensor:
        # 必须和 notebook 里的 cls_eval_tf 一致：Resize(短边) + CenterCrop(方图)。
        # 直接用 Resize((s, s)) 会把叶片横向拉扁，和训练分布对不上。
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.cls_size),
            transforms.CenterCrop(self.cls_size),
            transforms.ToTensor(),
            transforms.Normalize(self.cls_mean, self.cls_std),
        ])
        return tf(image_rgb).unsqueeze(0).to(self.device)

    @staticmethod
    def _bbox(mask: np.ndarray, padding: float = 0.06):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return [0, 0, 0, 0]
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        w, h = x2 - x1 + 1, y2 - y1 + 1
        pad_x, pad_y = int(w * padding), int(h * padding)
        return [
            max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(mask.shape[1] - 1, x2 + pad_x), min(mask.shape[0] - 1, y2 + pad_y),
        ]

    # ---------------------------------------------------------------- 推理
    # 分割：输出 0/1 mask，并按连通域过滤掉噪点，顺带给出病斑个数与面积占比
    def segment(self, image_rgb: np.ndarray):
        tensor = self._seg_tensor(image_rgb)
        with torch.no_grad():
            logits = self.seg_model(tensor)
            prob = torch.sigmoid(logits).cpu().numpy().squeeze()
        prob = cv2.resize(prob, (image_rgb.shape[1], image_rgb.shape[0]))
        binary = _clean_mask((prob > 0.5).astype(np.uint8))

        ratio = float(binary.mean())
        if ratio > MAX_LESION_RATIO:            # 大面积误检，直接判为无效
            binary = np.zeros_like(binary)
            ratio = 0.0

        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        lesions = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)
                   if stats[i, cv2.CC_STAT_AREA] >= MIN_LESION_PIXELS]
        return {
            "mask": binary,
            "mask_prob": prob,
            "lesion_ratio": round(ratio * 100, 2),
            "lesion_count": len(lesions),
            "bbox": self._bbox(binary),
        }

    # 分类：整叶原图 → 物种/疾病概率，再把非法组合置零、重新归一化
    def classify(self, image_rgb: np.ndarray):
        tensor = self._cls_tensor(image_rgb)
        with torch.no_grad():
            hidden = self.cls_model.forward_features(tensor)
            species_logits = self.cls_model.species(hidden)
            disease_logits = self.cls_model.disease(hidden)
            if self.cls_tta:                    # 原图 + 水平翻转取平均，和 §10 测试一致
                flip_hidden = self.cls_model.forward_features(torch.flip(tensor, dims=[3]))
                species_logits = (species_logits + self.cls_model.species(flip_hidden)) / 2
                disease_logits = (disease_logits + self.cls_model.disease(flip_hidden)) / 2
            species_probs = F.softmax(species_logits, dim=1)[0].float().cpu().numpy()
            disease_probs = F.softmax(disease_logits, dim=1)[0].float().cpu().numpy()

        # 非本模型样本判定：图像质量 + 物种头置信度 + hidden 特征的马氏距离
        try:
            ood_result = get_guard("leaf_disease").evaluate(
                hidden[0].float().cpu().numpy(),
                species_logits[0].float().cpu().numpy(),
                species_probs,
                image_rgb=image_rgb,
            )
        except Exception as exc:  # noqa: BLE001 - 判定异常不能拖垮正常推理
            print(f"[OOD] leaf_disease 判定异常，按正常样本处理: {exc}")
            ood_result = {
                "status": "ok", "score": None, "scores": {}, "reasons": [],
                "message": "", "hints": [], "quality": [], "calibrated": False,
            }

        joint = np.outer(species_probs, disease_probs)
        if self._valid is not None:
            joint = np.where(self._valid, joint, 0.0)
        total = float(joint.sum())
        if total > 0:
            joint = joint / total

        s_idx, d_idx = np.unravel_index(np.argmax(joint), joint.shape)

        # 候选：按联合概率取前 3 个合法 (物种, 疾病) 组合
        flat = joint.ravel()
        top_idx = np.argsort(-flat)[:3]
        top_pairs = []
        for flat_i in top_idx:
            s, d = divmod(int(flat_i), joint.shape[1])
            if flat[flat_i] <= 0:
                continue
            top_pairs.append({
                "species": self.species_names[s],
                "species_cn": SPECIES_CN.get(self.species_names[s], self.species_names[s]),
                "disease": self.disease_names[d],
                "probability": round(float(flat[flat_i]) * 100, 2),
            })

        return {
            "species_index": int(s_idx),
            "species": self.species_names[s_idx],
            "species_cn": SPECIES_CN.get(self.species_names[s_idx], self.species_names[s_idx]),
            "species_confidence": round(float(species_probs[s_idx]) * 100, 2),
            "disease_index": int(d_idx),
            "disease": self.disease_names[d_idx],
            "disease_confidence": round(float(disease_probs[d_idx]) * 100, 2),
            "joint_confidence": round(float(joint[s_idx, d_idx]) * 100, 2),
            "top_pairs": top_pairs,
            "species_probs": [round(float(v) * 100, 2) for v in species_probs],
            "disease_probs": [round(float(v) * 100, 2) for v in disease_probs],
            "valid": ood_result["status"] != "rejected",
            "ood": ood_result,
        }

    # 一次完整的「分割 + 分类」，并画好 overlay 供前端直接展示
    def predict(self, image_path: Path):
        started = time.perf_counter()
        image_rgb = _to_rgb(image_path)

        seg = self.segment(image_rgb)
        result = self.classify(image_rgb)           # 整叶原图

        binary = seg["mask"]
        overlay = image_rgb.copy()
        overlay[binary > 0] = (255, 66, 66)
        overlay = cv2.addWeighted(image_rgb, 0.65, overlay, 0.35, 0)
        contours, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (34, 197, 94), 2)
        x1, y1, x2, y2 = seg["bbox"]
        if x2 > x1 and y2 > y1:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (34, 197, 94), 2)

        result.update({
            "mask": binary,
            "overlay": overlay,
            "bbox": seg["bbox"],
            "lesion_ratio": seg["lesion_ratio"],
            "lesion_count": seg["lesion_count"],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        return result


_PIPELINE: Optional[LeafDiseasePipeline] = None


def get_pipeline() -> LeafDiseasePipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = LeafDiseasePipeline().load()
    return _PIPELINE


def pipeline_status() -> Dict[str, Any]:
    """已经加载过就报真实状态，否则只报权重文件在不在（不触发加载）。"""
    if _PIPELINE is not None:
        return _PIPELINE.status()
    return LeafDiseasePipeline().status()
