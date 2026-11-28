# -*- coding: utf-8 -*-
"""叶片病灶分割 + 物种/疾病识别 —— 预测页面 + 推理接口。

路由
----
GET  /leaf_disease/test_predict        预测页面（首屏会读模型状态）
GET  /leaf_disease/model_info          模型状态 JSON
POST /leaf_disease/submit_and_predict  上传图片 → 病灶 mask + 物种/疾病 + 防治建议

模型
----
分割：UNet++（EfficientNet-B0），权重 ``models/leaf_disease_segment_model.pth``
分类：DINOv2 ViT-S/14 微调 + 双头，权重 ``models/leaf_disease_classify_model.pth``

"""

import base64
from pathlib import Path

import cv2
from flask import Blueprint, jsonify, redirect, render_template, request, session

from leaf_disease_knowledge import DISEASE_CN, DISEASE_INFO
from leaf_disease_models import (
    CLS_CHECKPOINT,
    DEVICE,
    LeafDiseasePipeline,
    LeafModelNotReady,
    get_pipeline,
    pipeline_status,
)
from ood_detection import rejection_payload, success_fields
from user import save_upload_record
from upload_storage import save_uploaded_file


def _png_data_url(image) -> str:
    """把 mask（单通道 0/255）编成 PNG data URL，前端直接放 <img src>。"""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("图片编码失败")
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _jpg_data_url(image_rgb) -> str:
    """把 RGB overlay 编成 JPEG data URL；质量 90 兼顾体积和清晰度。"""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("图片编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _status():
    """模型状态：已经加载过就报真实信息（物种/病害列表），否则只报权重文件是否存在。"""
    return pipeline_status()


def leaf_disease_blueprint():
    blueprint = Blueprint("leaf_disease", __name__)

    # ---------------------------------------------------------------- 预测页面
    @blueprint.route("/test_predict")
    def test_predict():
        # 未登录直接回登录页
        if "login_name" not in session:
            return redirect("/")
        return render_template("test_predict_leaf_disease.html", model_status=_status())

    # ---------------------------------------------------------------- 模型状态
    @blueprint.route("/model_info")
    def model_info():
        return jsonify(_status())

    # ---------------------------------------------------------------- 推理接口
    @blueprint.route("/submit_and_predict", methods=["POST"])
    def submit_and_predict():
        if "login_name" not in session:
            return jsonify({"error": "登录已过期，请重新登录"}), 401

        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "请选择需要识别的图片"}), 400

        extension = Path(uploaded.filename).suffix.lower()
        if extension not in {".png", ".jpg", ".jpeg"}:
            return jsonify({"error": "仅支持 PNG、JPG、JPEG 图片"}), 400

        # 先落盘再推理：即使后面推理失败，上传记录也留得住
        image_path, upload_url, filename = save_uploaded_file(uploaded, "leaf_disease")

        try:
            pipeline = get_pipeline()
            result = pipeline.predict(image_path)
        # 权重没准备好：503 + 指向 notebook 的提示
        except LeafModelNotReady as exc:
            return jsonify({
                "error": str(exc),
                "hint": "请先运行 train/LeafDiseaseSegmentAndClassify.ipynb 训练分割与分类模型。",
                "model": _status(),
            }), 503
        except Exception as exc:  # noqa: BLE001
            print(f"[Leaf Disease] 推理失败: {exc}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"识别失败: {exc}"}), 500

        # mask 内部是 0/1 的 uint8，乘 255 才是能直接显示的黑白图
        mask_png = (result["mask"] * 255).astype("uint8")
        mask_url = _png_data_url(mask_png)
        overlay_url = _jpg_data_url(result["overlay"])

        # 不是作物叶片样本：不给诊断结论，也不写进识别记录
        if not result["valid"]:
            print(
                f"[Leaf Disease] 拒绝非本模型样本 · "
                f"{'；'.join(result['ood']['reasons']) or result['ood']['message']}"
            )
            return jsonify(rejection_payload(
                result["ood"],
                {
                    "device": str(DEVICE),
                    "segmentation": f"UNet++ · {pipeline.seg_encoder}",
                    "classifier": f"{pipeline.cls_backbone} · 双头",
                    "checkpoint": CLS_CHECKPOINT.name,
                },
                {"upload_image": upload_url},
            ))

        disease_cn = DISEASE_CN.get(result["disease"], result["disease"])
        # 写库失败只记日志，不影响本次响应
        try:
            save_upload_record(
                user_name=session["login_name"],
                image_name=filename,
                image_path=upload_url,
                disease_class=f"{result['species_cn']} - {disease_cn}",
                model_type="叶片病灶分割 + 物种/疾病识别",
                confidence=result["joint_confidence"],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Leaf Disease] 保存记录失败: {exc}")

        info = DISEASE_INFO.get(result["disease"], {
            "cause": "暂无该病害的详细病因说明。",
            "advice": ["建议结合当地农技部门建议进行防治。"],
        })

        response = {
            **success_fields(result["ood"]),
            "upload_image": upload_url,
            "mask_base64": mask_url,
            "overlay_base64": overlay_url,
            "species": result["species"],
            "species_cn": result["species_cn"],
            "species_confidence": result["species_confidence"],
            "disease": result["disease"],
            "disease_cn": disease_cn,
            "disease_confidence": result["disease_confidence"],
            "joint_confidence": result["joint_confidence"],
            "top_pairs": result["top_pairs"],
            "cause": info["cause"],
            "advice": info["advice"],
            "species_labels": [pipeline_species_cn(pipeline, s) for s in pipeline.species_names],
            "species_labels_raw": pipeline.species_names,
            "species_probs": result["species_probs"],
            "disease_labels": pipeline.disease_names,
            "disease_labels_cn": [DISEASE_CN.get(d, d) for d in pipeline.disease_names],
            "disease_probs": result["disease_probs"],
            "lesion_ratio": result["lesion_ratio"],
            "lesion_count": result["lesion_count"],
            "bbox": result["bbox"],
            "elapsed_ms": result["elapsed_ms"],
            "model": {
                "device": str(DEVICE),
                "segmentation": f"UNet++ · {pipeline.seg_encoder}",
                "classifier": f"{pipeline.cls_backbone} · 双头",
                "species_count": len(pipeline.species_names),
                "disease_count": len(pipeline.disease_names),
                "checkpoint": CLS_CHECKPOINT.name,
            },
        }
        print(
            f"[Leaf Disease] {result['species']} / {result['disease']} · "
            f"联合置信度 {result['joint_confidence']:.2f}% · "
            f"病灶 {result['lesion_count']} 处 {result['lesion_ratio']:.2f}% · {result['elapsed_ms']:.0f} ms"
        )
        return jsonify(response)

    return blueprint


def pipeline_species_cn(pipeline, name: str) -> str:
    from leaf_disease_models import SPECIES_CN
    return SPECIES_CN.get(name, name)
